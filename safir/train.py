"""Portable entry point for TIFDFusion and the six smoke-adaptation stages."""
import argparse
from importlib import import_module
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import yaml

from safir.model import build_model, FINAL_SHA256, sha256
from safir.training.a211_scjsfe_probe import seed_everything, validate_manifests

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['base', 'a211', 'a213', 'a215', 'a217', 'a219', 'a223']
SUFFIX = dict(a211='scjsfe_probe', a213='multiscale_standardized_router',
              a215='source_preserving_refiner', a217='mask_supervised_support',
              a219='dual_support_benefit_rescue', a223='dual_semantic_utility')
PREFIX = {'base': 'source_model.'*5+'base_model.', 'a211': 'source_model.'*5,
          'a213': 'source_model.'*5, 'a215': 'source_model.'*4,
          'a217': 'source_model.'*3, 'a219': 'source_model.'*2, 'a223': 'source_model.'}


def is_prior(key):
    return any('.prior_extractor.'+x+'.backbone.' in '.'+key for x in ['seg_extractor','det_extractor'])


def save_stage(model, path, stage):
    state = {k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items() if not is_prior(k)}
    save_file(state, str(path), metadata={'stage':stage,'format':'safir_training_stage_v1'})


def restore_predecessor(model, path, stage, released=False):
    if released:
        if sha256(path) != FINAL_SHA256:
            raise ValueError('Released initialization checkpoint SHA256 mismatch')
        prefix = 'state_dict.'+PREFIX[stage]
        state = {k[len(prefix):]:v for k,v in load_file(str(path)).items() if k.startswith(prefix)}
        if stage == 'a211':
            state = {k.replace('router.source_router.', 'router.', 1) if k.startswith('router.source_router.') else k:v
                     for k,v in state.items() if not k.startswith('router.') or k.startswith('router.source_router.')}
    else:
        with safe_open(str(path),framework='pt') as f:
            if (f.metadata() or {}).get('stage') != stage:
                raise ValueError(f'Expected a {stage} checkpoint')
        state = load_file(str(path))
    expected = model.state_dict()
    kept = {k for k in expected if not is_prior(k)}
    if state.keys() != kept:
        raise ValueError(f'Predecessor keys differ: missing={kept-state.keys()}, extra={state.keys()-kept}')
    for k,v in state.items():
        if v.shape != expected[k].shape or v.dtype != expected[k].dtype:
            raise ValueError(f'Predecessor tensor mismatch: {k}')
    state.update({k:v for k,v in expected.items() if is_prior(k)})
    model.load_state_dict(state, strict=True)


def add_stage(source, stage):
    if stage == 'a211':
        from models.a211_safe_joint_source_expert import SafeConditionalJointSourceFusion
        return SafeConditionalJointSourceFusion(source)
    if stage == 'a213':
        from models.a213_multiscale_standardized_router import install_a213_router, freeze_except_score_head
        install_a213_router(source)
        freeze_except_score_head(source)
        return source
    classes = {'a215':'A215CanonicalAnchoredFusion', 'a217':'A217MaskSupervisedSupportFusion',
               'a219':'A219DualSupportBenefitRescueFusion','a223':'A223DualSemanticUtilityFusion'}
    return getattr(import_module('models.'+stage+'_'+SUFFIX[stage]),classes[stage])(source)


def training_model(stage, checkpoint, segformer, yolo, backend='auto', released=True):
    predecessor = STAGES[STAGES.index(stage)-1]
    source = build_model(segformer_path=segformer, yolo_path=yolo, scan_backend=backend, through=predecessor)
    restore_predecessor(source, checkpoint, predecessor, released)
    source.requires_grad_(False)
    return add_stage(source, stage)


def train_base(cfg, args, output):
    from models.losses import TIFDLoss
    from data.dataset import get_train_loader, get_test_loader
    from torch.utils.tensorboard import SummaryWriter
    from safir.training.backbone import run_phase
    model = build_model(segformer_path=args.segformer, yolo_path=args.yolo,
                        scan_backend=args.scan_backend, through='base').to(args.device)
    criterion = TIFDLoss(pixel_adaptive_temperature=cfg['train']['pixel_adaptive_temperature']).to(args.device)
    msrs, tc = cfg['data']['msrs'], cfg['train']
    train = get_train_loader('msrs', msrs['root'], batch_size=tc['batch_size'], num_workers=args.workers,
                             img_size=tuple(msrs['img_size']), subset_file=msrs['train_manifest'], smoke_config=msrs['smoke'])
    val = get_test_loader('msrs', msrs['root'], batch_size=1, num_workers=args.workers,
                         img_size=tuple(msrs['img_size']), subset_file=msrs['val_manifest'],
                         split='train', has_label=True, smoke_config=msrs['validation_smoke'])
    epoch=0
    with SummaryWriter(str(output/'logs')) as writer:
        for phase in [1,2,3]:
            epoch = run_phase(model,train,val,criterion,tc[f'phase{phase}'],args.device,phase,epoch,
                              writer,str(output),smoke_supervision=tc.get('smoke_supervision'))
            # The research backbone reloads the selected validation checkpoint at each phase boundary.
            selected=torch.load(output/f'best_phase{phase}.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(selected['model_state_dict'],strict=True)
    save_stage(model,output/'model.safetensors','base')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=STAGES)
    p.add_argument('--msrs-root',required=True)
    p.add_argument('--segformer',default='pretrained/segformer-b2')
    p.add_argument('--yolo',default='pretrained/yolov8s.pt')
    p.add_argument('--checkpoint',default='weights/SAFIR_final.safetensors',help='Released weights, or the preceding stage when --previous-stage is used')
    p.add_argument('--previous-stage',action='store_true',help='Initialize from a preceding stage produced by this trainer')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--scan-backend',default='auto',choices=['auto','cuda','reference'])
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    args.device=torch.device(args.device)
    cfg=yaml.safe_load((ROOT/'configs'/f'{args.stage}.yaml').read_text(encoding='utf-8'))
    msrs=cfg['data']['msrs']
    msrs['root']=str(Path(args.msrs_root).resolve())
    for name in ['train_manifest','val_manifest']:
        msrs[name]=str(ROOT/'manifests'/Path(msrs[name]).name)
    validate_manifests(cfg)
    if args.stage == 'base':
        seed=cfg['train']['seed']
    else:
        key={'a211':'a211_scjsfe','a213':'a213_router'}.get(args.stage,args.stage+'_training')
        protocol=cfg[key]
        schedule=protocol['schedule']
        seed=schedule['seed']
        cfg['a211_scjsfe']['num_workers']=args.workers
    seed_everything(seed)
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory for a new training run')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    if args.stage == 'base':
        train_base(cfg,args,output)
        return
    model=training_model(args.stage,args.checkpoint,args.segformer,args.yolo,args.scan_backend,not args.previous_stage).to(args.device)
    params=[v for v in model.parameters() if v.requires_grad]
    expected=protocol.get('trainable_parameters')
    if expected is not None and sum(x.numel() for x in params)!=expected:
        raise RuntimeError('Trainable parameter count differs from the paper configuration')
    module=import_module('safir.training.'+args.stage+'_'+SUFFIX[args.stage])
    a211=cfg['a211_scjsfe']
    learning=schedule if args.stage!='a211' else protocol
    optimizer=torch.optim.AdamW(params,lr=learning['learning_rate'],weight_decay=learning['weight_decay'])
    def loader(epoch):
        fn=module.make_train_loader if args.stage in ['a211','a223'] else getattr(module,'make_'+args.stage+'_train_loader')
        return fn(cfg,protocol,epoch) if args.stage=='a211' else fn(cfg,a211,protocol,epoch)
    if args.stage=='a213':
        standardizer=module.fit_training_standardizer(model,loader(0),args.device)
        (output/'standardizer.json').write_text(json.dumps(standardizer,indent=2),encoding='utf-8')
    history=[]
    for epoch in range(schedule['epochs']):
        if args.stage=='a211':
            from utils.a211_protocol import stage_for_epoch
            stats=module.train_epoch(model,loader(epoch),optimizer,args.device,protocol,epoch,stage_for_epoch(epoch))
        elif args.stage=='a213':
            stats=module.train_epoch(model,loader(epoch),optimizer,args.device,protocol['loss'],epoch)
        elif args.stage=='a215':
            stats=module.train_epoch(model,loader(epoch),optimizer,args.device,cfg['a215_refiner'],protocol,epoch)
        else:
            stats=module.train_epoch(model,loader(epoch),optimizer,args.device,protocol,epoch)
        history.append({'epoch':epoch,**stats})
        save_stage(model,output/'model.safetensors',args.stage)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    if args.stage=='a223':
        from models.a32_prompt_selected_safe_residual import A32PromptSelectedSafeResidualFusion
        wrapped=A32PromptSelectedSafeResidualFusion(model)
        state={'state_dict.'+k:v.detach().cpu().contiguous().clone() for k,v in wrapped.state_dict().items() if not is_prior(k)}
        save_file(state,str(output/'SAFIR_final.safetensors'),metadata={'format':'safir_final_state_v1'})
    (output/'complete.json').write_text(json.dumps({'stage':args.stage,'epochs':len(history),'seed':seed,
        'initialization_sha256':sha256(args.checkpoint),'previous_stage':args.previous_stage,
        'torch':torch.__version__,'scan_backend':args.scan_backend,'device':str(args.device)},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
