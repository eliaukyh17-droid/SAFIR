"""Prepare smoke inputs, infer SAFIR/Base, and evaluate paper fusion metrics."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml
from safir.infer import paired_images


def run(*args):
    command=[sys.executable,*map(str,args)]
    print(' '.join(command),flush=True)
    subprocess.run(command,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/benchmark.yaml')
    p.add_argument('--datasets',nargs='+',default=['msrs','fmb','m3fd'])
    p.add_argument('--checkpoint',default='weights/SAFIR_final.safetensors')
    p.add_argument('--custom-checkpoint',action='store_true',help='Evaluate a final checkpoint exported by safir.train')
    p.add_argument('--segformer',default='pretrained/segformer-b2')
    p.add_argument('--yolo',default='pretrained/yolov8s.pt')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--scan-backend',default='auto',choices=['auto','cuda','reference'])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['datasets']
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use an empty directory for a new benchmark')
    rows=[];summary=[]
    for dataset in args.datasets:
        spec=config[dataset]
        pairs=paired_images(spec['visible'],spec['infrared'],spec['manifest'])
        if len(pairs)!=spec['count']:
            raise ValueError(f'{dataset}: expected {spec["count"]}, found {len(pairs)}')
        smoke_root=root/'inputs'/dataset
        run('tools/generate_smoke_dataset.py','--source-root',spec['root'],
            '--output-root',smoke_root,'--dataset-name',dataset.upper(),
            '--split','test','--vi-subdir',spec['vi_subdir'],'--ir-subdir',spec['ir_subdir'],
            '--manifest',spec['manifest'],'--morphology','patchy','--base-seed','20260820',
            '--severities','medium','heavy')
        clean_base=root/'outputs'/dataset/'clean'/'base'
        for condition in ['clean','medium','heavy']:
            if condition=='clean':
                vi,ir=Path(spec['visible']),Path(spec['infrared'])
            else:
                vi,ir=smoke_root/'test'/condition/'vi',smoke_root/'test'/condition/'ir'
            output=root/'outputs'/dataset/condition
            run('-m','safir.infer','--visible',vi,'--infrared',ir,'--manifest',spec['manifest'],
                '--condition','normal' if condition=='clean' else 'smoke',
                '--checkpoint',args.checkpoint,'--segformer',args.segformer,'--yolo',args.yolo,
                '--device',args.device,'--scan-backend',args.scan_backend,'--output',output,'--save-base',
                *(['--custom-checkpoint'] if args.custom_checkpoint else []))
            for method,fused in [('SAFIR',output),('TIFDFusion',output/'base')]:
                metrics=root/'metrics'/dataset/condition/method
                run('-m','safir.evaluate','--fused',fused,'--visible',vi,'--infrared',ir,
                    '--manifest',spec['manifest'],'--metrics','21','--clean-base',clean_base,'--output',metrics)
                record=json.loads((metrics/'summary.json').read_text())
                summary.append({'dataset':dataset,'condition':condition,'method':method,'count':record['count'],**record['mean']})
            for sample,v,i in paired_images(vi,ir,spec['manifest']):
                rows.extend([{'dataset':dataset,'severity':condition,'modality':mod,'sample':sample,'path':str(path.resolve())}
                             for mod,path in [('vi',v),('ir',i)]])
    with (root/'input_manifest.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    with (root/'fusion_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(f'Completed: {root / "fusion_summary.csv"}')


if __name__=='__main__':main()
