# Third-party notices

The model source retains attribution for adapted implementations from [CDTFusion](https://github.com/wangwenbo26/CDTFusion) and [S4Fusion](https://github.com/zipper112/S4Fusion):

| Local source | Upstream source |
| --- | --- |
| `models/encoder.py` | CDTFusion `util/encoder.py`; S4Fusion `modules/block.py` and `modules/layers.py` |
| `models/decoder.py` | CDTFusion `util/res_decoder.py` |
| SSIM implementation in `models/losses.py` | CDTFusion `util/loss_ssim.py` |

The upstream license check inspected CDTFusion at [fc169830](https://github.com/wangwenbo26/CDTFusion/tree/fc1698303798380086a10593aa834730f92762fe) and S4Fusion at [0d900f06](https://github.com/zipper112/S4Fusion/tree/0d900f06ced076ccd2e0b16bbcb4539f89cf5dfa). Neither inspected tree provided a repository license file. S4Fusion's README describes its planned training-code release for academic and research purposes. Retain the source comments and consult the upstream authors for applicable reuse terms. This repository does not grant additional rights to adapted third-party code.

Other dependencies include:

| Component | Upstream |
| --- | --- |
| PyTorch / TorchVision | https://github.com/pytorch/pytorch |
| Mamba selective scan | https://github.com/state-spaces/mamba |
| Causal Conv1d | https://github.com/Dao-AILab/causal-conv1d |
| PyTorch Wavelets | https://github.com/fbcotter/pytorch_wavelets |
| Transformers / SegFormer | https://github.com/huggingface/transformers |
| SegFormer-B2 pretrained prior | https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024 |
| Ultralytics / YOLOv8 | https://github.com/ultralytics/ultralytics |
| Segmentation Models PyTorch | https://github.com/qubvel-org/segmentation_models.pytorch |

Ultralytics is distributed under AGPL-3.0, with commercial licensing available from its authors. Third-party source, software packages, pretrained checkpoints, and datasets retain their respective terms. The download utility retrieves pretrained priors from their original hosts. This repository does not assign a new license to those materials.
