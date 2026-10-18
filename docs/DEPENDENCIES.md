# Dependencies and third-party assets

The archived implementation uses PyTorch, wavelet operations, Mamba-related modules, and frozen semantic/detection priors. There is no verified standalone installation environment for this partial release: installing packages cannot replace the withheld model implementation and private loaders.

The frozen priors include SegFormer-B2 and YOLOv8s. These are distinct from the downstream segmentation/detection checkpoints used to score fused outputs.

| Component | Upstream source | Treatment in this initial release |
| --- | --- | --- |
| SegFormer-B2 Cityscapes prior | [NVIDIA model card](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024), [upstream license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | Link only; no source or checkpoint redistributed |
| YOLOv8s prior | [Ultralytics repository](https://github.com/ultralytics/ultralytics), [license information](https://www.ultralytics.com/license) | Link only; no source or checkpoint redistributed |

The upstream projects have their own terms. Any proposed release checkpoint must also be inspected for embedded third-party prior tensors; omitting separate prior files does not establish that a fusion checkpoint contains only the authors' weights. The approved asset inventory and its hashes must be updated if tensors are separated or a new checkpoint format is exported.

The reference scripts are supplied for source inspection. Their imports are not evidence that the imported files are part of this release. Third-party datasets and comparison-method implementations are outside this package.
