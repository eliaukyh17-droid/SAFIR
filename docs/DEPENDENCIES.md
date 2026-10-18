# Dependencies and third-party assets

The archived implementation uses PyTorch, wavelet operations, Mamba-related modules, and frozen semantic/detection priors. There is no verified standalone installation environment for this partial release: installing packages cannot replace the withheld model implementation and private loaders.

The frozen priors include SegFormer-B2 and YOLOv8s. These are distinct from the downstream segmentation/detection checkpoints used to score fused outputs.

| Component | Upstream source | Treatment in this initial release |
| --- | --- | --- |
| SegFormer-B2 Cityscapes prior | [NVIDIA model card](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024), [upstream license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | Link only; no source or checkpoint redistributed |
| YOLOv8s prior | [Ultralytics repository](https://github.com/ultralytics/ultralytics), [license information](https://www.ultralytics.com/license) | Link only; no source or checkpoint redistributed |

The original Base checkpoint was inspected and contains both pretrained backbones. The component release excludes exactly `prior_extractor.seg_extractor.backbone.*` and `prior_extractor.det_extractor.backbone.*`: 719 tensors in total. It retains the trained alignment projections, all other fusion tensors, and persistent buffers. Original checkpoint metadata and the raw archive are not redistributed.

The upstream projects have their own terms. Separately obtaining the priors and excluding their tensors from these assets does not override those terms or establish that every possible integration meets them. No repository-wide license is assigned to the authors' components here. See the upstream license pages before using their models in an integration.

The inspected local SegFormer asset has SHA256 `584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe`, matching the official model-file metadata. The inspected local YOLOv8s file has SHA256 `268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d`; an exact current upstream download with identical bytes has not been independently established. Do not substitute an arbitrary prior checkpoint when comparing with the frozen model.

The reference scripts are supplied for source inspection. Their imports are not evidence that the imported files are part of this release. Third-party datasets and comparison-method implementations are outside this package.
