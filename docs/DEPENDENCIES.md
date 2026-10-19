# Dependencies and third-party assets

The archived implementation uses PyTorch, wavelet operations, Mamba-related modules, and frozen semantic/detection priors. There is no verified standalone installation environment for this partial release: packages cannot replace the withheld model implementation, loaders, and protocol assets.

The final model requires SegFormer-B2 and YOLOv8s priors. These differ from the downstream evaluator checkpoints used to score fused outputs.

| Prior | Upstream source | Release treatment |
| --- | --- | --- |
| SegFormer-B2 Cityscapes | [NVIDIA model card](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024), [upstream license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | External dependency; no source or checkpoint redistributed |
| YOLOv8s | [Ultralytics repository](https://github.com/ultralytics/ultralytics), [license information](https://www.ultralytics.com/license) | External dependency; no source or checkpoint redistributed |

`SAFIR_final.safetensors` excludes 719 embedded pretrained tensors from the canonical model's `prior_extractor.seg_extractor.backbone.*` and `prior_extractor.det_extractor.backbone.*` subtrees. The trained alignment projections and all other retained project-component state remain included. Exclusion from distribution does not remove these priors from the model's computation. The consolidated checkpoint is therefore not self-contained.

The inspected SegFormer prior file has SHA256 `584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe`, matching the official model-file metadata checked for this release. The inspected YOLOv8s prior file has SHA256 `268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d`; an exact upstream download with identical bytes has not been independently established. An arbitrary replacement prior does not establish equivalence to the frozen model.

Third-party projects retain their own terms. Separately obtaining their checkpoints does not override those terms. No repository-wide reuse license has been assigned to this release.

Dataset assets, downstream evaluator checkpoints, and comparison-method implementations are outside this package. Imports in the archived scripts do not imply that the imported files are included.
