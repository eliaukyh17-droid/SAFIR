# Runtime and pretrained dependencies

The fusion environment is pinned in `requirements.txt`; installation commands are in [INSTALL.md](INSTALL.md). Downstream segmentation uses a separate environment in `requirements-evaluation.txt` because its SMP dependency requires a different `timm` version.

## Frozen task priors

`python -m tools.download_weights` downloads these exact official assets:

| Prior | Source | SHA256 |
| --- | --- | --- |
| SegFormer-B2 Cityscapes | [Hugging Face revision d633b20](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/tree/d633b2072669ca68d8f8e309de9b52bfdbf6bf72) | `584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe` |
| YOLOv8s COCO | [Ultralytics assets v8.1.0](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | `268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d` |

SegFormer configuration SHA256: `3bafe3a84c87270732bb383f6ef33801d1cd441907da8cedc2c8a4986276945d`.

The priors supply 719 tensors; the published SAFIR file supplies 807 project tensors. The public loader checks the reconstructed model's complete key set and rejects missing priors. It requires the Haar wavelet implementation from `pytorch-wavelets`.

The segmentation/detection **evaluation models** are separate from these priors; see [EVALUATION.md](EVALUATION.md). Download datasets from their original providers and arrange them using the documented paths. Pretrained priors, software packages, and datasets retain their respective terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
