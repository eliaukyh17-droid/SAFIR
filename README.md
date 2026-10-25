<div align="center">

# SAFIR

### Task-Interactive Infrared–Visible Fusion With Frequency Division and Conditional Smoke Correction

<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0"><img src="https://img.shields.io/badge/release-weights--v0.2.0-1f8acb.svg" alt="weights release"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/tag/evaluators-v1.0.0"><img src="https://img.shields.io/badge/evaluators-v1.0.0-2e8b57.svg" alt="fixed evaluators release"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors"><img src="https://img.shields.io/badge/checkpoint-SAFIR__final.safetensors-2e8b57.svg" alt="final checkpoint"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/blob/main/docs/WEIGHT_FORMAT.md"><img src="https://img.shields.io/badge/format-safetensors-f39c12.svg" alt="safetensors format"></a>

**TIFDFusion backbone · task-guided frequency division · bounded conditional smoke correction**

<p>
  <a href="#overview">Overview</a> ·
  <a href="#visual-overview">Visual overview</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#data-preparation">Data</a> ·
  <a href="#training-and-evaluation">Training and evaluation</a> ·
  <a href="#project-updates">Project updates</a> ·
  <a href="#repository-map">Repository map</a>
</p>

</div>

This repository provides the SAFIR and TIFDFusion model source, pretrained weights, staged training, paired-image inference, smoke synthesis, and evaluation tools.

## Overview

SAFIR combines the original **TIFDFusion** backbone with a conditional smoke-correction expert. TIFDFusion exchanges semantic and detection evidence with the visible–infrared fusion features at multiple scales and separates low-frequency structure from high-frequency detail. SAFIR then accepts an externally supplied condition:

- `normal` returns the frozen backbone output exactly;
- `smoke` applies a bounded residual correction and clips the result to `[0, 1]`.

Inside the expert, localization support guides residual synthesis and a utility map controls its magnitude. Smooth and local-detail branches jointly bound the applied correction by 0.15 on the `[0, 1]` intensity scale. The caller supplies the condition when invoking the model.

## Visual overview

### SAFIR routing and smoke-expert interface

<p align="center">
  <img src="docs/assets/safir-routing-architecture.png" alt="SAFIR computation and condition interface" width="100%">
</p>

The routing diagram shows the identity-preserving normal path, the conditional smoke path, the learned support and utility maps, and the pointwise residual bound.

### TIFDFusion backbone architecture

<p align="center">
  <img src="docs/assets/tifdfusion-overall-architecture.png" alt="TIFDFusion overall architecture" width="100%">
</p>

The backbone uses independent visible and infrared encoders, four scale-specific TIFM blocks, frozen semantic and detection priors, and a coarse-to-fine fusion decoder.

### Frequency-specific task interaction in TIFDFusion

<p align="center">
  <img src="docs/assets/tifm-frequency-division.png" alt="TIFM frequency-specific task interaction and reconstruction" width="100%">
</p>

The TIFM detail view separates Haar low-frequency structure from high-frequency detail. Semantic evidence guides the low-frequency branch, while detection evidence corrects the high-frequency residual before inverse reconstruction.

## Output visualizations

The following qualitative panels are taken from the paper's fixed smoke comparisons. They illustrate the fused outputs and the learned correction response.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/assets/safir-output-comparison.png" alt="SAFIR output comparison under medium and heavy smoke" width="100%"><br>
      <sub>Medium/heavy smoke comparison across source images, external methods, TIFDFusion, and SAFIR.</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/assets/safir-mechanism-response.png" alt="SAFIR mechanism response maps and fused outputs" width="100%"><br>
      <sub>Smoke-visible input, inherited fusion, localization support, utility multiplier, applied correction, and SAFIR output.</sub>
    </td>
  </tr>
</table>


## Quick start

```bash
git clone https://github.com/eliaukyh17-droid/SAFIR.git
cd SAFIR
```

Follow **[Installation](docs/INSTALL.md)** to create the fusion environment. The documented GPU setup uses Linux, Python 3.10, PyTorch 2.4.1, CUDA 12.1, and the matching Mamba wheels. CPU checks use Python 3.12 and PyTorch 2.4.1. Run the remaining commands from the repository root.

Download the weights. The default command retrieves the final fusion model, frozen priors, and both downstream evaluators, with SHA256 checks:

```bash
python -m tools.download_weights
```

Place your registered visible/infrared image pairs in two directories, for example:

```text
inputs/
  visible/
    example.png
  infrared/
    example.png
```

Visible images are read as RGB and infrared images as grayscale. Pairs are matched by filename stem, so `example.jpg` can pair with `example.png`. Each pair must have the same spatial dimensions, with both dimensions at least 32 pixels.

Run a small clean-image example on CPU:

```bash
python -m safir.infer --visible inputs/visible --infrared inputs/infrared --condition normal --output runs/example_normal --device cpu --scan-backend reference
```

For smoke-degraded visible inputs, use `--condition smoke` and a new output directory. After installing the CUDA environment, use `--device cuda:0 --scan-backend cuda` for GPU inference. CPU inference uses the PyTorch recurrence and is slower on full-resolution scenes.

Results are original-resolution, 8-bit grayscale PNGs under `runs/example_normal/`. `inference.json` records the input/output hashes and run arguments. Each new run requires an empty output directory. Add `--save-base` to save the matched TIFDFusion output in `base/`. See [inference details](docs/TRAINING_AND_TESTING.md#inference) for the model API and exact image handling.

| Task | Entry point | Guide |
| --- | --- | --- |
| Fuse your image pairs | `python -m safir.infer` | [Inference and training](docs/TRAINING_AND_TESTING.md) |
| Train the backbone and smoke expert | `python -m safir.train` | [Stage schedules and commands](docs/TRAINING_AND_TESTING.md) |
| Generate smoke, infer, and score fusion outputs | `python -m safir.benchmark` | [Fusion benchmark](docs/EVALUATION.md#fusion-benchmark) |
| Evaluate existing fusion outputs | `python -m safir.evaluate` | [Metric definitions](docs/EVALUATION.md) |
| Score FMB semantic segmentation | `python eval_seg_fmb.py` | [FMB segmentation](docs/EVALUATION.md#fmb-segmentation) |
| Score M3FD object detection | `python eval_yolo_m3fd.py` | [M3FD detection](docs/EVALUATION.md#m3fd-detection) |

## Pretrained model

Download **[SAFIR_final.safetensors](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors)** from the [weights release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0), or use the downloader above. The single file contains the retained state of all fusion and smoke-adaptation stages. The loader restores the matching SegFormer and YOLO priors from their official checkpoints and checks the complete model state.

The frozen fusion priors are available directly from their official hosts:

| Prior | Official download | Local destination |
| --- | --- | --- |
| SegFormer-B2, Cityscapes | [Weights](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/resolve/d633b2072669ca68d8f8e309de9b52bfdbf6bf72/pytorch_model.bin), [configuration](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/resolve/d633b2072669ca68d8f8e309de9b52bfdbf6bf72/config.json) | `pretrained/segformer-b2/` |
| YOLOv8s, COCO | [Weights](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | `pretrained/yolov8s.pt` |

The FMB and M3FD task evaluators were trained separately for downstream scoring and are provided in the [evaluator release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/evaluators-v1.0.0). The download command supports these groups:

| Command | Files |
| --- | --- |
| `python -m tools.download_weights` | Fusion checkpoint, frozen priors, and both evaluators |
| `python -m tools.download_weights --only fusion` | `weights/SAFIR_final.safetensors` |
| `python -m tools.download_weights --only priors` | SegFormer-B2 weights/configuration and YOLOv8s prior |
| `python -m tools.download_weights --only evaluators` | `pretrained/evaluators/fmb_final.pth` and `m3fd_last.pt` |

Existing correct files are verified and reused. The [evaluation guide](docs/EVALUATION.md#fixed-downstream-evaluators) lists direct evaluator links, class counts, and checksums.

The release was checked against the original stage checkpoints: all **1,526 model tensors** match, and the reconstructed models produced identical outputs on the CPU comparison input. Real-image inference, fusion metrics, and one optimizer step for each smoke-adaptation stage were also checked. See [validation details](docs/VALIDATION.md).

## Data preparation

For your own images, use the paired directories in Quick start. For the paper benchmarks, obtain the original datasets and arrange them using the [data preparation guide](docs/DATASETS.md):

| Dataset | Source | Role in this project |
| --- | --- | --- |
| MSRS | [MSRS](https://github.com/Linfeng-Tang/MSRS) | 975 training pairs, 108 validation pairs, and 361 official test pairs |
| FMB | [SegMiF / FMB](https://github.com/JinyuanLiu-CV/SegMiF#fmb-dataset) | 280 test pairs for fusion and 15-class segmentation |
| M3FD | [TarDAL / M3FD](https://github.com/JinyuanLiu-CV/TarDAL#m3fd-dataset) | 4,200 detection pairs with the supplied 3,150/1,050 train/test split |
| TNO | [TNO Image Fusion Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | Extension using 37 prepared pairs |
| RoadScene | [RoadScene](https://github.com/hanna-xu/RoadScene) | Extension using 42 prepared pairs |

`manifests/` fixes the sample lists. `configs/benchmark.yaml` specifies the image directories. Downstream evaluation also requires the original FMB masks or M3FD YOLO-format labels. The TNO/RoadScene extensions use prepared subsets; the [data guide](docs/DATASETS.md#tno-and-roadscene-extensions) explains their naming and identity requirements.

## Training and evaluation

### Evaluate the released model

After preparing the datasets, run the clean/medium-smoke/heavy-smoke fusion benchmark in the fusion environment:

```bash
python -m safir.benchmark --datasets msrs fmb m3fd --output runs/benchmark --device cuda:0 --scan-backend cuda
```

This command generates deterministic smoke inputs, saves SAFIR and TIFDFusion outputs, and computes the 21 fusion metrics plus clean-backbone recovery measures. Clean inputs use `normal`; smoke inputs use `smoke`. It writes:

- `runs/benchmark/outputs/<dataset>/<condition>/`: fused PNGs and `base/` comparison outputs;
- `runs/benchmark/metrics/`: per-image scores and dataset means;
- `runs/benchmark/fusion_summary.csv`: the combined fusion-results table;
- `runs/benchmark/input_manifest.tsv`: condition-matched inputs for downstream processing.

Continue with the FMB segmentation and M3FD detection commands in the [evaluation guide](docs/EVALUATION.md#fixed-downstream-evaluators). Those steps use a separate evaluation environment because its segmentation package requires a different `timm` version. The guide also covers the supplementary FMB uniform-chroma evaluation. Add `tno roadscene` to `--datasets` after preparing the extension subsets.

### Train a new model

Training uses the supplied MSRS 975/108 split. Start with the three-phase TIFDFusion backbone, then train the six adaptation stages in order. A211 through A223 identify successive components consolidated into the final inference checkpoint. On Linux with the CUDA environment:

```bash
python -m safir.train --stage base --msrs-root datasets/MSRS --output runs/base --device cuda:0 --scan-backend cuda
previous=base
for stage in a211 a213 a215 a217 a219 a223; do
  python -m safir.train --stage "$stage" --msrs-root datasets/MSRS --previous-stage --checkpoint "runs/$previous/model.safetensors" --output "runs/$stage" --device cuda:0 --scan-backend cuda
  previous="$stage"
done
```

The final exported model is `runs/a223/SAFIR_final.safetensors`. To benchmark this newly trained checkpoint, pass its path explicitly:

```bash
python -m safir.benchmark --checkpoint runs/a223/SAFIR_final.safetensors --custom-checkpoint --datasets msrs fmb m3fd --output runs/benchmark_retrained --device cuda:0 --scan-backend cuda
```

The [training guide](docs/TRAINING_AND_TESTING.md#training-stages) provides the epochs, seeds, stage roles, checkpoint selection, and instructions for rerunning an individual stage. [Validation](docs/VALIDATION.md) records the release checks completed to date.

## Project updates

- Model definitions, checkpoint loading, staged training, inference, smoke synthesis, and evaluation code are now available.
- The original fixed FMB segmentation and M3FD detection evaluator checkpoints are available with verified SHA256 values.
- We will continue adding project files and supporting evaluation assets.

## Repository map

| Path | Purpose |
| --- | --- |
| `models/` | TIFDFusion, TIFM, encoders/decoder, and SAFIR smoke expert |
| `safir/` | Public model loader and inference/training/evaluation commands |
| `data/` | Paired-image loading and deterministic smoke synthesis |
| `configs/`, `manifests/` | Training schedules, benchmark paths, and sample splits |
| `utils/` | Fusion metrics and training protocol utilities |
| `weights/` | Checkpoint metadata and download information |
| `docs/assets/` | Architecture figures and qualitative outputs from the paper |
| `reference_scripts/` | Original archived training/inference runners |
| `tests/` | Numerical, data, and checkpoint-loading checks |

## Attribution

Please cite the accompanying SAFIR manuscript when using this work. Third-party software and pretrained priors retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). No repository-wide reuse license has been assigned yet.
