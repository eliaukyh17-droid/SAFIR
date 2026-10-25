# Inference and training

Install the fusion environment and download weights as described in [INSTALL.md](INSTALL.md). Dataset download sources and directory layouts are in [DATASETS.md](DATASETS.md). Run commands from the repository root.

## Inference

```bash
python -m safir.infer --visible path/to/visible --infrared path/to/infrared --condition smoke --output runs/predictions --device cuda:0 --scan-backend cuda --save-base
```

Use `normal` for the original TIFDFusion output and `smoke` for correction. Visible inputs are RGB; infrared inputs are grayscale. Files are paired by filename stem, and both images must have the same dimensions. The loader applies right/bottom padding to multiples of 16 and removes it from the output. It does not resize the scene. PNG outputs use the research inference convention: clip to [0,1], multiply by 255, and truncate to uint8.

`--manifest manifests/msrs_test_361.txt` selects an exact sample list. The output directory must be empty. `inference.json` records input/output hashes, arguments, sizes, and loading information. `--save-base` adds a `base/` directory with matched TIFDFusion outputs.

The model API is:

```python
from safir import load_model

model = load_model(
    'weights/SAFIR_final.safetensors',
    segformer_path='pretrained/segformer-b2',
    yolo_path='pretrained/yolov8s.pt', device='cuda:0',
)
# visible: Bx3xHxW; infrared: Bx1xHxW; float32 [0,1], padded to multiples of 16.
fused = model(visible, infrared, degradation_condition='smoke')
```

## MSRS training data

Arrange the original dataset as follows:

```text
datasets/MSRS/
  train/vi/                 RGB visible images
  train/ir/                 infrared images
  train/Segmentation_labels/
  test/vi/
  test/ir/
```

The supplied manifests split the 1,083 training pairs into 975 training and 108 validation pairs; the 361 official test pairs are separate. Training crops are 256x256. Smoke generation and paired clean-input supervision are implemented in `data/`. Preserve the configuration seeds and sample lists for protocol comparisons.

## Training stages

| Stage | Role | Epochs | Batch | Seed |
| --- | --- | --- | --- | --- |
| `base` | TIFDFusion backbone, three phases | 50 + 30 + 20 | 8 | 2026 |
| `a211` | Source-conditioned proposal and routing | 5 | 8 | 20260911 |
| `a213` | Multiscale standardized routing | 5 | 8 | 20260913 |
| `a215` | Bounded source-preserving residual | 5 | 2 | 20260916 |
| `a217` | Smoke-supervised support | 5 | 2 | 20260918 |
| `a219` | Support/benefit controller | 5 | 2 | 20260920 |
| `a223` | Localization and correction utility | 5 | 2 | 20260924 |

These stage names identify successive components in the final model. The explicit `normal`/`smoke` wrapper adds no learned parameters. The backbone is frozen during smoke adaptation. Each adaptation stage initializes its new module and freezes its predecessor. Original training functions are retained in `safir/training/`; [training_source_map.json](training_source_map.json) records their source hashes and extracted function names.

Adaptation uses `drop_last=True`: A211/A213 process 968 pairs per epoch, and A215–A223 process 974. A213 fits feature standardization on training pairs before optimization. Its inherited smoke-generation seed and its stage optimizer seed are distinct and retained in `configs/a213.yaml`. A211 uses one router warm-up epoch, two oracle-support epochs, and two joint epochs. A215 uses two oracle-support and three frozen-support epochs. A217/A219/A223 fit their predictors for five epochs. Adaptation uses the final epoch; the backbone uses the internal validation split for phase checkpoint selection.

## Complete new training run

On Linux with the CUDA environment:

```bash
python -m safir.train --stage base --msrs-root datasets/MSRS --output runs/base --device cuda:0 --scan-backend cuda
previous=base
for stage in a211 a213 a215 a217 a219 a223; do
  python -m safir.train --stage "$stage" --msrs-root datasets/MSRS --previous-stage --checkpoint "runs/$previous/model.safetensors" --output "runs/$stage" --device cuda:0 --scan-backend cuda
  previous="$stage"
done
```

`--previous-stage` reads the preceding training run. Without that flag, the trainer extracts the frozen predecessor from the published final checkpoint; this is useful for rerunning an individual stage while keeping its source fixed.

```bash
python -m safir.train --stage a223 --msrs-root datasets/MSRS --checkpoint weights/SAFIR_final.safetensors --output runs/a223_from_released_source --device cuda:0 --scan-backend cuda
```

Each adaptation run writes its resolved configuration, loss history, stage checkpoint, and completion record. A223 also writes `SAFIR_final.safetensors` in the public inference format. Evaluate a newly trained final checkpoint with:

```bash
python -m safir.infer --checkpoint runs/a223/SAFIR_final.safetensors --custom-checkpoint --visible path/to/visible --infrared path/to/infrared --condition smoke --output runs/new_predictions --device cuda:0 --scan-backend cuda
```

`--custom-checkpoint` relaxes only the published file hash; tensor-key, shape, dtype, and prior checks remain active. New training runs depend on hardware, libraries, and stochastic optimization. [Validation](VALIDATION.md) records which release checks were executed.

## Archived scripts

`reference_scripts/` preserves the original experiment runners and their historical paths/checkpoint sequence. Use `python -m safir.train` and `python -m safir.infer` for the portable release commands.
