# Evaluation

Install the fusion environment in [INSTALL.md](INSTALL.md) and download the final weights and matching frozen priors. Dataset images and annotations are obtained from their original providers. Run commands from the repository root.

## Dataset populations

Edit the dataset paths in `configs/benchmark.yaml` to match your local layout. The supplied sample lists define the evaluation populations:

| Dataset | Test pairs | Manifest |
| --- | ---: | --- |
| MSRS | 361 | `manifests/msrs_test_361.txt` |
| FMB | 280 | `manifests/fmb_test_280.txt` |
| M3FD | 1,050 | `manifests/m3fd_test_1050.txt` |
| TNO | 37 | `manifests/tno_test_37.txt` |
| RoadScene | 42 | `manifests/roadscene_test_42.txt` |

M3FD uses a separate 3,150-pair training list. MSRS backbone and adaptation training use the 975/108 internal training/validation split described in [training instructions](TRAINING_AND_TESTING.md).

## Fusion benchmark

```bash
python -m safir.benchmark --datasets msrs fmb m3fd --output runs/benchmark --device cuda:0 --scan-backend cuda
```

Add `tno roadscene` to `--datasets` to include the extensions. The benchmark generates deterministic medium/heavy patchy smoke with base seed `20260820`, preserves the infrared files, and runs both SAFIR and its TIFDFusion backbone on clean and smoke inputs. Clean inputs use `normal`; smoke inputs use `smoke`. Use an empty output directory.

The output contains:

- `inputs/`: generated smoke inputs and their generation records;
- `outputs/<dataset>/<condition>/`: SAFIR PNGs, `base/` backbone PNGs, and inference records;
- `metrics/<dataset>/<condition>/<method>/`: per-image CSV and metric means;
- `input_manifest.tsv`: condition-matched input paths for downstream evaluation;
- `fusion_summary.csv`: dataset/condition/method means.

Inference preserves the original spatial resolution. Scoring uses saved 8-bit grayscale PNGs and condition-matched visible/infrared sources. Each dataset mean gives equal weight to every test image.

To score existing outputs:

```bash
python -m safir.evaluate --fused runs/predictions --visible path/to/visible --infrared path/to/infrared --manifest manifests/msrs_test_361.txt --metrics 21 --output runs/metrics
```

The 21 measures are `EN`, `SD`, `SF`, `AG`, `MI`, `SCD`, `VIF`, `Qabf`, `SSIM_mean`, `CC`, `Nabf`, `MS_SSIM`, `NMI`, `QNCIE`, `QY`, `QCB`, `Source_MSE`, `Source_NRMSE`, `Source_PSNR`, `EI`, and `CE`. `--metrics 9` selects the first nine. `SSIM_mean` is the average of the two source-to-fused SSIM values; the underlying historical `SSIM` function returns their sum. Source-error measures average the visible-to-fused and infrared-to-fused errors. Definitions are retained in `utils/metrics.py`, `utils/metrics_extended.py`, and `utils/metrics_a37_supplemental.py`.

Optional `--clean-base path/to/clean/base` adds `Recovery_MAE`, `Recovery_PSNR`, and `Recovery_SSIM` relative to clean-input TIFDFusion outputs. The benchmark supplies this reference automatically. These recovery measures and the source-based fusion measures use different references.

## Fixed downstream evaluators

Segmentation and detection use trained evaluation models that remain fixed across fusion methods and smoke conditions. They are separate from the SegFormer/YOLO feature priors downloaded for fusion. Download the exact original checkpoints from the [evaluator release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/evaluators-v1.0.0), or run:

```bash
python -m tools.download_weights --only evaluators
```

| Evaluator | Download | Local destination |
| --- | --- | --- |
| FMB DeepLabV3+, ResNet-50, 15 classes | [fmb_final.pth](https://github.com/eliaukyh17-droid/SAFIR/releases/download/evaluators-v1.0.0/fmb_final.pth) | `pretrained/evaluators/fmb_final.pth` |
| M3FD YOLOv8s, six classes | [m3fd_last.pt](https://github.com/eliaukyh17-droid/SAFIR/releases/download/evaluators-v1.0.0/m3fd_last.pt) | `pretrained/evaluators/m3fd_last.pt` |

Both files were produced by the project's reference task training and subsequently frozen for scoring all fusion methods. FMB training initializes the ResNet-50 encoder with ImageNet weights; M3FD training initializes YOLOv8s with COCO weights. The final evaluation protocol records these trained checkpoint identities:

```text
662fb68a5d58e1f49df303119a1ddf040a662c2ecd1cdd7155dda44111d224be  pretrained/evaluators/fmb_final.pth
84fa4b72831a52b01f93a7417410b4649c84e7185ca9dc6363bd29a7b924871c  pretrained/evaluators/m3fd_last.pt
```

The downloader checks these SHA256 values. [weights/evaluators.json](../weights/evaluators.json) records their original experiment paths, file sizes, class counts, and download URLs. The official SegFormer/YOLO prior links are listed in [DEPENDENCIES.md](DEPENDENCIES.md); those priors supply features inside the fusion model.

The following commands use these fixed checkpoints. Their training entry points are also provided as `train_seg_fmb.py` and `train_yolo_m3fd.py`.

Create a separate evaluation environment to keep the segmentation dependency's `timm` version separate from the fusion environment:

```bash
python -m venv .venv-eval
# Linux:
source .venv-eval/bin/activate
# Windows PowerShell: .\.venv-eval\Scripts\Activate.ps1
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-evaluation.txt
```

Use the CPU PyTorch index from [INSTALL.md](INSTALL.md) for a CPU environment.

### FMB segmentation

For the primary native-output evaluation, pass the saved fusion PNGs directly to the fixed evaluator. Grayscale outputs are expanded to three channels by the dataset loader. The evaluator applies its established ImageNet normalization, resizes images to 640x480, and uses nearest-neighbor interpolation for label masks.

```bash
python eval_seg_fmb.py --fused_dir runs/benchmark/outputs/fmb/medium --label_dir datasets/FMB/test/Label --checkpoint pretrained/evaluators/fmb_final.pth --method SAFIR --severity medium --batch_size 1 --workers 2 --device 0 --output_json runs/downstream/fmb_medium.json
```

Replace `Label` with the actual annotation directory. Run each clean/medium/heavy condition separately. The supplementary uniform-chroma sensitivity analysis combines fused luminance with the corresponding visible input's chroma:

```bash
python eval_seg_fmb_uniform_chroma.py --fused-dir runs/benchmark/outputs/fmb/medium --label-dir datasets/FMB/test/Label --input-manifest runs/benchmark/input_manifest.tsv --checkpoint pretrained/evaluators/fmb_final.pth --method SAFIR --condition medium --batch-size 8 --workers 2 --device 0 --output-json runs/downstream/fmb_medium_uniform_chroma.json --metrics-csv runs/downstream/fmb_uniform_chroma.csv
```

Both evaluators report 15-class mIoU, foreground 14-class mIoU, all-pixel accuracy, and frequency-weighted IoU.

### M3FD detection

The preparation script uses the 3,150/1,050 manifests. Supply fused training images and the chosen condition's fused test images, along with the original YOLO-format labels:

```bash
python prepare_m3fd_detection.py --train-fused-dir path/to/clean_training_fused --test-fused-dir runs/benchmark/outputs/m3fd/medium --label-dir datasets/M3FD/M3FD_Detection/labels --train-manifest manifests/m3fd_train_3150.txt --test-manifest manifests/m3fd_test_1050.txt --output-dir runs/detection_data --tag medium --link-mode copy
python eval_yolo_m3fd.py --data runs/detection_data/medium/dataset.yaml --weights pretrained/evaluators/m3fd_last.pt --method SAFIR --severity medium --imgsz 640 --batch 16 --device 0 --workers 2 --project runs/downstream/detection --name medium --output_json runs/downstream/m3fd_medium.json
```

Generate clean training fusion images with `python -m safir.infer`, the original M3FD image directories, `--manifest manifests/m3fd_train_3150.txt`, and `--condition normal`. Preparation calls the held-out test partition `val` for the YOLO API; the 1,050-image population remains the supplied test list. Class order is People, Car, Bus, Motorcycle, Lamp, Truck. The evaluator reports AP50, AP50:95, precision, and recall from all predictions. Display-only box limits in qualitative figures do not enter these metrics.

See [VALIDATION.md](VALIDATION.md) for the checks executed on this source release.
