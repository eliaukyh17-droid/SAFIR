# Data preparation

Run commands from the repository root. Image inference accepts your own registered pairs; the paper benchmarks use the datasets, sample lists, and directory layout below.

## Dataset sources and sample lists

| Dataset | Original source | Supplied sample lists |
| --- | --- | --- |
| MSRS | [MSRS](https://github.com/Linfeng-Tang/MSRS) | `msrs_train_975.txt`, `msrs_val_108.txt`, `msrs_test_361.txt` |
| FMB | [SegMiF FMB Dataset](https://github.com/JinyuanLiu-CV/SegMiF#fmb-dataset) | `fmb_test_280.txt` |
| M3FD | [TarDAL M3FD Dataset](https://github.com/JinyuanLiu-CV/TarDAL#m3fd-dataset) | `m3fd_train_3150.txt`, `m3fd_test_1050.txt` |
| TNO | [TNO Image Fusion Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | `tno_test_37.txt` |
| RoadScene | [RoadScene](https://github.com/hanna-xu/RoadScene) | `roadscene_test_42.txt` |

All lists are under `manifests/`. Use the original providers' downloads and their accompanying dataset terms. MSRS supplies its images and segmentation labels through its repository; the FMB and M3FD project pages provide their dataset download links.

## Core dataset layout

The default paths in `configs/benchmark.yaml` expect:

```text
datasets/
  MSRS/
    train/
      vi/
      ir/
      Segmentation_labels/
    test/
      vi/
      ir/
  FMB/
    test/
      Visible/
      Infrared/
      Label/
  M3FD/
    M3FD_Detection/
      vi/
      ir/
      labels/
```

Filenames are paired by stem. Each pair must have the same dimensions. Preserve sample names, original image values, and geometric registration. Fusion inference retains native spatial resolution and handles its required padding internally.

### MSRS

Use the original 1,083 training pairs. `msrs_train_975.txt` and `msrs_val_108.txt` define disjoint internal training/validation populations; the 361 official test pairs are separate. Training reads the original segmentation masks from `train/Segmentation_labels/`. Its loader takes paired 256x256 crops and applies the documented training augmentation.

The inference and benchmark commands select the official test pairs with `msrs_test_361.txt`. Training schedules and stage-specific smoke generation are described in [TRAINING_AND_TESTING.md](TRAINING_AND_TESTING.md).

### FMB

Arrange the 280 test pairs in `test/Visible/` and `test/Infrared/`. Fusion scoring uses these images. Segmentation scoring additionally reads the corresponding original 15-class masks from `test/Label/` (IDs 0 through 14). If your download uses a different annotation folder name, pass that location with `--label_dir` or `--label-dir` in the evaluator command.

### M3FD

Download the **4,200-pair detection dataset** from TarDAL. That project also provides a separate 300-pair fusion dataset; the supplied 3,150/1,050 split refers to the detection collection.

Use the original YOLO-format TXT labels from the detection dataset. Each line contains `class_id center_x center_y width height`, with normalized coordinates. Class IDs follow this order:

| ID | Class |
| ---: | --- |
| 0 | People |
| 1 | Car |
| 2 | Bus |
| 3 | Motorcycle |
| 4 | Lamp |
| 5 | Truck |

Keep the supplied train/test manifests when preparing a YOLO dataset with `prepare_m3fd_detection.py`. The generated `val` directory contains the fixed 1,050-image held-out test population. Full scoring commands are in [EVALUATION.md](EVALUATION.md#m3fd-detection).

## TNO and RoadScene extensions

The paper uses prepared sets of **37 TNO pairs** and **42 RoadScene pairs**, named `001.png` onward in the supplied manifests. Their layout is:

```text
datasets/
  tno/
    vi/
    ir/
  roadscene/
    vi/
    ir/
```

The original source pages above describe the full collections. RoadScene contains 221 pairs, and the original TNO archive is organized by acquisition systems. The numbered prepared sets require the same image identities, registration, and dimensions as the paper's extension. The manifests record prepared filenames; the correspondence to original archive filenames is still being consolidated. Renumbering an arbitrary selection would change the evaluated population.

For a different TNO/RoadScene population, provide its paired directories, manifest, and count in a separate benchmark YAML and pass `--config path/to/benchmark.yaml`. Record that population alongside the resulting scores.

## Configure benchmark paths

Edit `configs/benchmark.yaml` if your dataset root differs. Each entry provides `root`, `visible`, `infrared`, the modality subdirectory names used by smoke generation, a manifest, and an expected pair count. Keep these fields consistent with the same input population.

For example, after preparing the three core datasets:

```bash
python -m safir.benchmark --config configs/benchmark.yaml --datasets msrs fmb m3fd --output runs/benchmark --device cuda:0 --scan-backend cuda
```

The benchmark materializes medium/heavy smoke with the fixed seed and morphology, then performs fusion inference and metric scoring. Downstream task commands consume those fusion outputs and the original task annotations; see [EVALUATION.md](EVALUATION.md).
