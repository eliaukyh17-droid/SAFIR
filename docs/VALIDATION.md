# Source-release validation

The source release was checked on Windows with Python 3.12.14, PyTorch 2.4.1+cpu, and the native PyTorch selective scan. Dependencies and installation commands are listed in [INSTALL.md](INSTALL.md). The original CUDA training environment is documented there separately.

## Checkpoint reconstruction and inference

[release_validation.json](release_validation.json) records the comparison against the original archived stage checkpoints:

| Check | Result |
| --- | --- |
| Reconstructed complete model state | 1,526 tensors compared; all equal |
| Project tensors in the consolidated checkpoint | 807 |
| Tensors restored from the pinned external priors | 719 |
| Reconstructed/original output on the CPU comparison input | Maximum absolute difference 0 |
| Adaptation optimization | One forward/backward/update step succeeded for each of A211, A213, A215, A217, A219, and A223 |
| Backbone optimization | Phase 1, batch 2, 64x64: finite loss/gradients and a successful encoder update; priors remained frozen |

The comparison includes tensor keys, shapes, dtypes, and values. The published weights-v0.2.0 binary is unchanged. Its provenance and tensor index are in [weights/](../weights/README.md).

The inference CLI was also run on one original-resolution MSRS image pair. The generated PNG was scored with all 21 fusion measures and the three clean-backbone recovery measures. Smoke-generation CLI checks used the same image pair. These checks exercise the executable paths; full dataset results require the corresponding datasets and evaluation runs in [EVALUATION.md](EVALUATION.md).

## Training-source and split checks

[training_source_map.json](training_source_map.json) records the archived experiment runners and the training functions included in the portable entry point. Original model and loss implementations are retained, with path/configuration handling provided by `safir.train`.

The MSRS manifests match the canonical experiment's saved split files byte for byte:

| Manifest | Samples | SHA256 |
| --- | ---: | --- |
| `msrs_train_975.txt` | 975 | `d74b324840f75aa4d8ce5a968b80fe5843c4469bb3021469538fcd4f5422e451` |
| `msrs_val_108.txt` | 108 | `be484710b46fd9b7b8c12176e355793ff53a0c32a5cd814847840b0e64746665` |

The two populations are disjoint and together contain 1,083 pairs. Both are disjoint from the supplied 361-pair official test list. The canonical saved configuration and source use the internal 108-pair validation split for backbone phase checkpoint selection. It is loaded without training augmentation or gradient updates. Smoke adaptation uses its documented final-epoch selection.

## Automated checks

```bash
python -m pip install pytest==8.3.3
python -m pytest -q
```

Tests cover smoke determinism, metric conventions, image pairing and native-resolution I/O, strict checkpoint restoration, selective-scan recurrence and gradients, and stage-checkpoint round trips.

On 2026-10-25, the suite completed with **39 passed, 1 skipped, 0 failed**, including the released-model integration test and download-integrity tests. The skipped test requires CUDA. Counts and the backbone step result are recorded in [release_validation.json](release_validation.json).

Set these environment variables to include the released-model integration test:

```text
SAFIR_TEST_CHECKPOINT=weights/SAFIR_final.safetensors
SAFIR_TEST_SEGFORMER=pretrained/segformer-b2
SAFIR_TEST_YOLO=pretrained/yolov8s.pt
```

The integration test verifies strict loading, the exact normal-path return, smoke-path execution, finite output, the 0.15 correction bound, and rejection of an unknown condition. The CUDA comparison runs when CUDA and `mamba-ssm` are available.

## Fixed downstream evaluator checks

The original FMB and M3FD checkpoints were checked against the final-evaluation protocol's SHA256 values. In the separate evaluation environment, the FMB state loaded strictly with 379 tensors and produced a finite `[1, 15, 64, 64]` output for a `[1, 3, 64, 64]` CPU input. The M3FD checkpoint loaded with the six expected classes and produced finite predictions. Environment versions, class names, and output dimensions are recorded in [evaluator_validation.json](evaluator_validation.json).

The [evaluator release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/evaluators-v1.0.0) provides both exact files. The download tests cover asset-group selection, manifest agreement, reuse of a correct existing file, and rejection of damaged downloads.

## Remaining evaluation runs

The release checks above used CPU inference and individual adaptation updates. A complete new seven-stage training run, a CUDA kernel comparison, full dataset fusion benchmarks, and fixed downstream scoring are subsequent validation runs. The original fixed downstream evaluator checkpoint filenames and their release status are listed in [EVALUATION.md](EVALUATION.md).
