# Training and testing script map

The scripts below are archived source references, not standalone commands. They retain their original names to avoid confusing the backbone, the final smoke expert, and the evaluation wrapper.

## Backbone

`reference_scripts/train.py` and `reference_scripts/test.py` are the TIFDFusion backbone entries. `test.py` is not the full SAFIR inference entry and cannot interpret an A2.23 predictor checkpoint as a backbone checkpoint.

The local training source alone does not establish the exact historical objectives, configuration, and software environment that produced the paper's frozen backbone. Those records still need reconciliation. It must not be presented as a verified from-scratch reproduction of the paper checkpoint.

## Smoke stages

| Archived entry | Role in the inherited model chain |
| --- | --- |
| `train_a211_scjsfe_probe.py` | Earlier source-aware smoke expert |
| `train_a213_multiscale_standardized_router.py` | Standardization and score-head stage |
| `train_a215_source_preserving_refiner.py` | Source-preserving residual refiner |
| `train_a217_mask_supervised_support.py` | Localization/support predictor |
| `train_a219_dual_support_benefit_rescue.py` | Inherited controller stage |
| `train_a223_dual_semantic_utility.py` | Final predictor training with gain-weighted utility supervision |

These entries reside in `reference_scripts/`. This table records inheritance in the archived loader, not six independent training commands and not six separate claims of novelty. Earlier phases also depend on configurations, reports, helpers, and intermediate artifacts that are absent from this initial release.

The final training entry saves a predictor state dictionary. The frozen backbone and preceding components must already exist. The A32 conditional wrapper introduces no additional learned parameters and has no independent trained checkpoint.

## Inference entries

| Entry under `reference_scripts/tools/` | Scope |
| --- | --- |
| `infer_a33_controlled.py` | Frozen MSRS/FMB/M3FD inference, explicit `normal`/`smoke` condition |
| `infer_a38_controlled.py` | Frozen TNO/RoadScene extension |

The original loader verifies hashes of checkpoints and protocol evidence. Some checks also bind configuration or source files. Merely providing a final weight file and deleting the model directory does not satisfy these dependencies. The supplied entries are therefore reference source, and are not accompanied by a misleading runnable installation command.

The model definitions, loader implementations that reconstruct the model, and architectural helpers are withheld. This release does not embed the model in a single-file script, wheel, bytecode, ONNX, or another executable representation.

Published component files use the safetensors format described in [WEIGHT_FORMAT.md](WEIGHT_FORMAT.md). The archived runners require original `.pth` file hashes, report metadata, and a strict full-model state, including pretrained prior tensors. They cannot directly consume the new component assets. `tools/read_component_weights.py` only recovers the tensor groups for inspection; it is not a fusion model or a replacement training/inference runner.

Full training/inference validation will require an explicitly chosen way of making the model executable, the correct component checkpoints, the relevant protocol assets, and a verified environment. No new model training, evaluation, or metric recomputation was performed to prepare this release.
