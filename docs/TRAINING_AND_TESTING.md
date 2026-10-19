# Archived training and testing entries

The ten scripts below are unchanged source references. Their original filenames identify the historical research stages; they are not runnable training or testing commands for this partial release.

## Backbone

`reference_scripts/train.py` and `reference_scripts/test.py` are the TIFDFusion backbone entries. `test.py` is not the full SAFIR inference entry.

The available training source alone does not establish the exact historical objectives, configuration, and software environment that produced the frozen paper backbone. These records still require reconciliation; this is not a verified from-scratch reproduction recipe.

## Smoke-expert training references

| Entry under `reference_scripts/` | Historical role |
| --- | --- |
| `train_a211_scjsfe_probe.py` | Source-aware smoke proposal expert |
| `train_a213_multiscale_standardized_router.py` | Standardization and automatic-routing score head |
| `train_a215_source_preserving_refiner.py` | Source-preserving residual refiner |
| `train_a217_mask_supervised_support.py` | Localization/support predictor |
| `train_a219_dual_support_benefit_rescue.py` | Automatic-routing controller stage |
| `train_a223_dual_semantic_utility.py` | Final localization/utility predictor training |

This table describes archived training history, not separate models to download or a claim that every historical head affects the final image. In particular, the A213 score head and A219 controller do not determine the output image under the final explicit `normal`/`smoke` selection; their registered state is retained in the consolidated snapshot.

The original A223 training checkpoint stores only its predictor state. The final model also contains frozen inherited components. A32 introduces the explicit condition selection and no additional learned weights.

## Inference references

| Entry under `reference_scripts/tools/` | Scope |
| --- | --- |
| `infer_a33_controlled.py` | Frozen MSRS/FMB/M3FD inference with explicit `normal`/`smoke` condition |
| `infer_a38_controlled.py` | Frozen TNO/RoadScene extension |

These runners depend on withheld model definitions, auxiliary loaders, configurations, and protocol evidence. Their original checks bind checkpoint hashes and metadata, and sometimes configuration/source files. They cannot directly load the consolidated `SAFIR_final.safetensors` asset.

The public [state reader](../tools/read_component_weights.py) can inspect the final checkpoint but cannot run fusion. No model implementation is embedded in another executable format. See [WEIGHT_FORMAT.md](WEIGHT_FORMAT.md) for the loading boundary. No new model training, inference validation, or metric recomputation was performed to prepare this release.
