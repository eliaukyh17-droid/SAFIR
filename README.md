# SAFIR

**Task-Interactive Infrared–Visible Fusion With Frequency Division and Conditional Smoke Correction**

This repository accompanies the SAFIR manuscript. SAFIR combines our TIFDFusion backbone with a conditional smoke-correction expert. Semantic and detection features guide separate wavelet branches. An externally supplied `normal` or `smoke` condition selects the backbone output or bounded residual correction; the system does not estimate this condition from the images.

## Release scope

The release provides **one consolidated final-model checkpoint**, [SAFIR_final.safetensors](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors), for the frozen **SAFIR-A32-Controlled** candidate, together with archived training and inference entry scripts. **Model implementation source is withheld at this stage.**

The checkpoint preserves all 807 retained project-component state tensors, including frozen inherited components, auxiliary heads, and buffers. It is a complete retained snapshot, not a pruned or minimum inference state. The 719 embedded pretrained SegFormer/YOLO backbone tensors are excluded and remain external dependencies. The asset is not a standalone runnable model.

The ten archived source scripts retain their original bytes. Their imports, paths, and defaults refer to the research workspace and require model definitions, loaders, configurations, and protocol records outside this release. They cannot independently train SAFIR or generate fused images, and cannot directly load the new checkpoint format. End-to-end reproduction is not currently supported.

## Download and verify

Download `SAFIR_final.safetensors` from [weights-v0.2.0](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0) into `weights/`, then run from the repository root:

```sh
python tools/verify_weights.py --root .
```

The verifier requires Python 3.9 or newer and reports missing or mismatched files with a nonzero exit status. Verification covers file integrity and retained tensor preservation; no forward-inference, training-reproduction, or benchmark test was performed for this export. See [checkpoint notes](weights/README.md) for the inspection command and release assets.

## Files

| Path | Contents |
| --- | --- |
| `reference_scripts/` | Ten unchanged archived training and inference entries |
| `weights/manifest.json` | Final checkpoint format, file hash, size, and download link |
| `weights/tensor_index.json` | Exported keys, dtypes, shapes, and raw-byte hashes |
| `weights/SHA256SUMS` | Final checkpoint file checksum |
| `tools/verify_weights.py` | Independent file checksum verifier |
| `tools/read_component_weights.py` | Tensor-state reader; no architecture imports |
| `docs/WEIGHT_FORMAT.md` | Native state-key format and loading boundary |
| `docs/TRAINING_AND_TESTING.md` | Archived script map and reproduction limits |
| `docs/DEPENDENCIES.md` | External prior dependencies and release boundaries |

## Evaluation context

The manuscript evaluates registered visible–infrared pairs from MSRS, FMB, and M3FD under clean and synthetic medium/heavy smoke, with TNO and RoadScene as an extension. Clean inputs use `normal`; smoke inputs use `smoke`. Infrared inputs remain unchanged during smoke synthesis. Dataset assets, downstream evaluator checkpoints, and comparison-method outputs are outside this repository. The archived entries alone do not reproduce the reported measurements.

SAFIR expands to *Smoke-Activated Fusion with Identity-Preserving Residuals*.

## Release terms and manuscript information

Code and weight release terms have not yet been selected; no repository-wide reuse license is assigned. Third-party components retain their own terms. The author list, manuscript link, and citation metadata will be added after confirmation.
