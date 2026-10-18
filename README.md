# SAFIR

**Task-Interactive Infrared–Visible Fusion With Frequency Division and Conditional Smoke Correction**

This repository accompanies the SAFIR manuscript. SAFIR combines our TIFDFusion backbone with a conditional smoke-correction expert. Semantic and detection features guide separate wavelet branches in the backbone. An externally supplied `normal` or `smoke` condition selects direct return of the backbone output or bounded residual correction. The system does not estimate this condition from the images.

## Initial release

This first stage provides archived training and inference entry scripts, documentation, and the expected hashes of the paper's checkpoint chain. **Checkpoint files are pending; no download assets are published yet. Model implementation source is withheld at this stage.**

The entry scripts require model definitions, auxiliary loaders, configurations, and protocol records that are outside this release. They are provided for source inspection and cannot independently train a model or generate fused images. The checksum utility runs independently using Python 3.9 or newer. End-to-end reproduction is not currently supported.

The manuscript uses the frozen **SAFIR-A32-Controlled** evaluation candidate and its final A2.23 expert predictor. Archived experiment identifiers are retained for provenance. SAFIR expands to *Smoke-Activated Fusion with Identity-Preserving Residuals*.

## Files

| Path | Contents |
| --- | --- |
| `reference_scripts/` | Original training and inference entry scripts |
| `weights/manifest.json` | Expected identities of the seven archived component checkpoints; files pending |
| `tools/verify_weights.py` | File-byte checksum verifier; no model imports or checkpoint deserialization |
| `docs/TRAINING_AND_TESTING.md` | Script map, stage dependencies, and reproduction limits |
| `docs/DEPENDENCIES.md` | Third-party prior sources and release boundaries |

The ten source scripts retain their original bytes. Their relative paths, imports, and command defaults refer to the archived research workspace. They are not installation commands for this partial release.

## Checkpoints

The final A2.23 checkpoint contains a predictor state dictionary rather than the entire SAFIR network. The archived loader reconstructs the backbone and preceding components before loading this predictor; see [the checkpoint notes](weights/README.md).

Once the corresponding assets are provided, check their original file hashes with:

```sh
python tools/verify_weights.py --root .
```

At present the verifier reports seven missing files and exits with status 1. A matching hash would confirm archived file identity only; it would not validate tensor contents or execute the model.

## Evaluation context

The primary evaluation uses registered visible–infrared pairs from MSRS, FMB, and M3FD under clean and synthetic medium/heavy smoke. TNO and RoadScene form an extension using the same frozen model. Infrared inputs remain unchanged during smoke synthesis. Dataset images, annotations, comparison-method outputs, and third-party checkpoints are outside this repository.

Under the reported protocol, clean inputs use `normal` and smoke inputs use `smoke`. Source-based fusion measures, frozen downstream evaluators, and the supplementary FMB uniform-chroma analysis serve distinct purposes. These entry scripts alone do not reproduce the reported measurements.

## Release terms and manuscript information

Code and weight release terms have not yet been selected; no repository-wide reuse license is assigned at this stage. Third-party components retain their own terms. The author list, manuscript link, and citation metadata will be added after confirmation.
