<div align="center">

# SAFIR

### Task-Interactive Infrared–Visible Fusion With Frequency Division and Conditional Smoke Correction

<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0"><img src="https://img.shields.io/badge/release-weights--v0.2.0-1f8acb.svg" alt="weights release"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors"><img src="https://img.shields.io/badge/checkpoint-SAFIR__final.safetensors-2e8b57.svg" alt="final checkpoint"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/blob/main/docs/WEIGHT_FORMAT.md"><img src="https://img.shields.io/badge/format-safetensors-f39c12.svg" alt="safetensors format"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/blob/main/docs/DEPENDENCIES.md"><img src="https://img.shields.io/badge/model%20source-withheld-6c757d.svg" alt="model source withheld"></a>

**TIFDFusion backbone · task-guided frequency division · bounded conditional smoke correction**

<p>
  <a href="#overview">Overview</a> ·
  <a href="#visual-overview">Visual overview</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#release-boundary">Release boundary</a> ·
  <a href="#repository-map">Repository map</a>
</p>

</div>

> **Reproducibility status.** This repository is a public weight, protocol, and audit release. It supports checkpoint verification and tensor inspection. The executable model implementation and a standalone inference runtime are not included yet.

## Overview

SAFIR combines the original **TIFDFusion** backbone with a conditional smoke-correction expert. TIFDFusion exchanges semantic and detection evidence with the visible–infrared fusion features at multiple scales and separates low-frequency structure from high-frequency detail. SAFIR then accepts an externally supplied condition:

- `normal` returns the frozen backbone output exactly;
- `smoke` applies a bounded residual correction and clips the result to `[0, 1]`.

The condition is supplied by the caller. SAFIR does not infer it from the input images in the released A32 controlled candidate.

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

The following qualitative panels are taken from the paper's fixed smoke comparisons. They illustrate the fused outputs and the learned correction response; they are visual examples rather than a replacement for the reported quantitative evaluation.

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

### Download the final checkpoint

Download **[`SAFIR_final.safetensors`](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors)** from the [weights-v0.2.0 release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0). The release also contains `manifest.json`, `tensor_index.json`, and `SHA256SUMS`.

### Verify the file

From the repository root:

```bash
python tools/verify_weights.py --root .
```

The verifier checks the release file bytes against the published SHA256. It does not import the model or run inference.

### Inspect the retained tensor state

After installing compatible PyTorch and `safetensors` packages:

```bash
python tools/read_component_weights.py --component safir_final
```

This command reports the consolidated state group without importing any architecture code.

## What the checkpoint contains

`SAFIR_final.safetensors` is one consolidated snapshot of the final **SAFIR-A32-Controlled** candidate:

- 807 retained project-component tensors, including inherited frozen modules, auxiliary heads, alignment projections, and buffers;
- native final-model state keys under the `state_dict.` prefix;
- exact dtype, shape, and raw tensor-byte preservation relative to the verified source checkpoints;
- 719 excluded third-party SegFormer/YOLO backbone tensors, which remain external dependencies.

The file is a complete retained state snapshot. It is not a pruned minimum inference checkpoint and it does not contain the model implementation, optimizer metadata, dataset assets, or third-party prior checkpoints.

## Release boundary

The repository intentionally separates public research records from the private model implementation.

| Available here | Not included in this release |
| --- | --- |
| Final consolidated weight file | Model architecture source |
| Weight manifest, tensor index, and checksums | A loader for the consolidated file |
| Archived training and inference entry scripts | Standalone executable inference runtime |
| Method and protocol documentation | Dataset images, annotations, and downstream evaluator checkpoints |
| External dependency and licensing notes | Matching SegFormer/YOLO prior checkpoints |

The archived scripts preserve their original bytes and document the historical research workflow. Their imports, paths, and checkpoint checks refer to the private research workspace; they are not a ready-to-run inference package for a fresh clone. The current release therefore does not claim end-to-end local reproduction of the paper metrics.

## Evaluation context

The manuscript evaluates visible–infrared pairs from MSRS, FMB, and M3FD under clean and synthetic medium/heavy smoke, with TNO and RoadScene as an extension. Clean inputs use `normal`; smoke inputs use `smoke`; infrared inputs remain unchanged during smoke synthesis. Dataset assets, downstream evaluator checkpoints, and comparison-method outputs are outside this repository.

## Repository map

| Path | Purpose |
| --- | --- |
| `docs/assets/` | Public architecture figures used in this README |
| `weights/` | Final checkpoint notes and machine-readable tensor metadata |
| `tools/verify_weights.py` | File-level checksum verification |
| `tools/read_component_weights.py` | Architecture-free tensor inspection |
| `reference_scripts/` | Byte-preserved historical training and inference entry points |
| `docs/WEIGHT_FORMAT.md` | Consolidated state format and loading boundary |
| `docs/TRAINING_AND_TESTING.md` | Historical stage map and reproduction limits |
| `docs/DEPENDENCIES.md` | Third-party prior dependencies and release terms |

## Archived training stages

The A211, A213, A215, A217, A219, and A223 names identify historical training stages in the inherited chain; they are not six independent models that users need to download. The public release presents the final retained state as one checkpoint. The original stage provenance remains documented in the private audit records.

## Release terms

No repository-wide reuse license has been assigned at this stage. Third-party components retain their own terms. Please cite the accompanying manuscript when using the released records or weights.
