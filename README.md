<div align="center">

# SAFIR

### Task-Interactive Infrared–Visible Fusion With Frequency Division and Conditional Smoke Correction

<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0"><img src="https://img.shields.io/badge/release-weights--v0.2.0-1f8acb.svg" alt="weights release"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors"><img src="https://img.shields.io/badge/checkpoint-SAFIR__final.safetensors-2e8b57.svg" alt="final checkpoint"></a>
<a href="https://github.com/eliaukyh17-droid/SAFIR/blob/main/docs/WEIGHT_FORMAT.md"><img src="https://img.shields.io/badge/format-safetensors-f39c12.svg" alt="safetensors format"></a>

**TIFDFusion backbone · task-guided frequency division · bounded conditional smoke correction**

<p>
  <a href="#overview">Overview</a> ·
  <a href="#visual-overview">Visual overview</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#project-updates">Project updates</a> ·
  <a href="#repository-map">Repository map</a>
</p>

</div>

This repository provides SAFIR weights, reference training and inference scripts, method documentation, and paper visualizations.

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

### Download the final checkpoint

Download **[`SAFIR_final.safetensors`](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors)** from the [weights-v0.2.0 release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0). The release also contains `manifest.json`, `tensor_index.json`, and `SHA256SUMS`.

### Verify the file

From the repository root:

```bash
python tools/verify_weights.py --root .
```

The verifier checks the checkpoint against the published SHA256.

### Inspect the retained tensor state

After installing compatible PyTorch and `safetensors` packages:

```bash
python tools/read_component_weights.py --component safir_final
```

This command reports the consolidated tensor state.

## What the checkpoint contains

`SAFIR_final.safetensors` is one consolidated snapshot of the final **SAFIR-A32-Controlled** candidate:

- 807 retained project-component tensors, including inherited frozen modules, auxiliary heads, alignment projections, and buffers;
- native final-model state keys under the `state_dict.` prefix;
- exact dtype, shape, and raw tensor-byte preservation relative to the verified source checkpoints.

See the [checkpoint format](docs/WEIGHT_FORMAT.md) for tensor metadata and the [dependency notes](docs/DEPENDENCIES.md) for SegFormer/YOLO prior requirements.

## Project updates

We will continue updating this repository and upload additional project files and supporting materials in future releases.

## Evaluation context

The manuscript evaluates visible–infrared pairs from MSRS, FMB, and M3FD under clean and synthetic medium/heavy smoke, with TNO and RoadScene as an extension. Clean inputs use `normal`; smoke inputs use `smoke`; infrared inputs remain unchanged during smoke synthesis.

## Repository map

| Path | Purpose |
| --- | --- |
| `docs/assets/` | Public architecture figures used in this README |
| `weights/` | Final checkpoint notes and machine-readable tensor metadata |
| `tools/verify_weights.py` | File-level checksum verification |
| `tools/read_component_weights.py` | Tensor state inspection |
| `reference_scripts/` | Reference training and inference entry points |
| `docs/WEIGHT_FORMAT.md` | Consolidated checkpoint format |
| `docs/TRAINING_AND_TESTING.md` | Training stages and inference script documentation |
| `docs/DEPENDENCIES.md` | Third-party prior dependencies and release terms |

## Archived training stages

The A211, A213, A215, A217, A219, and A223 names identify historical training stages. Their retained project-component weights are consolidated into the final checkpoint.

## Release terms

No repository-wide reuse license has been assigned at this stage. Third-party components retain their own terms. Please cite the accompanying manuscript when using the released records or weights.
