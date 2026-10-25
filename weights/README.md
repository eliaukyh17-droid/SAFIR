# Final SAFIR checkpoint

From the repository root:

```bash
python -m tools.download_weights
python tools/verify_weights.py --root .
```

The downloader obtains [SAFIR_final.safetensors](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors) and the exact official pretrained priors. The final checkpoint is a release attachment; model source and loading code are included in this repository.

The final file contains 807 project-component tensors. The loader restores 719 additional prior tensors from SegFormer-B2 and YOLOv8s. A211 through A223 are consolidated in this single file; no separate stage downloads are required for inference. The explicit-condition wrapper adds no learned parameters.

| File | Purpose |
| --- | --- |
| `manifest.json` | File hash, format, download URL, and current source integration status |
| `tensor_index.json` | Tensor keys, shapes, dtypes, and raw-byte hashes |
| `SHA256SUMS` | Final checkpoint checksum |

The binary and tensor index are unchanged from weights-v0.2.0. Release-attached metadata describes the original weight export; the repository manifest additionally records the current model-source integration. See [format](../docs/WEIGHT_FORMAT.md), [installation](../docs/INSTALL.md), and [validation](../docs/VALIDATION.md).
