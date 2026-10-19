# Final SAFIR checkpoint

Download **[SAFIR_final.safetensors](https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors)** into this directory. The [weights-v0.2.0 release](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.2.0) contains four assets:

- `SAFIR_final.safetensors`: one consolidated checkpoint for the final frozen A32 model.
- `manifest.json`: format, size, file hash, and download information.
- `tensor_index.json`: exported tensor keys, dtypes, shapes, and raw-byte hashes.
- `SHA256SUMS`: the checkpoint file checksum.

The binary checkpoint is a release attachment, not a Git-tracked file. Use the metadata from the same release.

The checkpoint retains 807 project-component state tensors, including frozen inherited components, auxiliary heads, and persistent buffers. Some retained state supports historical automatic-routing or diagnostic behavior and does not affect the final explicit `normal`/`smoke` image. This is a complete retained snapshot, not a pruned inference minimum. The A32 wrapper adds no learned weights. The original A223 training checkpoint stored only its predictor and was not a complete SAFIR checkpoint.

The 719 embedded pretrained prior-backbone tensors are excluded. Model definitions and the matching third-party priors remain external requirements; this file cannot independently run fusion. See [format and loading notes](../docs/WEIGHT_FORMAT.md) and [dependencies](../docs/DEPENDENCIES.md).

From the repository root, verify the downloaded file:

```sh
python tools/verify_weights.py --root .
```

To inspect the retained state after installing compatible PyTorch and safetensors packages:

```sh
python tools/read_component_weights.py --component safir_final
```

The reader does not instantiate SAFIR or generate images. Checks cover file integrity and preservation of retained tensor dtypes, shapes, and raw bytes. No inference-equivalence or training-reproduction test was performed for this export.
