# Component weights

Download the seven `.safetensors` files from [weights-v0.1.0](https://github.com/eliaukyh17-droid/SAFIR/releases/tag/weights-v0.1.0), and place them in this directory. Large weight files are release attachments rather than Git-tracked files.

`manifest.json` records each new file's SHA256, size, state-group mapping, parent checkpoint hash, and download URL. `SHA256SUMS` lists published component-file hashes. `tensor_index.json` records the dtype, shape, and raw-byte hash of every retained tensor.

The archived loader's component chain is:

**TIFDFusion Base → A211 → A213 → A215 → A217 → A219 → A223 → A32 wrapper**

These seven files preserve the component states in that chain. They are not seven alternative models. The A223 original checkpoint stores only its final predictor; the parameter-free A32 wrapper has no separate trained weights.

The original seven `.pth` files were verified against the frozen paper hashes. For this release, their tensor groups were repackaged as safetensors, with 719 embedded pretrained SegFormer/YOLO backbone tensors excluded from Base. The 807 retained tensors, including trained alignment projections and all persistent buffers, were compared against the originals for identical dtype, shape, and raw bytes. All checks passed.

The new files have different hashes and contain no original training/report metadata. The archived runners require the old hashes, metadata, and full model state, so they cannot directly use these files. Model definitions and the loading adapter are not provided in this initial release. These assets do not support standalone training or inference. See [the format notes](../docs/WEIGHT_FORMAT.md) and [third-party dependencies](../docs/DEPENDENCIES.md).

Verify downloaded component files from the repository root:

```sh
python tools/verify_weights.py --root .
```

Inspect a component's tensor groups after installing compatible PyTorch and safetensors packages:

```sh
python tools/read_component_weights.py --component a213
```

The reader does not instantiate a model or generate a fused image. The tensor-level export checks do not establish forward-output equivalence or historical training reproducibility.
