# Checkpoint inventory

No model weight files are bundled in this initial release. `manifest.json` records the expected identities of the archived checkpoint chain; it does not provide download URLs or establish redistribution clearance.

The paper's existing inference loader reconstructs:

**TIFDFusion Base → A211 → A213 → A215 → A217 → A219 → A223 → A32 wrapper**

The intermediate checkpoints contain separately trained components. The final A223 checkpoint saves only its predictor; A32 is a parameter-free conditional wrapper. A single `final_epoch4.pth` is consequently not a complete SAFIR release.

Names under `weights/` in the manifest are the planned asset filenames for the original, byte-identical archives. Expected hashes refer to those original files. If a checkpoint is stripped of optimizer metadata, merged, or separated from third-party tensors, it becomes a new artifact and needs a new hash and a verified comparison with the frozen paper model.

Files must be matched by SHA256, not by a similar filename. In particular, the locally available historical backbone backup has not matched the canonical backbone hash and is not included as a substitute.

Checkpoint verification uses file bytes only and never deserializes PyTorch/pickle content. A matching checksum confirms identity against this inventory; it does not independently prove model behavior or absence of embedded third-party components.

For distribution, large approved weights can be attached to GitHub Releases while scripts and documentation remain in Git. GitHub blocks files above 100 MiB in ordinary repository storage; see [GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Release assets should be published only after the matching checkpoint files and release terms are confirmed.
