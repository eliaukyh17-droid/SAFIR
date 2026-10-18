# Component tensor format

The release uses `safir_component_tensors_v1` in seven safetensors files. Each key is the original checkpoint state-group name, a dot, and the original tensor key. For example, `standardizer_state_dict.fitted` maps to `checkpoint["standardizer_state_dict"]["fitted"]`. `state_groups` in the manifest provides the exact reversible mapping.

| Component | Original tensor groups retained |
| --- | --- |
| `canonical_tifdfusion` | `model_state_dict`, with the two pretrained backbone prefixes excluded |
| `a211` | `router_state_dict`, `expert_state_dict` |
| `a213` | `score_head_state_dict`, `standardizer_state_dict` |
| `a215` | `refiner_state_dict` |
| `a217` | `predictor_state_dict` |
| `a219` | `controller_state_dict` |
| `a223` | `predictor_state_dict` |

The only omitted model-state prefixes are:

```text
prior_extractor.seg_extractor.backbone.
prior_extractor.det_extractor.backbone.
```

The trained `align_projs` layers are retained. Persistent buffers are retained, including BatchNorm statistics/counters and A213's `mean`, `scale`, and boolean `fitted` state. Tensor values are not quantized or cast.

The export keeps each tensor's original dtype, shape, and raw data bytes. Contiguous storage is used in the new container; original Python objects, optimizer/report metadata, storage alias relationships, and checkpoint file bytes are not preserved. `original_checkpoint_sha256` identifies the original checkpoint file, while `sha256` identifies the published component file. These fields are not interchangeable.

## Loading boundary

`tools/read_component_weights.py` verifies a published file and reconstructs dictionaries of tensor groups. It contains no model definitions. Instantiating SAFIR, recovering its prior dependencies, and running fusion require implementation and protocol assets outside this release.

The original research loaders also check whole-file hashes and report fields before strict state loading. They are incompatible with the new safetensors assets. A future full loader would need to obtain the exact matching priors, restore the complete key set, and check all reconstructed states. Broad `strict=False`, random prior initialization, or zero-prior fallback would not establish equivalence to the frozen paper model.

The verification covers all 807 retained tensors and the exact 719-tensor exclusion boundary. No model inference, training, or benchmark recomputation was performed during this export. Separate upstream prior acquisition and this file split do not replace the upstream license terms.
