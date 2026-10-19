# Final checkpoint format

`SAFIR_final.safetensors` uses format version `safir_final_state_v1` and manifest component ID `safir_final`. It consolidates the retained state under the native registered keys of the final A32 model. Every exported key is:

```text
state_dict.<native A32 registered state key>
```

The manifest declares one state group:

```json
{"original_checkpoint_group": "state_dict", "exported_key_prefix": "state_dict."}
```

The reader removes only the leading `state_dict.` to recover the native state dictionary. `tensor_index.json` records the exact exported keys. The group name describes this consolidated state, not a training-stage checkpoint format.

## Retained state

All 807 retained project-component tensors are included: frozen inherited components, auxiliary heads, trained alignment projections, and persistent buffers, including BatchNorm statistics/counters and standardization state. This preserves every retained tensor's dtype, shape, and raw data bytes without quantization or casting. It is a complete retained snapshot, not a pruned or minimum set for inference. Historical automatic-routing and diagnostic state remains present even where it does not affect explicit-mode final images.

The exclusions are the 719 pretrained backbone tensors in these two subtrees of the frozen canonical model, nested beneath its native A32 state path:

```text
prior_extractor.seg_extractor.backbone.
prior_extractor.det_extractor.backbone.
```

The container uses contiguous tensor storage. It does not preserve original checkpoint file bytes, Python objects, optimizer/report metadata, or storage alias relationships. The published SHA256 identifies the new consolidated file.

## Loading boundary

`tools/read_component_weights.py --component safir_final` verifies the file and recovers its tensor state for inspection. The utility retains its existing filename and contains no model definitions.

The archived research runners expect original checkpoint hashes, metadata, and a different loading sequence. They cannot directly consume this asset. Running SAFIR would require the withheld model implementation, a compatible loader, the exact matching external priors, and the relevant configuration/protocol assets. Restoring a runnable model requires checking the complete reconstructed key set; permissive loading or substitute prior values would not establish equivalence to the frozen paper model.

Validation of this export covers file integrity and retained tensor preservation. No model inference, training, or benchmark recomputation was performed, and no output-equivalence claim is made.
