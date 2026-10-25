"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

import torch

def benefit_target(model, visible, clean_visible, infrared) -> torch.Tensor:
    with torch.no_grad():
        teacher = model.source_model.source_model(
            clean_visible, infrared, route_mode="canonical"
        )
        canonical = model.source_model(visible, infrared, route_mode="canonical")
        forced = model.source_model(visible, infrared, route_mode="smoke")
        dimensions = (1, 2, 3)
        canonical_mae = (canonical - teacher).abs().mean(dim=dimensions)
        forced_mae = (forced - teacher).abs().mean(dim=dimensions)
    return (forced_mae < canonical_mae).float()
