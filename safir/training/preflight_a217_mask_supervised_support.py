"""Research training functions, extracted without changes to their bodies.

Use python -m safir.train for portable dataset paths and checkpoint handling.
"""

from __future__ import annotations

import torch

def predictor_inputs(model, visible, infrared):
    router = model.source_model.source_model.router
    with torch.no_grad():
        output = router(visible, infrared)
        evidence = router.source_router.evidence(visible, infrared)
    return evidence, output["features"]
