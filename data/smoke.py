"""Deterministic non-uniform smoke synthesis for SAFIR-Fuse.

The implementation follows the spatially varying atmospheric-scattering model

    I_s = I * t + A * (1 - t),  t = exp(-beta * rho)

where ``rho`` combines multi-scale smooth noise and local Gaussian smoke
plumes.  No ground-truth smoke parameters are consumed by the fusion model;
they are returned only for dataset auditing and controlled experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SmokeSeverity:
    beta: Tuple[float, float]
    density: Tuple[float, float]


@dataclass(frozen=True)
class SmokeMorphology:
    base_weight: float
    plume_weight: float
    plumes: Tuple[int, int]
    gamma: float
    target_mean: float
    density_floor: float = 0.0


SEVERITY_PRESETS: Dict[str, SmokeSeverity] = {
    "mild": SmokeSeverity((0.65, 0.80), (0.55, 0.65)),
    "medium": SmokeSeverity((1.10, 1.30), (0.75, 0.90)),
    "heavy": SmokeSeverity((1.60, 2.00), (0.90, 1.10)),
}


MORPHOLOGY_PRESETS: Dict[str, SmokeMorphology] = {
    # Sparse plumes with large clear regions.
    "local": SmokeMorphology(0.12, 0.88, (2, 4), 2.20, 0.22),
    # Main SAFIR-Fuse condition: smooth background plus non-uniform plumes.
    "patchy": SmokeMorphology(0.55, 0.45, (4, 6), 1.80, 0.35),
    # Broad diffuse smoke with weaker local variation.
    "diffuse": SmokeMorphology(0.82, 0.18, (3, 5), 1.15, 0.50, 0.08),
}


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand((), generator=generator).item()


def _randint(generator: torch.Generator, low: int, high_inclusive: int) -> int:
    return int(torch.randint(low, high_inclusive + 1, (), generator=generator).item())


def _normalize(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def _multiscale_density(height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    """Build smooth, non-periodic density using several random spatial scales."""
    density = torch.zeros(1, 1, height, width, dtype=torch.float32)
    for divisor, weight in ((32, 0.48), (16, 0.32), (8, 0.20)):
        gh = max(2, (height + divisor - 1) // divisor)
        gw = max(2, (width + divisor - 1) // divisor)
        noise = torch.rand((1, 1, gh, gw), generator=generator)
        noise = F.interpolate(noise, size=(height, width), mode="bicubic", align_corners=False)
        density.add_(weight * noise.clamp(0, 1))
    return _normalize(density)


def _add_local_plumes(
    density: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add soft elliptical smoke plumes without rectangular paste boundaries."""
    _, _, height, width = density.shape
    yy = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    plume_field = torch.zeros_like(density)
    for _ in range(count):
        cx = _uniform(generator, -0.05, 1.05)
        cy = _uniform(generator, -0.05, 1.05)
        sx = _uniform(generator, 0.10, 0.34)
        sy = _uniform(generator, 0.08, 0.28)
        strength = _uniform(generator, 0.35, 1.0)
        plume = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        plume_field = torch.maximum(plume_field, strength * plume)
    return _normalize(plume_field)


def _smoke_density(
    height: int,
    width: int,
    morphology: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create one severity-independent spatial smoke field."""
    preset = MORPHOLOGY_PRESETS[morphology]
    base = _multiscale_density(height, width, generator)
    plumes = _add_local_plumes(
        base, _randint(generator, *preset.plumes), generator
    )
    rho = _normalize(
        preset.base_weight * base + preset.plume_weight * plumes
    ).pow(preset.gamma)
    if preset.density_floor > 0:
        rho = preset.density_floor + (1.0 - preset.density_floor) * rho
    rho = rho * (preset.target_mean / (rho.mean() + 1e-6))
    return rho.clamp(0, 1)


def synthesize_smoke(
    vi_clean: torch.Tensor,
    severity: str,
    seed: int,
    airlight: Optional[torch.Tensor] = None,
    morphology: str = "patchy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Synthesize one non-uniform smoke image.

    Args:
        vi_clean: RGB or grayscale float tensor ``(C,H,W)`` in ``[0,1]``.
        severity: ``mild``, ``medium`` or ``heavy``.
        seed: explicit seed; identical inputs and seed produce identical output.
        airlight: optional scalar or ``(C,1,1)`` tensor in ``[0,1]``.
        morphology: ``local``, ``patchy`` or ``diffuse``.  With the same
            seed and morphology, all severities reuse the same spatial field.

    Returns:
        ``(vi_smoke, smoke_mask, transmission, metadata)``.  Smoke mask is the
        continuous opacity ``1-t`` rather than a fabricated binary label.
    """
    if vi_clean.ndim != 3:
        raise ValueError(f"vi_clean must have shape (C,H,W), got {tuple(vi_clean.shape)}")
    if severity not in SEVERITY_PRESETS:
        raise ValueError(f"Unknown severity '{severity}'. Choose from {tuple(SEVERITY_PRESETS)}")
    if morphology not in MORPHOLOGY_PRESETS:
        raise ValueError(
            f"Unknown morphology '{morphology}'. Choose from {tuple(MORPHOLOGY_PRESETS)}"
        )
    if not torch.is_floating_point(vi_clean):
        raise TypeError("vi_clean must be a floating point tensor in [0,1]")

    preset = SEVERITY_PRESETS[severity]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    channels, height, width = vi_clean.shape

    rho = _smoke_density(height, width, morphology, generator)
    density_scale = _uniform(generator, *preset.density)
    rho = (rho * density_scale).clamp(0, 1.5)
    beta = _uniform(generator, *preset.beta)
    # Remove only the synthetic batch dimension.  Keeping the singleton
    # channel yields the public contract (1,H,W) and broadcasts cleanly over
    # a source tensor shaped (C,H,W).
    transmission = torch.exp(-beta * rho).clamp(0.05, 1.0).squeeze(0)

    if airlight is None:
        # Near-neutral airlight with a small channel variation; it avoids a
        # synthetic pure-white veil while retaining the scattering semantics.
        base = _uniform(generator, 0.78, 0.96)
        jitter = (torch.rand((channels, 1, 1), generator=generator) - 0.5) * 0.06
        airlight = (torch.full((channels, 1, 1), base) + jitter).clamp(0.72, 1.0)
    else:
        airlight = torch.as_tensor(airlight, dtype=vi_clean.dtype).cpu()
        if airlight.ndim == 0:
            airlight = airlight.expand(channels, 1, 1)

    source_device = vi_clean.device
    source_dtype = vi_clean.dtype
    clean_cpu = vi_clean.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
    smoke = clean_cpu * transmission + airlight * (1.0 - transmission)
    smoke = smoke.clamp(0, 1).to(device=source_device, dtype=source_dtype)
    transmission = transmission.to(device=source_device, dtype=source_dtype)
    smoke_mask = (1.0 - transmission).clamp(0, 1)

    metadata = {
        "seed": int(seed),
        "severity": severity,
        "morphology": morphology,
        "beta": float(beta),
        "density_scale": float(density_scale),
        "mean_transmission": float(transmission.mean().item()),
        "coverage": float((smoke_mask > 0.15).float().mean().item()),
        "airlight_mean": float(airlight.mean().item()),
    }
    return smoke, smoke_mask, transmission, metadata


def sample_severity(generator: Optional[torch.Generator] = None) -> str:
    """Uniformly sample one of the three protocol severities."""
    index = int(torch.randint(0, 3, (), generator=generator).item())
    return ("mild", "medium", "heavy")[index]


def sample_morphology(
    generator: Optional[torch.Generator] = None,
    morphologies: Tuple[str, ...] = ("local", "patchy", "diffuse"),
    probabilities: Tuple[float, ...] = (0.30, 0.50, 0.20),
) -> str:
    """Sample a configurable morphology mixture for online training."""
    if not morphologies or len(morphologies) != len(probabilities):
        raise ValueError("morphologies and probabilities must have equal non-zero length")
    invalid = set(morphologies) - set(MORPHOLOGY_PRESETS)
    if invalid:
        raise ValueError(f"Unknown morphologies: {sorted(invalid)}")

    weights = torch.tensor(probabilities, dtype=torch.float32)
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0
    ):
        raise ValueError("morphology probabilities must be finite, non-negative and non-zero")

    index = int(torch.multinomial(weights, 1, generator=generator).item())
    return morphologies[index]
