"""Selective state-space recurrence used by the SAFIR image encoder.

The CUDA path calls mamba-ssm, as in the research implementation. The native
PyTorch path evaluates the same recurrence sequentially and supports CPU smoke
tests and inspection. It is much slower and is not a substitute for a CUDA
paper-metric run. Floating-point rounding can differ between the two backends.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan_reference(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, return_last_state=False,
):
    """Evaluate the real-valued grouped scan used by ``encoder.SS2D``.

    x[t] = exp(delta[t] * A) * x[t-1] + delta[t] * B[t] * u[t]
    y[t] = sum(C[t] * x[t]) + D * u[t]

    B and C have shape (batch, groups, state, sequence); their channels are
    shared within each group. Accumulation is float32, matching SS2D's inputs.
    """
    if A.is_complex() or B.ndim != 4 or C.shape != B.shape:
        raise ValueError("SAFIR reference scan requires real grouped B/C tensors")
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must have identical (batch, channels, length) shape")
    batch, channels, length = u.shape
    groups, states = B.shape[1:3]
    if length == 0 or channels % groups or A.shape != (channels, states):
        raise ValueError("incompatible selective scan dimensions")
    if B.shape[0] != batch or B.shape[-1] != length:
        raise ValueError("B/C batch or sequence dimension does not match u")
    original_dtype = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    grouped_u = u.reshape(batch, groups, channels // groups, length)
    grouped_delta = delta.reshape_as(grouped_u)
    grouped_A = A.float().reshape(1, groups, channels // groups, states)
    state = u.new_zeros(batch, groups, channels // groups, states)
    outputs = []
    for index in range(length):
        step = grouped_delta[..., index, None]
        state = (
            torch.exp(step * grouped_A) * state
            + step * grouped_u[..., index, None] * B[..., index].float().unsqueeze(2)
        )
        outputs.append((state * C[..., index].float().unsqueeze(2)).sum(-1))
    output = torch.stack(outputs, dim=-1).reshape(batch, channels, length)
    if D is not None:
        output = output + D.float().view(1, -1, 1) * u
    if z is not None:
        output = output * F.silu(z.float())
    output = output.to(original_dtype)
    if return_last_state:
        return output, state.reshape(batch, channels, states)
    return output


def selective_scan_fn(*args, backend="auto", **kwargs):
    if backend not in {"auto", "cuda", "reference"}:
        raise ValueError(f"unknown selective scan backend: {backend}")
    use_cuda = backend == "cuda" or (backend == "auto" and args[0].is_cuda)
    if use_cuda:
        if not args[0].is_cuda:
            raise RuntimeError("The cuda scan backend requires CUDA input tensors")
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Install a mamba-ssm build matching PyTorch/CUDA, or explicitly "
                "choose scan_backend='reference' for the slower PyTorch recurrence"
            ) from error
        return cuda_scan(*args, **kwargs)
    return selective_scan_reference(*args, **kwargs)
