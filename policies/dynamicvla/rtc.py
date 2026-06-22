# -*- coding: utf-8 -*-
#
# @File:   rtc.py
# RTC (Real-Time Chunking) guided-inpainting math for DynamicVLA streaming.
# Pure functions only: no model, no GPU, no LeRobot imports.

import math

import torch


def compute_rtc_weights(chunk_size: int, freeze: int, overlap: int) -> torch.Tensor:
    """Soft-masking weights W_i over a chunk of length `chunk_size`.

    Implements RTC Eq. 5 with H = chunk_size, d = freeze, (H - s) = overlap:
        W_i = 1                              for i < d
            = c_i * (exp(c_i) - 1)/(e - 1)   for d <= i < (H - s)
            = 0                              for i >= (H - s)
      where c_i = (overlap - i) / (overlap - freeze + 1).
    """
    overlap = max(0, min(overlap, chunk_size))
    freeze = max(0, min(freeze, overlap))

    w = torch.zeros(chunk_size, dtype=torch.float32)
    if overlap == 0:
        return w

    w[:freeze] = 1.0
    denom = (overlap - freeze + 1) if (overlap - freeze + 1) != 0 else 1
    for i in range(freeze, overlap):
        c_i = (overlap - i) / denom
        w[i] = c_i * (math.exp(c_i) - 1.0) / (math.e - 1.0)
    return w


def align_prev_chunk(
    prev_actions_abs: torch.Tensor,
    latest_state_abs: torch.Tensor,
    shift: int,
    chunk_size: int,
    n_delta_dims: int,
) -> tuple[torch.Tensor, int]:
    """Re-express the previously committed (absolute) chunk as a delta-space
    inpainting target aligned to the *new* chunk's indices.

    New-chunk local index i corresponds to previous-chunk local index (shift + i).
    Delta dims are made relative to the current state; remaining dims stay absolute.
    """
    h_prev, a = prev_actions_abs.shape
    overlap = max(0, min(chunk_size, h_prev - shift))
    # Match the input device so the (delta) arithmetic below does not mix
    # CPU/CUDA tensors when called from the streaming worker (state is on GPU).
    target = torch.zeros(chunk_size, a, dtype=torch.float32, device=prev_actions_abs.device)
    if overlap == 0:
        return target, 0

    target[:overlap] = prev_actions_abs[shift : shift + overlap]
    target[:overlap, :n_delta_dims] -= latest_state_abs[:n_delta_dims]
    return target, overlap


def apply_softmask_inpaint(
    x_t: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """RePaint-style blend toward the forward-noised known trajectory.

    x_known(time) = time*noise + (1 - time)*target
    x_t <- (1 - w)*x_t + w*x_known(time)   (broadcast w over action dims)
    """
    if target.ndim == 2:
        target = target.unsqueeze(0)
    w = weights.view(1, -1, 1).to(x_t.dtype)
    x_known = time * noise + (1.0 - time) * target
    return (1.0 - w) * x_t + w * x_known


def pigdm_guidance_coef(time: float, beta: float) -> float:
    """RTC guidance clip min(beta, (1-tau)/(tau*r^2)) in DynamicVLA flow-time.

    With tau = 1 - time and r^2 = (1-tau)^2/(tau^2 + (1-tau)^2), the factor
    (1-tau)/(tau*r^2) simplifies to ((1-t)^2 + t^2) / ((1-t)*t).
    """
    t = min(max(time, 1e-4), 1.0 - 1e-4)
    factor = ((1.0 - t) ** 2 + t ** 2) / ((1.0 - t) * t)
    return min(beta, factor)
