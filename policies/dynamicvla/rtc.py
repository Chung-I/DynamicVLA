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
    target = torch.zeros(chunk_size, a, dtype=torch.float32)
    if overlap == 0:
        return target, 0

    target[:overlap] = prev_actions_abs[shift : shift + overlap]
    target[:overlap, :n_delta_dims] -= latest_state_abs[:n_delta_dims]
    return target, overlap
