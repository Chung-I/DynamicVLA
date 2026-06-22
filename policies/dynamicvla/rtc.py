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
