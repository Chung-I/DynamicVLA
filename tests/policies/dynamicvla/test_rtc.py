import math

import pytest
import torch

from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig


def _make_config(**overrides):
    cfg = DynamicVLAConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg


def test_config_defaults_rtc_off():
    cfg = DynamicVLAConfig()
    assert cfg.rtc_mode == "off"
    assert cfg.rtc_beta == 5.0
    assert cfg.rtc_delay_buffer_size == 8


def test_config_rejects_unknown_rtc_mode():
    with pytest.raises(ValueError):
        _make_config(rtc_mode="bogus")


from policies.dynamicvla import rtc


def test_weights_shape_and_endpoints():
    w = rtc.compute_rtc_weights(chunk_size=50, freeze=5, overlap=40)
    assert w.shape == (50,)
    assert torch.all(w[:5] == 1.0)          # frozen region
    assert torch.all(w[40:] == 0.0)         # beyond previous chunk
    assert torch.all((w[5:40] > 0.0) & (w[5:40] <= 1.0))


def test_weights_monotonic_decay_in_soft_region():
    w = rtc.compute_rtc_weights(chunk_size=50, freeze=5, overlap=40)
    soft = w[5:40]
    assert torch.all(soft[1:] <= soft[:-1] + 1e-6)  # non-increasing


def test_weights_freeze_clamped_to_overlap():
    # freeze beyond overlap -> everything in overlap is frozen, rest zero
    w = rtc.compute_rtc_weights(chunk_size=10, freeze=20, overlap=4)
    assert torch.all(w[:4] == 1.0)
    assert torch.all(w[4:] == 0.0)


def test_weights_zero_overlap_all_zero():
    w = rtc.compute_rtc_weights(chunk_size=10, freeze=3, overlap=0)
    assert torch.all(w == 0.0)
