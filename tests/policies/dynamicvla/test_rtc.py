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
