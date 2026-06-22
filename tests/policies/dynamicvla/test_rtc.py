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


def test_align_prev_chunk_basic_overlap_and_delta():
    H_prev, A = 50, 7
    prev = torch.arange(H_prev * A, dtype=torch.float32).reshape(H_prev, A)
    state = torch.full((A,), 100.0)
    # delta dims = first 6 (gripper = last, stays absolute)
    target, overlap = rtc.align_prev_chunk(
        prev_actions_abs=prev,
        latest_state_abs=state,
        shift=3,
        chunk_size=50,
        n_delta_dims=A - 1,
    )
    assert overlap == 47  # 50 - 3
    assert target.shape == (50, A)
    # row i of target corresponds to prev row (shift + i)
    expected_row0 = prev[3].clone()
    expected_row0[: A - 1] -= 100.0       # delta on non-gripper dims
    assert torch.allclose(target[0], expected_row0)
    # gripper (last dim) is NOT delta'd
    assert target[0, A - 1] == prev[3, A - 1]
    # rows beyond overlap are zero
    assert torch.all(target[overlap:] == 0.0)


def test_align_prev_chunk_no_overlap():
    prev = torch.zeros(10, 7)
    target, overlap = rtc.align_prev_chunk(prev, torch.zeros(7), shift=50, chunk_size=50, n_delta_dims=6)
    assert overlap == 0
    assert torch.all(target == 0.0)


def test_softmask_blend_frozen_region_at_data_time_equals_target():
    B, H, A = 1, 50, 7
    x_t = torch.randn(B, H, A)
    noise = torch.randn(B, H, A)
    target = torch.randn(H, A)
    weights = rtc.compute_rtc_weights(H, freeze=5, overlap=40)
    time = torch.tensor(0.0)  # data end of the trajectory
    out = rtc.apply_softmask_inpaint(x_t, noise, target, weights, time)
    # at time=0, x_known == target; frozen region (w=1) must equal target exactly
    assert torch.allclose(out[0, :5], target[:5], atol=1e-6)
    # zero-weight region (i >= overlap) is untouched
    assert torch.allclose(out[0, 40:], x_t[0, 40:], atol=1e-6)


def test_softmask_blend_is_convex_combination():
    B, H, A = 1, 4, 2
    x_t = torch.zeros(B, H, A)
    noise = torch.zeros(B, H, A)
    target = torch.ones(H, A)
    weights = torch.tensor([1.0, 0.5, 0.0, 1.0])
    out = rtc.apply_softmask_inpaint(x_t, noise, target, weights, torch.tensor(0.0))
    assert torch.allclose(out[0, :, 0], torch.tensor([1.0, 0.5, 0.0, 1.0]))


def test_pigdm_coef_clipped_by_beta_near_endpoints():
    assert rtc.pigdm_guidance_coef(time=1.0, beta=5.0) == 5.0
    assert rtc.pigdm_guidance_coef(time=0.0, beta=5.0) == 5.0


def test_pigdm_coef_unclipped_midrange():
    # at t=0.5: ((0.5)^2 + (0.5)^2)/((0.5)(0.5)) = 0.5/0.25 = 2.0 < beta
    assert abs(rtc.pigdm_guidance_coef(time=0.5, beta=5.0) - 2.0) < 1e-6
