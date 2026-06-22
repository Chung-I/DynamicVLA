# -*- coding: utf-8 -*-
#
# @File:   rtc_smoke_test.py
# RTC validation smoke test (the deferred Task 5 Step 6 check, made self-contained).
#
# Loads a real DynamicVLA checkpoint and, with ONE fixed noise seed, runs the
# flow-matching sampler in three RTC modes against a synthetic inpaint target:
#
#   1. off       -> baseline (no inpainting)
#   2. softmask  -> frozen region MUST equal the target exactly at the end
#   3. pigdm     -> must run without crashing AND move the weighted region
#                   CLOSER to the target than the baseline (validates the
#                   +coef*g gradient sign that could not be checked on CPU)
#
# Usage:
#   .venv/bin/python scripts/rtc_smoke_test.py /path/to/checkpoint [--freeze 5] [--overlap 40]
#
# Exit code 0 = all checks passed; nonzero = a check failed (details printed).

import argparse
import logging
import os
import sys

import torch

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
)
import utils.helpers  # noqa: E402
from lerobot.constants import OBS_STATE  # noqa: E402
from policies.dynamicvla import rtc  # noqa: E402
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy  # noqa: E402


def build_synthetic_batch(cfg, device):
    """A shape-valid batch matching the model's input_features (zeros + a task)."""
    batch = {}
    for key, feat in cfg.image_features.items():
        # prepare_images expects (b, n_obs_steps, c, h, w)
        batch[key] = torch.zeros(
            1, cfg.n_obs_steps, *feat.shape, dtype=torch.float32, device=device
        )
    state_dim = cfg.input_features[OBS_STATE].shape[0]
    batch[OBS_STATE] = torch.zeros(
        1, cfg.n_obs_steps, state_dim, dtype=torch.float32, device=device
    )
    batch["task"] = ["pick up the object"]
    return batch


def weighted_l2_to_target(sample, target, weights):
    """Sum of w_i * ||sample_i - target_i||^2 over the chunk (batch 0)."""
    diff = (sample[0] - target) ** 2  # (chunk, dim)
    return (weights.view(-1, 1) * diff).sum().item()


def main():
    parser = argparse.ArgumentParser(description="RTC validation smoke test")
    parser.add_argument("checkpoint", type=str, help="Path to DynamicVLA checkpoint dir")
    parser.add_argument("--freeze", type=int, default=5, help="RTC freeze count d")
    parser.add_argument("--overlap", type=int, default=40, help="RTC overlap (H - s)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Frozen-region tolerance")
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO
    )
    if not torch.cuda.is_available():
        logging.warning("No CUDA device — running on CPU (slow). pigdm autodiff still valid.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = utils.helpers.get_policy_cfg(
        cfg_file=os.path.join(args.checkpoint, "config.json")
    )
    # Smoke test exercises sample_actions directly; keep streaming off here.
    cfg.enable_streaming = False
    logging.info("Loading checkpoint: %s", args.checkpoint)
    policy = DynamicVLAPolicy.from_pretrained(args.checkpoint, config=cfg)
    policy.eval().to(device)
    fm = policy.model

    # Prepare a fixed set of model inputs (deterministic across modes).
    batch = build_synthetic_batch(cfg, device)
    batch = policy.normalize_inputs(batch)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)

    H, A = cfg.chunk_size, cfg.max_action_dim
    torch.manual_seed(0)
    noise = torch.randn(1, H, A, device=device)
    target = torch.randn(H, A, device=device)
    weights = rtc.compute_rtc_weights(H, freeze=args.freeze, overlap=args.overlap).to(device)

    def sample(mode):
        with torch.no_grad() if mode != "pigdm" else torch.enable_grad():
            return fm.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise.clone(),
                rtc_mode=mode,
                inpaint_target=(None if mode == "off" else target.unsqueeze(0)),
                inpaint_weights=(None if mode == "off" else weights),
                rtc_beta=cfg.rtc_beta,
            )

    failures = []

    # --- off baseline ---
    base = sample("off")
    logging.info("off: produced chunk %s", tuple(base.shape))

    # --- softmask: frozen region must equal target exactly ---
    sm = sample("softmask")
    frozen_ok = torch.allclose(sm[0, : args.freeze], target[: args.freeze], atol=args.atol)
    logging.info(
        "softmask: frozen-region match=%s (max abs diff %.2e)",
        frozen_ok,
        (sm[0, : args.freeze] - target[: args.freeze]).abs().max().item(),
    )
    if not frozen_ok:
        failures.append("softmask frozen region does not equal the target")
    # NOTE: the beyond-overlap region is NOT directly blended (weight 0), but it
    # is still expected to differ from the baseline end-to-end: the action tokens
    # attend to each other in the expert, so pinning the frozen region changes the
    # velocity predicted for the tail at later steps. The isolated-blend invariant
    # (tail untouched per-step) is covered by the CPU unit test; here we only
    # report the magnitude for visibility.
    tail_drift = (sm[0, args.overlap :] - base[0, args.overlap :]).abs().max().item()
    logging.info(
        "softmask: beyond-overlap drift vs off = %.3e (expected nonzero via attention)",
        tail_drift,
    )

    # --- pigdm: must run, and move weighted region closer to target than baseline ---
    try:
        pg = sample("pigdm")
        d_off = weighted_l2_to_target(base, target, weights)
        d_pg = weighted_l2_to_target(pg, target, weights)
        logging.info(
            "pigdm: weighted L2 to target  off=%.4f  pigdm=%.4f  (pigdm should be smaller)",
            d_off, d_pg,
        )
        if not (d_pg < d_off):
            failures.append(
                "pigdm did NOT reduce weighted distance to target "
                "(possible gradient sign/scale error — flip sign of coef*g or check VJP)"
            )
    except RuntimeError as e:
        failures.append("pigdm crashed at runtime: %s" % e)

    print("\n==================== RTC SMOKE TEST ====================")
    if failures:
        for f in failures:
            print("  FAIL:", f)
        print("=======================================================")
        sys.exit(1)
    print("  PASS: softmask pins the frozen region to the target, and pigdm")
    print("        reduces the weighted distance to the target.")
    print("=======================================================")


if __name__ == "__main__":
    main()
