# Real-Time Chunking (RTC) for DynamicVLA streaming

RTC makes a newly generated action chunk continuous with the actions already
committed for execution, reducing discontinuities at chunk boundaries during
streaming inference. It is an inference-time method (no retraining) applied
inside the flow-matching denoising loop, and is **opt-in**.

## Modes (`rtc_mode` in the policy config)

- `off` (default): the original index-splice merge. Unchanged behavior.
- `softmask`: closed-form RePaint-style blend each Euler step. The previously
  committed chunk is forward-noised and blended into `x_t` with the RTC
  exponential soft-mask weights. No gradients; negligible extra latency.
- `pigdm`: RTC's PiGDM guidance. Each step computes a vector-Jacobian product
  (autodiff) that nudges the clean-data estimate toward the committed chunk,
  with the guidance weight clipped by `rtc_beta`. Most faithful to the paper;
  higher per-step cost.

## Enabling

RTC only applies during streaming inference (`enable_streaming=True`,
`scripts/inference.py -s`). Set in the checkpoint's `config.json`:

```json
{ "rtc_mode": "softmask", "rtc_beta": 5.0, "rtc_delay_buffer_size": 8 }
```

## How it works (data flow)

1. The main process snapshots the currently queued (absolute) actions, their
   start index, and a conservative delay estimate (max of recent realized
   skips), and ships them to the streaming worker via `q_in["rtc"]`.
2. The worker re-expresses that chunk as a delta-space, normalized, padded
   inpainting target aligned to the new chunk's indices, and computes the
   soft-mask weights (frozen for the first `d` = delay actions, exponential
   decay over the overlap, zero beyond the previous chunk).
3. `sample_actions` applies the target during denoising (`softmask` blend or
   `pigdm` guidance).

## Validating in sim

Run `simulations/evaluate.py` against `scripts/inference.py -s` once per mode
and compare success rates and `avg_path_length` (smoother trajectories are
shorter / less jerky). Use `inference.py -o <dir>` to dump per-step actions
for seam inspection.

## Caveats

- `pigdm` increases inference latency (autodiff per step); on a latency-bound
  setup this can widen the very seams RTC is trying to close. Prefer `softmask`
  unless seam quality is inadequate.
- RTC trades reactivity for smoothness: frozen actions cannot react to new
  observations within the delay window. On highly dynamic objects, validate
  that success rate does not regress vs `off`.

## Empirical findings (DOM sim, Isaac Sim, RTX 5090, paper-regime latency)

Released `hzxie/dynamic-vla-DOM` checkpoint, all 89 DOM test envs × 2 trials/mode
(jerk via `scripts/rtc_seam_metric.py`, EE-position 2nd difference):

| mode      | success | mean_jerk     | p95_jerk      |
|-----------|---------|---------------|---------------|
| off       | 20.8%   | 0.0263        | 0.1082        |
| softmask  | 24.2%   | 0.0166 (-37%) | 0.0623 (-42%) |
| pigdm     | 19.7%   | 0.0387 (+47%) | 0.2022 (+87%) |

- **`softmask` is recommended**: higher success AND markedly smoother
  trajectories (lower jerk) than `off`, in the low-latency regime RTC targets.
- **`pigdm` currently regresses** end-to-end (lower success, higher jerk) even
  though its single-chunk guidance reduces distance-to-target. Treat as
  experimental — needs `rtc_beta` tuning / a guidance decay schedule; not for use as-is.
- **Absolute success vs the paper (47.06%):** the gap is *not* precision (bf16
  gave no speedup) nor inference latency (forcing `dt_scale=1` did not raise
  success). The residual ~2x likely reflects released-checkpoint / eval-protocol
  differences not recoverable from public artifacts. The comparison above is
  relative (same checkpoint + harness + envs) and is unaffected by it.

### How `mean_jerk` / `p95_jerk` are computed

Both come from `scripts/rtc_seam_metric.py` (`compute_jerk`), on each episode's
executed end-effector **position** stream `p` of shape `(T, 3)`:

1. Per-step jerk series = L2 norm of the **second difference** of position:
   `jerk_t = ‖ p[t+1] − 2·p[t] + p[t−1] ‖` → a length-`(T−2)` array. This is the
   discrete 2nd derivative of position (a smoothness/jerkiness proxy; larger =
   more abrupt speed/direction changes). `mean_step` is the analogous L2 norm of
   the **first** difference (per-step travel distance).
2. Per episode: `mean_jerk` = mean of that series; `p95_jerk` = its 95th
   percentile (captures the spikiest moments, not just the average).
3. Per mode (the table values): `aggregate_mode` averages each per-episode number
   across all episodes — i.e. mean-over-episodes of the per-episode mean (and of
   the per-episode 95th percentile).

This is a **whole-trajectory** position-jerk proxy (not seam-localized — the
dumps don't record chunk-boundary indices), but the softmask↓ / pigdm↑ deltas
are large and consistent enough to be meaningful.

### Statistical significance (softmask vs off, paired)

Same checkpoint/harness/envs/observations across modes, so episodes are matched
by (env, trial) and compared with paired tests. "softmask better" = number of
paired episodes where softmask's value is lower (smoother/fewer/faster).

**Smoothness — all matched episodes (n=178; jerk is defined regardless of success):**

| metric    | off    | softmask | Δ    | softmask better | Wilcoxon p | t-test p | dz   |
|-----------|--------|----------|------|-----------------|------------|----------|------|
| mean_jerk | 0.0263 | 0.0166   | −37% | 148/178 (83%)   | 1.4e-18    | 1.4e-19  | 0.77 |
| p95_jerk  | 0.1082 | 0.0623   | −42% | 146/178 (82%)   | 5.3e-19    | 1.4e-18  | 0.74 |

Strongly significant (large effect). Holds under the conservative env-level test
(collapse 2 trials/env → n=89): mean_jerk smoother in 77/89 envs, Wilcoxon
p ≈ 8e-13, dz ≈ 1.07.

**Steps / execution time — both-success tasks only (n=21):**

| metric            | off    | softmask | Δ    | softmask better | Wilcoxon p | t-test p | dz   |
|-------------------|--------|----------|------|-----------------|------------|----------|------|
| action steps      | 156.5  | 140.9    | −10% | 17/21           | 0.003      | 0.048    | 0.46 |
| policy actions    | 149.1  | 133.4    | −11% | 17/21           | 0.003      | 0.046    | 0.46 |
| exec time (sim)   | 6.26 s | 5.63 s   | −10% | 17/21           | 0.003      | 0.048    | 0.46 |
| exec time (wall)  | 11.22 s| 10.17 s  | −9%  | 18/21           | 0.002      | 0.062    | 0.43 |

Significant by paired Wilcoxon (p ≤ 0.003, survives Bonferroni for 4 metrics),
small-to-medium effect; paired t-test borderline (p ≈ 0.05–0.06) at this small n.

**Success rate (24.2% vs 20.8%):** within noise at 2 trials/env — not claimed
as significant.
