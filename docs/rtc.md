# Real-Time Chunking (RTC) for DynamicVLA streaming — method & measurement

RTC makes a newly generated action chunk continuous with the actions already
committed for execution, reducing discontinuities at chunk boundaries during
streaming inference. It is an inference-time method (no retraining) applied
inside the flow-matching denoising loop, and is **opt-in**.

> **Scope.** This file is the **run-independent reference**: how RTC works and
> how each metric is computed. **All experimental results** (the 5090 / cml18 /
> cross-machine runs, baseline-gap resolution, significance, Table-I reproduction,
> demo videos) live in [`INTRODUCTION.md`](INTRODUCTION.md). Nothing here should
> contain run-specific numbers.

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

## Caveats

- `pigdm` increases inference latency (autodiff per step); on a latency-bound
  setup this can widen the very seams RTC is trying to close. Prefer `softmask`
  unless seam quality is inadequate.
- RTC trades reactivity for smoothness: frozen actions cannot react to new
  observations within the delay window. On highly dynamic objects, validate
  that success rate does not regress vs `off`.

## Validating in sim

Run `simulations/evaluate.py` against `scripts/inference.py -s` once per mode and
compare success rate, jerk, and path length. Use `inference.py -o <dir>` to dump
per-step actions for seam inspection; `scripts/rtc_seam_metric.py` aggregates jerk.

---

# Metric definitions (how each number is computed)

## `mean_jerk` / `p95_jerk`

Both come from `scripts/rtc_seam_metric.py` (`compute_jerk`), on each episode's
executed end-effector **position** stream `p` of shape `(T, 3)`:

1. Per-step jerk series = L2 norm of the **second difference** of position:
   `jerk_t = ‖ p[t+1] − 2·p[t] + p[t−1] ‖` → a length-`(T−2)` array. This is the
   discrete 2nd derivative of position (a smoothness/jerkiness proxy; larger =
   more abrupt speed/direction changes). `mean_step` is the analogous L2 norm of
   the **first** difference (per-step travel distance).
2. Per episode: `mean_jerk` = mean of that series; `p95_jerk` = its 95th
   percentile (captures the spikiest moments, not just the average).
3. Per mode: average each per-episode number across all episodes.

This is a **whole-trajectory** position-jerk proxy (not seam-localized — the
dumps don't record chunk-boundary indices).

## `Time` and `Path Len` — and why they are render-fair

The eval loop (`evaluate.py`, `while sim_results["status"] == -1`) runs **one
control step per iteration**, appending one `ee_path` entry per step. So:

```
Time     = (#control steps until termination) × step_dt
Path Len = Σ ‖ ee_path[t+1] − ee_path[t] ‖   (meters)
```

- **`step_dt` = `--physics_time_step` = 0.04 s** (`env_cfg.dt`; 25 Hz, matching the
  paper's 25-FPS cameras). Termination: `status=0` (success place-term) or
  `status=1` (drop **or** the env's `time_out` term, ~300 steps ≈ 12 s). Averaged
  over **all** trials (success short, timeout ~12 s) — as in the paper.
- **`Time` is *simulated* seconds (step-count × 0.04 s), NOT wall-clock.** Each
  control step is 0.04 s of sim-world time regardless of how long the GPU takes to
  render it, so a fast and a slow GPU recording the *same* trajectory get the
  *same* `Time`. The real-time floor (`if step_time < step_dt: sleep(...)`) plus
  the client's `dt_scale` sleep keep the sim advancing at the right rate relative
  to inference, but the *recorded* time is always `n_steps × step_dt`.

**Hardware fairness:** `Time`/`Path Len` are immune to **render/GPU-render speed**
(the `dt_scale` cancellation — see below), but they **intentionally reflect
inference latency**: a faster policy (lower `inf/sim_dt`) reacts sooner →
completes in fewer steps and times-out less → lower `Time`. Fair across render
hardware, while still crediting a faster model — by design.

## Emulated latency: `dt_scale` vs `inf/sim_dt`

The benchmark injects inference latency via `dt_scale` = (sim loop wall-time) /
(control period, 0.04 s). The client paces itself (`sleep = inf·(dt_scale−1)`),
which makes `dt_scale` **cancel** in the latency the policy actually experiences:

```
sim steps elapsed per chunk = (inf·dt_scale) / (dt_scale·step_dt) = inf / step_dt
```

So the **real, render-independent latency** the policy faces is `inf/sim_dt`
(inference time in control-step units) — **not** `dt_scale`. Consequences:

- `dt_scale` is **render-bound** (server-side sim loop). Slow rendering raises it,
  but it cancels — it is *not* the policy's latency.
- `inf/sim_dt` is **inference-bound** and render-independent: a faster model → less
  latency, regardless of GPU render speed.
- **Caveat — shared-GPU coupling:** if inference and rendering share one GPU,
  heavy (clean) rendering steals cycles from the model forward and slows
  inference → more latency. Running sim and policy on **separate GPUs** restores
  the intended decoupling.

## RTC chunk-split: `frozen` / `changeable` / `fresh`

The soft-mask schedule (the RTC guidance-weight curve) partitions the
`H`-action chunk (`H` = `chunk_size` = 20) into three regions, logged per chunk by
the `[RTC]` line in `_build_inpaint_target`:

| region | weight | meaning |
|---|---|---|
| **frozen** (`d`) | 1 | inference delay; executes as-is |
| **changeable** (`overlap − d`) | exp-decay | overlaps the previous chunk; soft-blended |
| **fresh** (`H − overlap`) | 0 | beyond the previous chunk; freshly generated |

with two runtime quantities:

```
overlap = h_prev − shift        # how many of the PREVIOUS chunk's actions are still queued
d (frozen) = min(freeze, overlap)   # freeze = max recent skip_n_actions (full obs→compute→queue→execute pipeline depth)
```

**Key point:** `frozen` is **not** the inference delay — it is
`min(delay, overlap)`. When the policy is slow relative to consumption, the action
queue drains between chunks, the `overlap` collapses toward 0, and RTC degenerates
to freezing only the few still-queued actions with **no** exp-decay blend. So RTC's
soft-masking is **gated by latency**: it operates only when `inf/sim_dt` is low
enough to keep meaningful overlap. (Measured per-regime values: see
`INTRODUCTION.md`.)
