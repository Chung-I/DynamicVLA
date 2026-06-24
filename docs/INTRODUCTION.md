# Introduction: DynamicVLA, the DOM task, and RTC

> Source material for a progress-report slide deck. Sections are self-contained;
> numbers are cited to their source (the DynamicVLA paper, the RTC paper, or
> *this project's* experiments). Where a figure comes from our own runs it is
> labelled **(ours)** and carries its caveats.

---

## 1. DynamicVLA — a compact VLA for *dynamic* object manipulation

**The problem.** Vision-Language-Action (VLA) models do well on *static*
manipulation but struggle when the target object keeps moving. The core issue is
**inference latency**: while the model reasons over an observation, the world
keeps changing, so by the time an action is produced it can already be stale —
perception and execution desynchronize. Handling moving objects needs *temporal
anticipation* and *continuous control*, not just a better single-shot policy.

**The model.** DynamicVLA (Xie et al., S-Lab @ NTU; arXiv 2601.22153) is a
**compact ~0.4B-parameter VLA** designed for low latency:
- **Vision encoder:** FastViT (convolutional) for efficient spatial compression
  and fast inference.
- **Language/backbone:** SmolLM2-360M, truncated to 16 transformer layers.
- **Action head:** a conditional **flow-matching** transformer ("action expert")
  that denoises an action chunk (DOM checkpoint: chunk size 20, 10 Euler denoise
  steps, dual 384×384 cameras `wrist_cam`+`opst_cam`, 2 observation timesteps).

**Two systems-level ideas that make it "dynamic":**
- **Continuous (pipelined) inference** — overlap reasoning with execution so the
  robot never stalls waiting for the next chunk.
- **Latent-aware action streaming** — discard stale actions and prioritize the
  freshest predictions to keep perception and action aligned.

Reported efficiency: ~88 Hz on an NVIDIA RTX A6000.

---

## 2. The DOM task — Dynamic Object Manipulation benchmark

**What it is.** A purpose-built benchmark (released with DynamicVLA) for
manipulating objects *in motion*, built in **NVIDIA Isaac Sim** with a **Franka
Emika Panda** arm. Actions are a 32-d end-effector-pose + gripper vector.

**Data scale (training):** ~200K synthetic episodes across ~2,800 scenes and
~206 objects, plus ~2,000 teleoperation-free real episodes (dual-camera pose
estimation), on Franka and PiPER arms.

**Evaluation protocol:** **1,800 trials = 10 scenes × 9 "dimensions" × 20 trials**.
Success = *complete the instructed manipulation without dropping the object or
timing out*. Secondary metrics: path length (m) and task-completion time (s).

**The 9 evaluation dimensions** (3 groups × 3):
- **Interaction:** closed-loop reactivity · dynamic adaptation · long-horizon sequencing
- **Perception:** visual understanding · spatial reasoning · motion perception
- **Generalization:** visual generalization · motion generalization · disturbance robustness

**Headline results (paper, average success over the 1,800 trials):**

| method | success |
|---|---|
| **DynamicVLA** | **47.06%** |
| VLA-Adapter-Pro | 13.61% |
| GR00T-N1.5 | 13.05% |
| SmolVLA | 12.67% |
| π₀.₅ | 11.06% |
| VLASH | 12.33% |
| π₀ | 8.11% |
| OpenVLA-OFT | 1.33% |
| Diffusion Policy | 0.38% |

---

## 3. RTC — Real-Time Chunking

**Origin.** *Real-Time Execution of Action Chunking Flow Policies* (Black,
Galliker, Levine; Physical Intelligence / UC Berkeley; arXiv 2506.07339). RTC is
an **inference-time algorithm** (no retraining) for flow/diffusion action-chunking
policies — exactly the family DynamicVLA's action expert belongs to.

**The problem it targets.** When a policy executes in *chunks*, the boundary
between the current chunk and the next (generated during the inference delay)
causes **discontinuities/jerk** and out-of-distribution motion — worst on
precision and dynamic tasks.

**The idea.** Frame asynchronous chunking as **guided inpainting** on the
flow-matching sampler:
- **Freeze** the actions guaranteed to execute during the inference delay
  (`d` = ⌊latency/Δt⌋), so the new chunk *starts from* the committed trajectory.
- **Soft-mask** the overlapping region with exponentially-decaying weights so the
  new chunk blends smoothly into the old one.
- **Guidance** (the ΠiGDM variant) steers the denoiser toward the committed
  actions via a clipped vector-Jacobian term.

Net effect (paper): smooth chunk transitions, robust to 300 ms+ latency, ~20%
faster real-robot execution vs synchronous inference.

---

## 4. This project — RTC on DynamicVLA (progress)

**Goal.** Add RTC to DynamicVLA's streaming inference and test whether it
improves chunk-to-chunk smoothness (RTC's claim) in the latency regime
DynamicVLA targets.

**What was built.** An opt-in `rtc_mode ∈ {off, softmask, pigdm}` (default `off`,
non-breaking) threaded through the streaming worker:
- `off` — the original index-splice merge (baseline).
- `softmask` — closed-form RePaint-style blend toward the committed chunk each
  denoise step (no gradients; negligible overhead).
- `pigdm` — RTC's ΠiGDM guidance via a per-step vector-Jacobian product (autodiff).

**Validation method.** CPU unit tests (RTC math) → GPU smoke test on the real
checkpoint → full Isaac Sim evaluation on the released `hzxie/dynamic-vla-DOM`
checkpoint, comparing modes on **success rate** and a **seam-jerk** metric
(L2 norm of the 2nd difference of the executed end-effector trajectory; lower =
smoother).

**Findings (ours — Isaac Sim, 89 envs × 2 trials/mode):**

| mode | success | mean_jerk | steps (both-success) | exec time (both-success) |
|---|---|---|---|---|
| off (baseline) | 20.8% | 0.0263 | 156.5 | 11.2 s |
| **softmask** | **24.2%** | **0.0166 (−37%)** | **140.9 (−10%)** | **10.2 s (−9%)** |
| pigdm | 19.7% | 0.0387 (+47%) | — | — |

- **`softmask` is the validated win:** smoother trajectories (−37% jerk),
  fewer steps and faster completion on tasks both modes solve, and slightly
  higher success — in the low-latency regime RTC is designed for. Recommended.
**Statistical significance (softmask vs off, paired).** "softmask better" =
paired episodes where softmask is lower (smoother/fewer/faster).

*Smoothness — all matched episodes (n=178):*

| metric    | off    | softmask | Δ    | softmask better | Wilcoxon p | t-test p | dz   |
|-----------|--------|----------|------|-----------------|------------|----------|------|
| mean_jerk | 0.0263 | 0.0166   | −37% | 148/178         | 1.4e-18    | 1.4e-19  | 0.77 |
| p95_jerk  | 0.1082 | 0.0623   | −42% | 146/178         | 5.3e-19    | 1.4e-18  | 0.74 |

→ strongly significant, large effect (holds env-level too: 77/89, p ≈ 8e-13, dz ≈ 1.07).

*Steps / execution time — both-success tasks only (n=21):*

| metric           | off     | softmask | Δ    | softmask better | Wilcoxon p | t-test p | dz   |
|------------------|---------|----------|------|-----------------|------------|----------|------|
| action steps     | 156.5   | 140.9    | −10% | 17/21           | 0.003      | 0.048    | 0.46 |
| policy actions   | 149.1   | 133.4    | −11% | 17/21           | 0.003      | 0.046    | 0.46 |
| exec time (sim)  | 6.26 s  | 5.63 s   | −10% | 17/21           | 0.003      | 0.048    | 0.46 |
| exec time (wall) | 11.22 s | 10.17 s  | −9%  | 18/21           | 0.002      | 0.062    | 0.43 |

→ significant by paired Wilcoxon (p ≤ 0.003, survives Bonferroni), small-medium
effect; t-test borderline at this small n. Success-rate differences are within
noise — **the smoothness result is the strong, headline claim.**

**Does lower jerk *cause* better success? No evidence.** (1) The success gain
isn't significant (McNemar p = 0.42). (2) Per-env jerk-reduction vs success-gain:
Spearman r = −0.15, p = 0.16 — no correlation. (3) The weak failure↔jerk link
(failures 1.2× jerkier) is likely reverse causation (timeouts wander → jerk).
So smoothness and the success blip look like **independent effects**; we claim
the smoothness win, **not** a success benefit (success test is underpowered, so
"no evidence," not "proven none").

**Inference cost & emulated latency.** The benchmark's `dt_scale` (sim-loop /
control-period) cancels out via client self-pacing, so the latency the policy
actually faces = `inference_time / control_period` (`inf/sim_dt`) — **independent
of GPU render speed**; that, not `dt_scale`, is the latency that matters.

| mode | avg inference | inf/sim_dt (latency, control steps) |
|---|---|---|
| off | 114 ms | 2.8 |
| softmask | 120 ms (+5%) | 3.0 |
| pigdm | 281 ms (2.5×) | 7.0 |

softmask is essentially free (~5%); pigdm's per-step autodiff costs 2.5× more
inference → 2.5× more latency (GPU-independent) — a real handicap behind its
regression. (Caveat: on a shared GPU, clean rendering slows inference too — the
design assumes sim and policy can be decoupled.)

**RTC chunk split (measured, 5090).** Mapping our run onto the RTC soft-mask
figure — chunk H=20, partitioned per-inference from the measured pipeline delay:

| region | frozen (d) | changeable (H−d−s) | fresh (s) |
|---|---|---|---|
| measured | ≈ 6–7 | ≈ 4–10 | ≈ 4–9 |

So ~6–7 of 20 actions are hard-frozen; the rest is the exp-decay blend over the
overlap — softmask's smoothing is mostly the soft blend, not the freeze. `d` =
the full obs→compute→queue→execute pipeline depth (hence > the per-step `dt_scale≈2`),
and scales with latency (slower machines → larger frozen region).
- **`pigdm` currently regresses** end-to-end (lower success, +47% jerk) despite
  its single-chunk guidance being correct; needs tuning (`rtc_beta` / a guidance
  decay schedule). Experimental.

**Side-by-side demo videos (ours) — baseline vs softmask.** Both clips are the
**same task and scene** (`1-1_place cup03d`: pick up the cup, place it in the
container) and **both succeed**, so the difference you see is *motion quality*,
not task outcome. This is the matched pair with the largest jerk gap among
trials successful in both modes.

| file | method | mean_jerk | what to look for |
|---|---|---|---|
| `…cup03d…-022909-SUCCESS.mp4` | **off (baseline)** | 0.0361 | end-effector is visibly **twitchy / oscillatory**, especially at action-chunk boundaries |
| `…cup03d…-035938-SUCCESS.mp4` | **softmask (RTC)** | 0.0099 | **~3.6× smoother**, continuous motion across chunk seams while still completing the task |

Each frame is three camera views side-by-side (the policy's wrist + scene cams).
Suggested slide: play the two clips side-by-side; caption "same task, both
succeed — RTC (softmask) removes the chunk-boundary jerk (3.6× lower)."

**Baseline gap RESOLVED — render quality + latency (not RTC, not protocol).**
The initial 20.8% (far below the paper's 47.06%) was a *hardware artifact*, now
fully explained by re-running `off` while varying render quality and policy
latency independently:

| setup | renders | policy latency (`inf/sim_dt`) | off success |
|---|---|---|---|
| standalone RTX 5090 | noisy (Blackwell denoiser broken) | low (~2.8 steps) | 20.8% |
| standalone cml18 (RTX 4090) | clean | high (~11 steps) | 37.1% |
| **cross-machine (cml18 sim + 5090 model)** | **clean** | **low (~2.6 steps)** | **56.2%** |
| paper (A6000) | clean | low | 47.06% |

Render quality ≈ **+16 pts**, low latency ≈ **+19 pts**; with both fixed the
released checkpoint **matches/exceeds** the paper. The cross-machine setup runs
the sim+render on cml18's 4090 (clean) and the policy on the local 5090 (fast),
ZMQ-linked over a 1 ms-LAN SSH tunnel — decoupling render speed from inference
latency onto separate GPUs.

**RTC is latency-gated smoothing — confirmed across three regimes.** softmask vs
off, paired:

| regime | overlap | mean_jerk Δ | significance | success Δ |
|---|---|---|---|---|
| 5090 (noisy, low-lat) | 11–16 | −37% | p≈1e-18, dz≈0.77 | ns |
| cml18 (clean, high-lat) | ~2 | −4% | p=0.14 (**ns**) | ns |
| **cross-machine (clean, low-lat)** | **~15** | **−36%** | **p≈2e-20, dz≈0.68** | ns |

The cross-machine run **reproduces** the strong smoothing (−36%/−42% jerk) with
*verified* clean renders + low latency + full RTC engagement (overlap ~15) — so
it is real, not a noisy-5090 artifact. It **vanishes at high latency** (cml18,
overlap collapses to ~2). And **success is flat in every regime** (cross-machine
off 56.2% vs softmask 56.7%, McNemar p=1.0): RTC's value is *purely* smoothness,
gated by latency.

**Our eval subset vs the paper's test set.** Same DOM structure, fewer trials:
ours = **89 scenes × 2 trials = 178 episodes**; paper (Table I) = **90 = 9
dimensions × 10 scenes, × 20 trials = 1,800**. Our 9 tiers map one-to-one onto the
paper's 9 dimensions (`1-x`=Interaction CR/DA/LS, `2-x`=Perception VU/SR/MP,
`3-x`=Generalization VG/MG/DR; e.g. our 1-1=60% vs paper CR=60.5%). With only **2
trials/env** our per-env success is 0.0/0.5/1.0-granular (vs the paper's 0.05 over
20 trials), so our 56.2% vs 47.06% is "at/above paper within sampling noise", not
a precise superiority claim (we also run 89 of 90 scenes — one DR scene missing
from the released `test-envs.txt`).

**Engineering notes worth a slide.** The end-to-end sim eval caught real bugs
the unit + smoke tests missed — most notably a CPU/CUDA device mismatch that
crashed the RTC streaming worker mid-episode (fixed; red-green regression test),
and a pigdm divergence fixed by step-size scaling. Takeaway: full closed-loop
evaluation is a necessary gate beyond component tests.

---

### Citations
- DynamicVLA & DOM: Xie et al., *DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation*, arXiv:2601.22153.
- RTC: Black, Galliker, Levine, *Real-Time Execution of Action Chunking Flow Policies*, arXiv:2506.07339.
- This project: `docs/rtc.md` (operator guide + empirical findings), `docs/rtc-validation.md` (runbook).
