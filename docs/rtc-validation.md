# RTC validation runbook

Two gates were deferred from the implementation because they need a model
checkpoint (and, for the second, Isaac Sim) — neither was available on the dev
machine. Run both before relying on RTC at inference time.

Prereqs: the project venv (`.venv`, set up via `uv pip install --torch-backend=auto
-r requirements.txt`) and a trained DynamicVLA checkpoint directory (the dir that
holds `config.json` + `model.safetensors`).

---

## Gate 1 — GPU smoke test (fast, ~1 min, GPU only)

Validates the RTC math on the real model with one fixed noise seed:
- `softmask` pins the frozen region to the target and leaves the beyond-overlap tail untouched;
- `pigdm` runs without crashing **and** reduces the weighted distance to the target
  vs the baseline (this is the check that confirms the `+coef*g` gradient sign —
  the one thing CPU unit tests could not verify).

```bash
.venv/bin/python scripts/rtc_smoke_test.py /path/to/checkpoint
# options: --freeze 5 --overlap 40 --atol 1e-4
```

Exit code 0 = PASS. If `pigdm` reports it did NOT reduce the weighted distance,
the gradient sign/scale is wrong — flip the sign of `coef * g` in
`VLAFlowMatching.sample_actions` (or re-derive the VJP) and re-run. `softmask`
is unaffected by that and is the recommended production mode meanwhile.

Runs on the local RTX 5090, on a CML node (`CUDA_VISIBLE_DEVICES=N`, ≤50 GB VRAM
rule), or as an NCHC Slurm job (`#SBATCH --gres=gpu:1`).

---

## Gate 2 — End-to-end sim comparison (Isaac Sim)

Compares `off` vs `softmask` vs `pigdm` on the Dynamic Object Manipulation tasks
through the existing ZMQ harness. Smoother RTC trajectories should show lower
`avg_path_length` (less jerk) without a success-rate regression.

The streaming worker reads `rtc_mode` from the checkpoint's `config.json`, so each
mode is one edit + one rerun. Two processes per run:

**Terminal A — sim server (Isaac Sim env):**
```bash
python3 simulations/evaluate.py   # existing invocation/args for your setup
```

**Terminal B — inference client (the venv), once per mode:**
```bash
# 1) off (baseline)
#    set "rtc_mode": "off" in <checkpoint>/config.json
.venv/bin/python scripts/inference.py -s -p /path/to/checkpoint -o runs/rtc_off

# 2) softmask
#    set "rtc_mode": "softmask" in <checkpoint>/config.json
.venv/bin/python scripts/inference.py -s -p /path/to/checkpoint -o runs/rtc_softmask

# 3) pigdm   (only after Gate 1 passes)
#    set "rtc_mode": "pigdm" in <checkpoint>/config.json
.venv/bin/python scripts/inference.py -s -p /path/to/checkpoint -o runs/rtc_pigdm
```

`-s` enables streaming (spawns the CUDA inference worker); `-o <dir>` dumps
per-step states/actions for seam inspection. Compare the `success_rate` and
`avg_path_length` that `get_test_stats` logs at the end of each run, and diff the
dumped action sequences at chunk boundaries.

Notes:
- Isaac Sim is the constraint here — run on whichever machine has it (CML node
  with the sim installed, or NCHC). The client/`-o` artifacts are small.
- `pigdm` adds per-step autodiff latency; with `dt_scale` latency simulation in
  `evaluate.py` this is reflected in the throughput numbers.
