# -*- coding: utf-8 -*-
# RTC seam-jerk metric: quantifies executed-trajectory smoothness per rtc_mode.
# Jerk proxy = magnitude of the 2nd difference of the end-effector position stream.
import argparse
import glob
import os
import pickle

import numpy as np


def compute_jerk(positions: np.ndarray) -> dict:
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return {"mean_jerk": 0.0, "p95_jerk": 0.0,
                "mean_step": 0.0, "n": int(p.shape[0]) if p.ndim else 0}
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)        # |Δpos|     (T-1,)
    jerk = np.linalg.norm(np.diff(p, axis=0, n=2), axis=1)   # |Δ²pos|    (T-2,)
    return {"mean_jerk": float(jerk.mean()),
            "p95_jerk": float(np.percentile(jerk, 95)),
            "mean_step": float(step.mean()),
            "n": int(p.shape[0])}


def load_episode_positions(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    acts = np.asarray(d["action"], dtype=np.float64)         # (T,1,8) or (T,8)
    acts = acts.reshape(acts.shape[0], -1)                    # (T,8)
    return acts[:, :3]                                        # position


def aggregate_mode(dump_dir: str):
    pkls = sorted(glob.glob(os.path.join(dump_dir, "**", "*.pkl"), recursive=True))
    per = [compute_jerk(load_episode_positions(p)) for p in pkls]
    per = [x for x in per if x["n"] >= 3]
    if not per:
        return None
    return {"episodes": len(per),
            "mean_jerk": float(np.mean([x["mean_jerk"] for x in per])),
            "p95_jerk": float(np.mean([x["p95_jerk"] for x in per])),
            "mean_step": float(np.mean([x["mean_step"] for x in per]))}


def main():
    ap = argparse.ArgumentParser(description="RTC seam-jerk comparison")
    ap.add_argument("--mode", action="append", required=True,
                    help="label:dump_dir (repeatable), e.g. off:output/dumps-off")
    args = ap.parse_args()
    print(f"{'mode':<10}{'episodes':>9}{'mean_jerk':>12}{'p95_jerk':>12}{'mean_step':>12}")
    for spec in args.mode:
        label, d = spec.split(":", 1)
        a = aggregate_mode(d)
        if a is None:
            print(f"{label:<10}{'(no dumps)':>9}")
            continue
        print(f"{label:<10}{a['episodes']:>9}{a['mean_jerk']:>12.5f}"
              f"{a['p95_jerk']:>12.5f}{a['mean_step']:>12.5f}")


if __name__ == "__main__":
    main()
