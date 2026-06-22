import importlib.util
import os

import numpy as np
import pytest

# load scripts/rtc_seam_metric.py by path (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "rtc_seam_metric",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "rtc_seam_metric.py"),
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_straight_line_has_zero_jerk():
    # constant-velocity straight line -> second difference is 0
    t = np.arange(50)[:, None]
    pos = np.hstack([t * 0.01, t * 0.0, t * 0.0])  # (50,3) moving in x
    r = m.compute_jerk(pos)
    assert r["n"] == 50
    assert r["mean_jerk"] < 1e-9
    assert r["mean_step"] == pytest.approx(0.01, abs=1e-9)


def test_zigzag_has_more_jerk_than_smooth():
    t = np.linspace(0, 1, 100)
    smooth = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    zig = smooth.copy()
    zig[::2, 1] += 0.05  # alternating offset -> high second difference
    assert m.compute_jerk(zig)["mean_jerk"] > 10 * m.compute_jerk(smooth)["mean_jerk"]


def test_short_sequence_safe():
    r = m.compute_jerk(np.zeros((1, 3)))
    assert r["n"] == 1 and r["mean_jerk"] == 0.0
