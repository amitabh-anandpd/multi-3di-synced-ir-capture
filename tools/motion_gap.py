"""N5: does the hand's motion between the two cameras' exposures show in the merged hand cloud?

  python tools/motion_gap.py sessions/2026-11-06_035809 sessions/2026-11-06_035901 sessions/2026-11-06_035942

For every --every-th frame group: both cameras' single-frame hand points (segment.py), cam_B's moved
into cam_A's frame with rig.json, gap = median distance from cam_B hand points to the nearest cam_A
hand point (within --overlap mm). Hand motion between the exposures = 95th-percentile joint speed
(LeapC tracking of cam_A, else cam_B) x |cam_B - cam_A exposure gap|. Prints the gap binned by that
motion; N5 exit: gap on moving hands <= still-hand gap + 1 mm.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment import Segmenter  # noqa: E402
from stereo import rig_path  # noqa: E402


def joint_speed(cam_dir):
    """frame_id -> 95th-percentile joint speed (mm/s) from tracking.csv (first hand), by central difference."""
    rows = list(csv.DictReader(open(os.path.join(cam_dir, "tracking.csv"))))
    by = {}
    for r in rows:
        by.setdefault(int(r["frame_id"]), r)
    ids = sorted(by)
    J = {i: np.array([[float(by[i][f"j{k:02d}_{a}"]) for a in "xyz"] for k in range(25)]) for i in ids}
    t = {i: int(by[i]["device_ts_us"]) for i in ids}
    out = {}
    for a, i, b in zip(ids, ids[1:], ids[2:]):
        if b - a > 4:  # tracking gap
            continue
        v = np.linalg.norm(J[b] - J[a], axis=1) / ((t[b] - t[a]) / 1e6)
        out[i] = float(np.percentile(v, 95))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="+")
ap.add_argument("--every", type=int, default=9, help="use every Nth frame group (~10 per second)")
ap.add_argument("--overlap", type=float, default=15)
args = ap.parse_args()
rows = []
for s in args.sessions:
    rig = json.load(open(rig_path(s)))  # the session's pinned rig, else the current one
    Rb, tb = np.array(rig["cameras"]["cam_B"]["R"]), np.array(rig["cameras"]["cam_B"]["t_mm"])
    note = json.load(open(os.path.join(s, "session.json")))["notes"]
    groups = json.load(open(os.path.join(s, "groups.json")))["groups"]
    sa, sb = Segmenter(s, "cam_A"), Segmenter(s, "cam_B")
    spA, spB = joint_speed(sa.cam.dir), joint_speed(sb.cam.dir)
    for g in groups[::args.every]:
        fa, fb = g["frames"]["cam_A"], g["frames"]["cam_B"]
        speed = spA.get(fa, spB.get(fb))
        if speed is None:
            continue
        a, b = sa.points(fa), sb.points(fb)
        if len(a) < 200 or len(b) < 200:
            continue
        b = b @ Rb.T + tb
        d, _ = cKDTree(a).query(b, distance_upper_bound=args.overlap)
        ov = np.isfinite(d)
        if ov.mean() < 0.1:
            continue
        dt = abs(g["gap_us"]["cam_B"]) / 1e6
        rows.append((s[-6:], note, speed, dt * 1e3, speed * dt, float(np.median(d[ov])), ov.mean()))
    print(f"{s}: {sum(r[0] == s[-6:] for r in rows)} groups used", flush=True)

r = np.array([x[2:] for x in rows], float)
print(f"\n{'session':<8}{'pose':<20}{'groups':>7}{'speed mm/s':>12}{'|gap| ms':>9}{'motion mm':>10}{'cloud gap mm':>13}{'overlap':>8}")
for sid in dict.fromkeys(x[0] for x in rows):
    m = np.array([x[0] == sid for x in rows])
    note = next(x[1] for x in rows if x[0] == sid)
    print(f"{sid:<8}{note[3:]:<20}{m.sum():>7}{np.median(r[m, 0]):>12.0f}{np.median(r[m, 1]):>9.2f}"
          f"{np.median(r[m, 2]):>10.2f}{np.median(r[m, 3]):>13.2f}{np.median(r[m, 4]) * 100:>7.0f}%")
print("\nby hand motion between the two exposures:")
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
    m = (r[:, 2] >= lo) & (r[:, 2] < hi)
    if m.any():
        print(f"  {lo:>4g} - {hi if hi < 1e9 else 'more':<4} mm: {m.sum():>4} groups, cloud gap median {np.median(r[m, 3]):.2f} mm")
still = r[:, 2] < 0.25
if still.any() and (~still).any():
    print(f"\nstill (< 0.25 mm motion) {np.median(r[still, 3]):.2f} mm vs moving {np.median(r[~still, 3]):.2f} mm "
          f"(N5 exit: moving <= still + 1 mm); correlation of gap with motion {np.corrcoef(r[:, 2], r[:, 3])[0, 1]:+.2f}")
