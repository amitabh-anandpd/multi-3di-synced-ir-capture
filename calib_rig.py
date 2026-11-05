"""N1: rig calibration. Board session -> pose of every camera in cam_A's frame -> rig.json.

  python calib_rig.py sessions/2026-11-06_011406 --check sessions/2026-11-06_013241 sessions/2026-11-06_021241

Each camera's board corners are triangulated with its own unit calibration (units/<serial>.json,
calib_unit.py) in its rectified-left frame (stereo.Unit). Only groups where the board moves < --still mm
between the two cameras' exposures are used. The 9x6 grid looks the same turned 180 deg, so each
view's corner order is chosen by consensus. Writes <board>/rig.json and ./rig.json (the current rig).

--check: merge both cameras' SGBM clouds of static-object sessions with the new rig and report the
gap (cam_B points to the nearest cam_A point, over points within --overlap mm); writes
<object>/rig_merge.ply (cam_A red, cam_B green).
"""
import argparse
import json
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from stereo import GRID, Cam, Unit, angle_deg, board_2d, kabsch, write_ply

ap = argparse.ArgumentParser()
ap.add_argument("board", help="checkerboard session, both cameras simultaneous")
ap.add_argument("--check", nargs="*", default=[], help="static-object sessions to merge with the new rig")
ap.add_argument("--ref", default="cam_A")
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--still", type=float, default=0.3, help="mm the board may move between the two exposures")
ap.add_argument("--overlap", type=float, default=15)
ap.add_argument("--frames", type=int, default=10, help="--check: frames fused per camera")
ap.add_argument("--force", action="store_true", help="make this the current rig.json even from a weak board session")
args = ap.parse_args()

meta = json.load(open(os.path.join(args.board, "session.json")))
names = sorted(meta["cameras"])
other = [n for n in names if n != args.ref]
if len(other) != 1:
    raise SystemExit("rig fitting handles two cameras for now")
other = other[0]
units = {n: Unit(meta["cameras"][n]) for n in names}
size, slope = units[args.ref].size, units[args.ref].slope
cams = {n: Cam(args.board, n, size, slope) for n in names}
if not all(c.dense for c in cams.values()):
    print("WARNING: no dense map for some camera (tools/dense_map.py); the units were fitted on dense maps")

# --- board corners, both cameras, per sampled group (2D cached: finding them takes minutes)
groups = json.load(open(os.path.join(args.board, "groups.json")))["groups"][::args.stride]
cache = os.path.join(args.board, f"rig_corners2d_s{args.stride}_{size}_{slope}.npz")
if os.path.exists(cache):
    c2 = np.load(cache)["c"]
else:
    c2 = np.full((len(groups), 2, 2, GRID[0] * GRID[1], 2), np.nan, np.float32)  # group, cam (ref, other), L/R
    for i, g in enumerate(groups):
        a = board_2d(cams[args.ref], g["frames"][args.ref])
        b = board_2d(cams[other], g["frames"][other]) if a is not None else None
        if b is not None:
            c2[i] = [np.stack(a), np.stack(b)]
    np.savez_compressed(cache, c=c2)
seen = np.isfinite(c2[:, 0, 0, 0, 0])
pts = np.full((len(groups), 2, GRID[0] * GRID[1], 3), np.nan)
for i in np.flatnonzero(seen):
    pts[i, 0] = units[args.ref].triangulate(*c2[i, 0])
    pts[i, 1] = units[other].triangulate(*c2[i, 1])

pairs = []
for i in range(1, len(groups)):
    if not (seen[i] and seen[i - 1]):
        continue
    (pa, pb), (qa, qb) = pts[i], pts[i - 1]
    move = max(np.median(np.linalg.norm(pa - qa, axis=1)),
               min(np.median(np.linalg.norm(pb - qb, axis=1)), np.median(np.linalg.norm(pb[::-1] - qb, axis=1))))
    slip = move * abs(groups[i]["gap_us"][other]) / (groups[i]["t_us"] - groups[i - 1]["t_us"])
    if slip < args.still:
        pairs.append((pa, pb))
spread = np.ptp(np.array([p[0].mean(0) for p in pairs]), axis=0) if pairs else np.zeros(3)
print(f"board: {len(groups)} groups sampled, seen by both in {seen.sum()}, still enough in {len(pairs)}, "
      f"positions span {spread[0]:.0f} x {spread[1]:.0f} x {spread[2]:.0f} mm")
if len(pairs) < 5:
    raise SystemExit("too few still board views")

# --- corner order by consensus, then one fit over the agreeing views, dropping outlier views
cands = [[(*kabsch(o, pa), o) for o in (pb, pb[::-1])] for pa, pb in pairs]
best, support = None, -1
for cs in cands:
    for R0, t0, _ in cs:
        n = sum(any(angle_deg(R @ R0.T) < 3 and np.linalg.norm(t - t0) < 30 for R, t, _ in c) for c in cands)
        if n > support:
            best, support = (R0, t0), n
src, dst = [], []
for (pa, _), cs in zip(pairs, cands):
    R, t, o = min(cs, key=lambda c: angle_deg(c[0] @ best[0].T))
    if angle_deg(R @ best[0].T) < 3 and np.linalg.norm(t - best[1]) < 30:
        src.append(o)
        dst.append(pa)
for _ in range(3):
    R, t = kabsch(np.concatenate(src), np.concatenate(dst))
    med = np.array([np.median(np.linalg.norm(s @ R.T + t - d, axis=1)) for s, d in zip(src, dst)])
    keep = med < max(3 * np.median(med), 1.0)
    src = [s for s, k in zip(src, keep) if k]
    dst = [d for d, k in zip(dst, keep) if k]
R, t = kabsch(np.concatenate(src), np.concatenate(dst))
res = np.linalg.norm(np.concatenate(src) @ R.T + t - np.concatenate(dst), axis=1)
print(f"{other} -> {args.ref}: {len(src)} views, rotation {angle_deg(R):.1f} deg, translation {np.linalg.norm(t):.0f} mm")
print(f"corner residual: median {np.median(res):.2f} mm, p95 {np.percentile(res, 95):.2f} mm  (N1 exit: median <= 2 mm)")

rig = {"board_session": args.board, "reference": args.ref, "views": len(src),
       "corner_residual_median_mm": float(np.median(res)), "corner_residual_p95_mm": float(np.percentile(res, 95)),
       "cameras": {args.ref: {"serial": meta["cameras"][args.ref], "R": np.eye(3).tolist(), "t_mm": [0, 0, 0]},
                   other: {"serial": meta["cameras"][other], "R": R.tolist(), "t_mm": t.tolist()}},
       "note": "X_ref = R @ X_cam + t, each X in its unit's rectified-left frame (stereo.Unit), mm"}
path = os.path.join(args.board, "rig.json")
json.dump(rig, open(path, "w"), indent=2)
weak = len(src) < 20 or spread.max() < 50
if weak and not args.force:
    print(f"wrote {path}; NOT copied to ./rig.json: weak board session ({len(src)} views, positions span "
          f"{spread.max():.0f} mm; want >= 20 views over >= 50 mm). --force copies it anyway.")
else:
    shutil.copy(path, "rig.json")
    print("wrote", path, "and rig.json")

# --- merge check on static objects
for s in args.check:
    og = json.load(open(os.path.join(s, "groups.json")))["groups"]
    mid = len(og) // 2
    sel = og[max(0, mid - args.frames // 2):][:args.frames]
    clouds = {}
    for n in names:
        cam = Cam(s, n, size, slope, pad=256)
        clouds[n], _ = units[n].cloud(cam, [g["frames"][n] for g in sel])
    a, b = clouds[args.ref], clouds[other] @ R.T + t
    d, _ = cKDTree(a).query(b, distance_upper_bound=args.overlap)
    ov = np.isfinite(d)
    print(f"{s}: {len(a)} / {len(b)} points, {ov.mean() * 100:.1f} % overlap, "
          f"gap median {np.median(d[ov]):.2f} mm, p90 {np.percentile(d[ov], 90):.2f} mm  (N1 exit: median <= 3 mm)")
    write_ply(os.path.join(s, "rig_merge.ply"), np.vstack([a, b]),
              np.vstack([np.tile([255, 60, 60], (len(a), 1)), np.tile([60, 255, 60], (len(b), 1))]))
