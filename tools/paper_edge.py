"""PLAN.md P2: does stereo make bright edges stick out (and so the fingertips look too long)?

  python tools/paper_edge.py sessions/2026-11-12_103212 sessions/2026-11-12_103250

Uses the checkerboard's white paper margin against a dark background. Per camera and frame:
  - the board corners give the board's pose and an exact mm scale (25 mm squares; the cloud is rescaled by it);
  - image edge: where the brightness, sampled along lines running out from the printed squares, falls halfway from
    the white margin to the background (a single image's brightness edge is not inflated by stereo matching);
  - cloud edge: the 95th (and 99th) percentile outward of the stereo points (SGBM, same setting as the hand clouds)
    on the board plane along the same lines - the same measure as the fingertip overshoot (STATUS.md N7);
  - inflation = cloud edge - image edge, mm (positive: the cloud sticks out past the paper).
Sides where the background is not dark (the hand holding it, the desk) are dropped.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth import Depth  # noqa: E402
from stereo import GRID, Cam, Unit, board_2d, board_object_points, kabsch  # noqa: E402

SQ = 25.0
HALF = np.array([(GRID[0] - 1) * SQ / 2, (GRID[1] - 1) * SQ / 2])  # grid centre, board mm
OUTER = HALF + SQ  # outer edge of the printed squares (centred board mm)


def measure(session, name, frames=8, lr=None, erode=0):
    cam_c = Cam(session, name, 800, 1.5)            # corners (LeapC pinhole images)
    cam_d = Cam(session, name, 800, 1.5, pad=256)   # depth
    unit = Unit(cam_c.serial)
    dep = Depth(unit, cam_d)
    if lr is not None:  # left-right consistency check (off by default; the 3-way mode ignores it)
        dep.sgbm.setMode(cv2.STEREO_SGBM_MODE_SGBM)
        dep.sgbm.setDisp12MaxDiff(lr)
    obj = board_object_points(SQ)[:, :2] - HALF
    ids = [int(r["frame_id"]) for r in cam_c.rows][::max(1, len(cam_c.rows) // (frames * 4))]
    res = {k: [] for k in ("u-", "u+", "v-", "v+")}
    scales, used = [], 0
    for fid in ids:
        if used >= frames:
            break
        b = board_2d(cam_c, fid)
        if b is None:
            continue
        pl, pr = b
        P3 = unit.triangulate(pl, pr)
        g = P3.reshape(GRID[1], GRID[0], 3)
        s = np.mean(np.r_[np.linalg.norm(np.diff(g, axis=1), axis=-1).ravel(),
                          np.linalg.norm(np.diff(g, axis=0), axis=-1).ravel()]) / SQ
        R, t = kabsch(np.c_[obj, np.zeros(len(obj))] * s, P3)
        ul = cv2.undistortPoints(pl.reshape(-1, 1, 2).astype(np.float64), unit.K1, unit.D1, R=unit.R1,
                                 P=unit.P1).reshape(-1, 2)
        H, _ = cv2.findHomography(obj, ul)
        # cloud of the paper area
        corners = cv2.perspectiveTransform(np.array([[[-200, -170], [200, -170], [200, 170], [-200, 170]]],
                                                    np.float64), H)[0]
        x0, y0 = np.clip(corners.min(0).astype(int), 0, unit.size - 1)
        x1, y1 = np.clip(corners.max(0).astype(int), 1, unit.size)
        disp, L = dep.disparity(fid, roi=(y0, y1, x0, x1))
        if erode:  # drop the outer `erode` pixels of every region with depth
            ok = cv2.erode(np.isfinite(disp).astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
            disp = np.where(ok, disp, np.nan)
        P = dep.points(disp)
        Q = ((P - t) @ R) / s  # board mm, centred
        Q = Q[np.abs(Q[:, 2]) < 6]
        for side in res:
            ax, sign = (0 if side[0] == "u" else 1), (-1 if side[1] == "-" else 1)
            other = 1 - ax
            span = OUTER[other] - SQ * 1.6
            lines = np.linspace(-span, span, 9)
            out = np.arange(OUTER[ax] + 2, OUTER[ax] + 90, 0.5)  # outward from the printed squares
            edges = []
            for c in lines:
                uv = np.zeros((len(out), 2))
                uv[:, ax], uv[:, other] = sign * out, c
                px = cv2.perspectiveTransform(uv[None], H)[0].astype(np.float32)
                prof = cv2.remap(L, px[:, 0].reshape(1, -1), px[:, 1].reshape(1, -1), cv2.INTER_LINEAR,
                                 borderValue=0).ravel().astype(float)
                white, dark = np.median(prof[:6]), np.median(prof[-20:])
                if white - dark < 40:
                    continue
                below = np.nonzero(prof < (white + dark) / 2)[0]
                if not len(below) or below[0] == 0:
                    continue
                k = below[0]
                f = (prof[k - 1] - (white + dark) / 2) / max(prof[k - 1] - prof[k], 1e-6)
                edges.append(out[k - 1] + f * 0.5)
            band = (np.abs(Q[:, other]) <= span) & (sign * Q[:, ax] > OUTER[ax])
            if len(edges) >= 3 and band.sum() > 50:
                pos = sign * Q[band, ax]
                res[side].append((np.median(edges), np.percentile(pos, 95), np.percentile(pos, 99), band.sum()))
        scales.append(s)
        used += 1
    return res, scales, used


ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="+")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--lr", type=int, help="SGBM left-right check: max disparity difference in px (off by default)")
ap.add_argument("--cams", nargs="+", default=["cam_A", "cam_B"])
ap.add_argument("--erode", type=int, default=0, help="drop the outer N pixels of every region with depth")
args = ap.parse_args()
allinf = {c: [] for c in args.cams}
for s in args.sessions:
    for name in args.cams:
        res, scales, used = measure(s, name, args.frames, args.lr, args.erode)
        print(f"{os.path.basename(s)} {name}: {used} frames, corner scale {np.mean(scales) if scales else float('nan'):.4f}")
        for side, v in res.items():
            if not v:
                print(f"   {side}: no usable edge (background not dark / out of view)")
                continue
            a = np.array(v)
            inf = a[:, 1] - a[:, 0]
            allinf[name] += inf.tolist()
            print(f"   {side}: paper edge (image) {np.median(a[:, 0]):6.1f} mm, cloud edge p95 {np.median(a[:, 1]):6.1f} "
                  f"(p99 {np.median(a[:, 2]):6.1f}), inflation {np.median(inf):+5.1f} mm "
                  f"(p99: {np.median(a[:, 2] - a[:, 0]):+5.1f})  [{len(v)} frames, ~{int(np.median(a[:, 3]))} pts]")
for name, v in allinf.items():
    if v:
        print(f"{name}: edge inflation median {np.median(v):+.1f} mm (p10 {np.percentile(v, 10):+.1f}, "
              f"p90 {np.percentile(v, 90):+.1f}) over {len(v)} side-frames")
