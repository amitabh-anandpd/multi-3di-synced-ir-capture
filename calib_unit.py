"""N0: stereo calibration of each 3Di's left / right imagers from checkerboard sessions.

  python calib_unit.py --fit sessions/2026-11-06_011406 --test sessions/2026-11-05_004009 --square 25

LeapC's distortion map turns each imager into a pinhole view (stereo.Cam). LeapC treats the two
imagers as parallel, but each unit's depth reads short with a ~3 px disparity offset (STATUS.md),
which is what a small rotation between the imagers does. This fits that rotation (and the baseline)
with cv2.stereoCalibrate on the board corners seen in both imagers, holding the pinhole K fixed,
and checks it on a session not used for fitting:
  - square size from triangulated corners (exit: within +-1 % p10-p90)
  - row error between L and R after rectification (exit: < 0.5 px)
Writes units/<serial>.json. Corners are cached per session and camera (<session>/<cam>/corners2d_*.npz).
"""
import argparse
import json
import os

import cv2
import numpy as np

from stereo import GRID, Cam, board_2d, board_object_points, pinhole_K

ap = argparse.ArgumentParser()
ap.add_argument("--fit", nargs="+", required=True, help="board sessions to fit on")
ap.add_argument("--test", nargs="*", default=[], help="board sessions to check on (not used for fitting)")
ap.add_argument("--square", type=float, required=True, help="measured square size, mm")
ap.add_argument("--cams", nargs="+", default=["cam_A", "cam_B"])
ap.add_argument("--stride", type=int, default=12, help="use every Nth frame")
ap.add_argument("--size", type=int, default=800)
ap.add_argument("--slope", type=float, default=1.5)
ap.add_argument("--free-K", action="store_true", help="also refine focal length and centre of each imager")
ap.add_argument("--free-dist", action="store_true",
                help="with --free-K: also fit radial distortion k1, k2 left over after the LeapC map")
args = ap.parse_args()

K0 = pinhole_K(args.size, args.slope)
SZ = (args.size, args.size)
obj = board_object_points(args.square)


def corners(session, name):
    """[(pl, pr)] for every --stride-th frame where both imagers see the board (cached)."""
    cam = Cam(session, name, args.size, args.slope)
    cache = os.path.join(cam.dir, f"corners2d_s{args.stride}_{args.size}_{args.slope}{'_dense' if cam.dense else ''}.npz")
    if os.path.exists(cache):
        c = np.load(cache)["c"]
    else:
        found = [board_2d(cam, int(r["frame_id"])) for r in cam.rows[::args.stride]]
        c = np.array([np.stack(x) for x in found if x is not None]).reshape(-1, 2, GRID[0] * GRID[1], 2)
        np.savez_compressed(cache, c=c)
    return cam, [(v[0].astype(np.float32), v[1].astype(np.float32)) for v in c]


def evaluate(views, K1, K2, R, T, D1=None, D2=None):
    """Square size error (fractions, per view) and |row error| (px, per corner) after rectification."""
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K1, D1, K2, D2, SZ, R, T, alpha=0, flags=0)
    err, rows = [], []
    for pl, pr in views:
        ul = cv2.undistortPoints(pl.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(pr.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
        rows.append(np.abs(ul[:, 1] - ur[:, 1]))
        X = cv2.triangulatePoints(P1, P2, ul.T, ur.T)
        X = (X[:3] / X[3]).T.reshape(GRID[1], GRID[0], 3)
        e = np.concatenate([np.linalg.norm(np.diff(X, axis=0), axis=-1).ravel(),
                            np.linalg.norm(np.diff(X, axis=1), axis=-1).ravel()])
        err.append(np.median(e) / args.square - 1)
    return np.array(err), np.concatenate(rows)


def report(label, views, K1, K2, R, T, D1=None, D2=None):
    err, rows = evaluate(views, K1, K2, R, T, D1, D2)
    p10, p50, p90 = np.percentile(err, [10, 50, 90]) * 100
    print(f"  {label:<22} squares {p50:+5.1f} % (p10 {p10:+5.1f}, p90 {p90:+5.1f}), "
          f"row error median {np.median(rows):.2f} px, p95 {np.percentile(rows, 95):.2f} px")
    return p10, p90, np.median(rows)


os.makedirs("units", exist_ok=True)
for name in args.cams:
    fit, test = [], []
    for s in args.fit:
        cam, v = corners(s, name)
        fit += v
    for s in args.test:
        _, v = corners(s, name)
        test += v
    print(f"\n{name} ({cam.serial}): {len(fit)} fit views, {len(test)} test views")
    if len(fit) < 10:
        print("  too few views, skipped")
        continue
    flags = cv2.CALIB_FIX_INTRINSIC
    if args.free_K:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
                 | cv2.CALIB_FIX_ASPECT_RATIO)
        if not args.free_dist:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K1, D1, K2, D2, R, T, *_ = cv2.stereoCalibrate(
        [obj] * len(fit), [v[0] for v in fit], [v[1] for v in fit], K0.copy(), np.zeros(5), K0.copy(), np.zeros(5),
        SZ, flags=flags, criteria=crit)
    rvec = cv2.Rodrigues(R)[0].ravel()
    print(f"  fit: reprojection rms {rms:.2f} px, baseline {np.linalg.norm(T):.2f} mm (LeapC {cam.baseline:.2f}), "
          f"rotation L->R {np.degrees(np.linalg.norm(rvec)):.2f} deg, axis {np.round(rvec / np.linalg.norm(rvec), 2)}")
    if args.free_K:
        print(f"  K left f {K1[0, 0]:.1f} c ({K1[0, 2]:.1f}, {K1[1, 2]:.1f}); right f {K2[0, 0]:.1f} "
              f"c ({K2[0, 2]:.1f}, {K2[1, 2]:.1f}); start f {K0[0, 0]:.1f} c {K0[0, 2]:.1f}; "
              f"k1, k2 left {D1.ravel()[:2].round(4)}, right {D2.ravel()[:2].round(4)}")
    T_leap = np.array([[-cam.baseline], [0], [0]])
    for label, views in (("fit", fit), ("test", test)):
        if not views:
            continue
        print(f" {label} views:")
        report("LeapC (parallel)", views, K0, K0, np.eye(3), T_leap)
        res = report("calibrated", views, K1, K2, R, T, D1, D2)
    if test:
        ok = -1 <= res[0] and res[1] <= 1 and res[2] < 0.5
        print(f"  N0 exit on test views: {'PASS' if ok else 'not yet'}")
    out = {"serial": cam.serial, "camera": name, "size": args.size, "slope": args.slope,
           "matrix_version": cam.matrix_version, "K_left": K1.tolist(), "K_right": K2.tolist(),
           "D_left": D1.ravel().tolist(), "D_right": D2.ravel().tolist(),
           "R": R.tolist(), "T_mm": T.ravel().tolist(), "rms_px": rms, "fit_sessions": args.fit,
           "test_sessions": args.test, "views": len(fit), "square_mm": args.square, "free_K": args.free_K, "free_dist": args.free_dist,
           "note": "X_right = R @ X_left + T, both in the LeapC-map pinhole frames of the rectified images"}
    path = os.path.join("units", f"{cam.serial}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("  wrote", path)
