"""M7b: fit the cam_B -> cam_A rigid transform from a checkerboard session, then merge both
cameras' point clouds of a static object and measure the gap where they overlap.

  python tools/merge_check.py sessions/2026-11-04_235105 sessions/2026-11-04_235306 --square 25

Board: corners are found on each camera's rectified L and R images (same virtual pinhole as
depth_preview.py) and triangulated, z = f * baseline / disparity, in that camera's left-imager
frame (x right, y down, z forward, mm). The cameras run up to half a frame apart, so only groups
where the board moves < --still mm between the two exposures are used.
The 9x6 corner grid looks the same turned 180 deg, so each view's corner order is chosen by
consensus: the transform most frames agree on wins.

Object: SGBM cloud per camera (median of --frames frames), cam_B's moved into cam_A's frame.
Gap = distance from each cam_B point to the nearest cam_A point, over points within --overlap mm.
Writes <object>/merge/: merged.ply (cam_A red, cam_B green), transform.json.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_io import load_rows, load_session, read_pair  # noqa: E402

GRID = (9, 6)  # inner corners of a 10 x 7 squares board


def rectify_maps(dmap, size, slope, raw_wh, pad=0):
    """Same as depth_preview.rectify_maps: virtual pinhole of +-slope rays -> raw image pixel maps."""
    f = size / 2 / slope
    ax = (np.arange(-pad, size) + 0.5 - size / 2) / f
    ay = (np.arange(size) + 0.5 - size / 2) / f
    rx, ry = np.meshgrid(ax, ay)
    gc = ((rx + 4) / 8 * 63).astype(np.float32)
    gr = ((4 - ry) / 8 * 63).astype(np.float32)
    nx = cv2.remap(np.ascontiguousarray(dmap[..., 0]), gc, gr, cv2.INTER_LINEAR)
    ny = cv2.remap(np.ascontiguousarray(dmap[..., 1]), gc, gr, cv2.INTER_LINEAR)
    return nx * raw_wh[0] - 0.5, ny * raw_wh[1] + 0.5, f


class Cam:
    def __init__(self, session, name, size, slope, pad=0):
        self.meta = load_session(session)
        self.dir = os.path.join(session, name)
        self.rows = load_rows(self.dir)
        self.by_id = {int(r["frame_id"]): r for r in self.rows}
        self.wh = tuple(self.meta["image_size"])
        cal = self.meta["calibration"][name]
        self.baseline = cal["right"]["extrinsic"][0][3] - cal["left"]["extrinsic"][0][3]
        versions = {r["matrix_version"] for r in self.rows}
        if len(versions) > 1:
            print(f"{name}: distortion map changed during the session {sorted(versions)}, using the first")
        dm = np.load(os.path.join(self.dir, f"distortion_v{self.rows[0]['matrix_version']}.npz"))
        self.maps = []
        for side in ("left", "right"):
            mx, my, self.f = rectify_maps(dm[side], size, slope, self.wh, pad)
            self.maps.append((mx, my))
        self.c = size / 2

    def rect(self, frame_id):
        lr = read_pair(self.dir, self.by_id[frame_id], self.wh)
        return [cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0) for img, (mx, my) in zip(lr, self.maps)]


def find(img):
    # no CALIB_CB_FAST_CHECK: it rejects these dim IR boards
    ok, pts = cv2.findChessboardCorners(img, GRID, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    pts = cv2.cornerSubPix(img, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return pts.reshape(-1, 2)


def board_3d(cam, frame_id):
    L, R = cam.rect(frame_id)
    pl = find(L)
    if pl is None:
        return None
    pr = find(R)
    if pr is None:
        return None
    if np.abs(pl[:, 1] - pr[::-1, 1]).mean() < np.abs(pl[:, 1] - pr[:, 1]).mean():
        pr = pr[::-1]  # L and R found the grid in opposite orders
    d = pl[:, 0] - pr[:, 0]
    if np.abs(pl[:, 1] - pr[:, 1]).max() > 5 or (d <= 0).any():  # coarse distortion map leaves ~2 px row error
        return None
    z = cam.f * cam.baseline / d
    x = (pl[:, 0] + 0.5 - cam.c) / cam.f * z
    y = ((pl[:, 1] + pr[:, 1]) / 2 + 0.5 - cam.c) / cam.f * z
    return np.stack([x, y, z], 1)


def kabsch(src, dst):
    """R, t minimising |R @ src + t - dst|."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    s = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    R = vt.T @ s @ u.T
    return R, cd - R @ cs


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def sgbm_cloud(cam, ids, max_disp, zmin, zmax, min_gray, min_hits, pad):
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=5, P1=8 * 25, P2=32 * 25,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disps = []
    for i in ids:
        L, R = cam.rect(i)
        d = sgbm.compute(L, R).astype(np.float32)[:, pad:] / 16
        d[d <= 0] = np.nan
        disps.append(d)
        ref = L[:, pad:]
    stack = np.stack(disps)
    keep = np.isfinite(stack).mean(0) >= min_hits
    disp = np.zeros_like(disps[0])
    disp[keep] = np.nanmedian(stack[:, keep], axis=0)
    v, u = np.indices(disp.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = cam.f * cam.baseline / disp
    ok = (disp > 0) & (z > zmin) & (z < zmax) & (ref >= min_gray)
    x = (u + 0.5 - cam.c) / cam.f * z
    y = (v + 0.5 - cam.c) / cam.f * z
    return np.stack([x[ok], y[ok], z[ok]], 1)


def write_ply(path, xyz, rgb):
    xyz = xyz * [1, -1, -1]  # same flip as depth_preview.py so viewers show it upright
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(fh, np.hstack([xyz, rgb]), fmt="%.2f %.2f %.2f %d %d %d")


ap = argparse.ArgumentParser()
ap.add_argument("board", help="checkerboard session (both cameras)")
ap.add_argument("object", help="static rigid object session (both cameras)")
ap.add_argument("--square", type=float, required=True, help="measured square size, mm (used for the scale check)")
ap.add_argument("--stride", type=int, default=6, help="use every Nth frame group of the board session")
ap.add_argument("--still", type=float, default=0.3,
                help="mm: max board motion in the time between cam_A's and cam_B's frame of a group")
ap.add_argument("--scale-fix", action="store_true",
                help="scale each camera's points so the board squares measure --square (corrects stereo depth scale)")
ap.add_argument("--size", type=int, default=800, help="rectified image size, px")
ap.add_argument("--slope", type=float, default=1.5, help="half field of view as ray slope (1.5 = +-56 deg)")
ap.add_argument("--frames", type=int, default=10, help="object: frames fused per camera")
ap.add_argument("--max-disp", type=int, default=256)
ap.add_argument("--zmin", type=float, default=150)
ap.add_argument("--zmax", type=float, default=600)
ap.add_argument("--min-gray", type=int, default=40)
ap.add_argument("--min-hits", type=float, default=0.3)
ap.add_argument("--overlap", type=float, default=15, help="mm: cam_B points this close to a cam_A point count as overlap")
args = ap.parse_args()

# --- 1. board corners in 3D, per still group
groups = json.load(open(os.path.join(args.board, "groups.json")))["groups"]
A, B = Cam(args.board, "cam_A", args.size, args.slope), Cam(args.board, "cam_B", args.size, args.slope)
cache = os.path.join(args.board, f"board_corners_s{args.stride}_{args.size}_{args.slope}.npz")
if os.path.exists(cache):  # corner finding takes minutes; reuse it
    corners = np.load(cache)["corners"]
else:
    nan = np.full((2, GRID[0] * GRID[1], 3), np.nan)
    corners = []
    for g in groups[::args.stride]:
        pa = board_3d(A, g["frames"]["cam_A"])
        pb = board_3d(B, g["frames"]["cam_B"]) if pa is not None else None
        corners.append(nan if pb is None else np.stack([pa, pb]))
    corners = np.array(corners)
    np.savez_compressed(cache, corners=corners)
seen = np.isfinite(corners[:, 0, 0, 0])
sampled = groups[::args.stride]
pairs, slips = [], []
for i in range(1, len(corners)):
    if not (seen[i] and seen[i - 1]):
        continue
    (pa, pb), (qa, qb) = corners[i], corners[i - 1]
    move = max(np.median(np.linalg.norm(pa - qa, axis=1)),
               min(np.median(np.linalg.norm(pb - qb, axis=1)), np.median(np.linalg.norm(pb[::-1] - qb, axis=1))))
    # how far the board moves in the time between cam_A's and cam_B's exposure of this group
    slip = move * abs(sampled[i]["gap_us"]["cam_B"]) / (sampled[i]["t_us"] - sampled[i - 1]["t_us"])
    slips.append(slip)
    if slip < args.still:
        pairs.append((pa, pb))
print(f"board: {len(corners)} groups sampled, seen by both cameras in {seen.sum()}, still enough in {len(pairs)} "
      f"(median board motion during the camera gap {np.median(slips):.2f} mm)")
if len(pairs) < 5:
    raise SystemExit("too few still board views to fit the transform")
spread = np.ptp(np.array([q[0].mean(0) for q in pairs]), axis=0)
print(f"board positions span {spread[0]:.0f} x {spread[1]:.0f} x {spread[2]:.0f} mm (x, y, z in cam_A)")
if spread.max() < 50:
    print("WARNING: the board hardly moved. One pose gives a weak fit; record it again moving it through ~15 poses.")

scale = {}
for name, k in (("cam_A", 0), ("cam_B", 1)):
    p = np.stack([q[k].reshape(GRID[1], GRID[0], 3) for q in pairs])
    edges = np.concatenate([np.linalg.norm(np.diff(p, axis=1), axis=-1).ravel(),
                            np.linalg.norm(np.diff(p, axis=2), axis=-1).ravel()])
    scale[name] = args.square / np.median(edges) if args.scale_fix else 1.0
    print(f"{name}: square size median {np.median(edges):.2f} mm vs {args.square} measured "
          f"({(np.median(edges) / args.square - 1) * 100:+.1f} %), depth {np.median(p[..., 2]):.0f} mm"
          + (f", scaled x{scale[name]:.3f}" if args.scale_fix else ""))
pairs = [(pa * scale["cam_A"], pb * scale["cam_B"]) for pa, pb in pairs]

# --- 2. corner order by consensus, then one fit over all agreeing views
cands = []  # per view: [(R, t, pb_ordered), (R, t, reversed)]
for pa, pb in pairs:
    cands.append([(*kabsch(o, pa), o) for o in (pb, pb[::-1])])
best, best_support = None, -1
for cs in cands:
    for R0, t0, _ in cs:
        support = sum(any(angle_deg(R @ R0.T) < 3 and np.linalg.norm(t - t0) < 30 for R, t, _ in c2) for c2 in cands)
        if support > best_support:
            best, best_support = (R0, t0), support
src, dst = [], []
for (pa, _), cs in zip(pairs, cands):
    R, t, o = min(cs, key=lambda c: angle_deg(c[0] @ best[0].T))
    if angle_deg(R @ best[0].T) < 3 and np.linalg.norm(t - best[1]) < 30:
        src.append(o)
        dst.append(pa)
for _ in range(3):  # refit, dropping views that disagree with the overall fit
    R, t = kabsch(np.concatenate(src), np.concatenate(dst))
    errs = [np.linalg.norm(s @ R.T + t - d, axis=1) for s, d in zip(src, dst)]
    med = np.array([np.median(e) for e in errs])
    keep = med < max(3 * np.median(med), 1.0)
    src = [s for s, k in zip(src, keep) if k]
    dst = [d for d, k in zip(dst, keep) if k]
R, t = kabsch(np.concatenate(src), np.concatenate(dst))
res = np.linalg.norm(np.concatenate(src) @ R.T + t - np.concatenate(dst), axis=1)
print(f"transform cam_B -> cam_A from {len(src)} of {len(pairs)} still views ({best_support} agreed on corner order): "
      f"rotation {angle_deg(R):.1f} deg, translation {np.linalg.norm(t):.0f} mm")
print(f"corner residual: median {np.median(res):.2f} mm, p95 {np.percentile(res, 95):.2f} mm, max {res.max():.2f} mm")

# --- 3. object clouds, merged
og = json.load(open(os.path.join(args.object, "groups.json")))["groups"]
mid = len(og) // 2
sel = og[max(0, mid - args.frames // 2):][:args.frames]
clouds = {}
for name in ("cam_A", "cam_B"):
    cam = Cam(args.object, name, args.size, args.slope, pad=args.max_disp)
    clouds[name] = sgbm_cloud(cam, [g["frames"][name] for g in sel], args.max_disp, args.zmin, args.zmax,
                              args.min_gray, args.min_hits, args.max_disp) * scale[name]
a, b = clouds["cam_A"], clouds["cam_B"] @ R.T + t
d, _ = cKDTree(a).query(b, distance_upper_bound=args.overlap)
ov = np.isfinite(d)
print(f"object: cam_A {len(a)} points, cam_B {len(b)} points, {ov.mean() * 100:.1f} % of cam_B within {args.overlap:g} mm of cam_A")
if ov.any():
    print(f"gap where they overlap: median {np.median(d[ov]):.2f} mm, p90 {np.percentile(d[ov], 90):.2f} mm")

out = os.path.join(args.object, "merge")
os.makedirs(out, exist_ok=True)
write_ply(os.path.join(out, "merged.ply"), np.vstack([a, b]),
          np.vstack([np.tile([255, 60, 60], (len(a), 1)), np.tile([60, 255, 60], (len(b), 1))]))
json.dump({"board_session": args.board, "R": R.tolist(), "t_mm": t.tolist(), "views": len(src), "scale": scale,
           "corner_residual_median_mm": float(np.median(res))}, open(os.path.join(out, "transform.json"), "w"), indent=2)
print("wrote", out)
