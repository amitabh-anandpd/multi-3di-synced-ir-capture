"""N6: fused hand cloud per frame from both cameras.

  python fuse.py sessions/2026-11-06_035942

Per frame group: each camera's single-frame hand points (segment.py), cam_B's moved into cam_A's
frame with rig.json and a small per-session refinement, stray points removed.

Refinement: rig.json comes from a board in the middle of the views; the hand clouds still sit ~4 mm
apart (STATUS.md, N5). Point-to-plane ICP over the hand clouds of every --refine-every-th frame gives one
small correction (capped at --max-mm / --max-deg) for the whole session, fitted on half of those frames
and checked on the other half before it is used.

Check: cam_B's point-to-plane distance to cam_A's surface where both see the same surface (normals
within 30 deg); nearest-point distance mixes up different sides of the hand at 90 deg.
Writes <session>/fused/: hands.npz (points, camera of each point, frame offsets), every --ply-every-th
frame as <frame>.ply (cam_A red, cam_B green), playback.mp4 (front and side views), refine.json.
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

from segment import Segmenter
from stereo import angle_deg, rig_path, write_ply


def normals(P, k=12):
    """Unit surface normals of points P from their k nearest neighbours (PCA)."""
    _, idx = cKDTree(P).query(P, k=min(k, len(P)))
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]


def gap(a, b, overlap=15):
    d, _ = cKDTree(a).query(b, distance_upper_bound=overlap)
    ov = np.isfinite(d)
    return (float(np.median(d[ov])) if ov.any() else np.nan), float(ov.mean())


def surface_gap(a, b, overlap=15, max_deg=30):
    """Point-to-plane distance from cam_B points to cam_A's surface, only where both see the same surface
    (normals within max_deg). At 90 deg the cameras mostly see different sides of the hand, and nearest-point
    distance then pairs points a finger's thickness apart. Returns (median mm, share of pairs that qualify)."""
    na, nb = normals(a), normals(b)
    d, j = cKDTree(a).query(b, distance_upper_bound=overlap)
    ok = np.isfinite(d)
    same = np.abs((nb[ok] * na[j[ok]]).sum(1)) > np.cos(np.radians(max_deg))
    if not same.any():
        return np.nan, 0.0
    q = j[ok][same]
    return float(np.median(np.abs(((b[ok][same] - a[q]) * na[q]).sum(1)))), float(same.mean())


def clean(P, k=8, factor=2.5):
    """Drop points whose k-th neighbour is far (isolated specks, edge streaks)."""
    if len(P) < k + 1:
        return P
    d, _ = cKDTree(P).query(P, k=k + 1)
    d = d[:, -1]
    return P[d < factor * np.median(d)]


def icp_point_to_plane(pairs, iters=15, max_dist=10):
    """One rigid correction (R, t) for B -> A over many frames: pairs = [(A points, A normals, B points)]."""
    R, t = np.eye(3), np.zeros(3)
    for _ in range(iters):
        rows, rhs = [], []
        for A, NA, B in pairs:
            Bt = B @ R.T + t
            d, j = cKDTree(A).query(Bt, distance_upper_bound=max_dist)
            ok = np.isfinite(d)
            if ok.sum() < 50:
                continue
            p, q, n = Bt[ok], A[j[ok]], NA[j[ok]]
            r = ((p - q) * n).sum(1)
            keep = np.abs(r) < np.percentile(np.abs(r), 80)  # trim the worst 20 %
            rows.append(np.hstack([np.cross(p[keep], n[keep]), n[keep]]))
            rhs.append(-r[keep])
        Aa, bb = np.vstack(rows), np.concatenate(rhs)
        x = np.linalg.lstsq(Aa, bb, rcond=None)[0]
        dR = cv2.Rodrigues(x[:3])[0]
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x[3:]) < 1e-3 and np.linalg.norm(x[:3]) < 1e-5:
            break
    return R, t


def render(points, cams, size=480, center=None, scale=None):
    """Front (cam_A view, x-y) and side (z-y) dot images; cam_A red, cam_B green."""
    out = []
    for ax in ((0, 1), (2, 1)):
        img = np.zeros((size, size, 3), np.uint8)
        if len(points):
            uv = (points[:, ax] - center[list(ax)]) * scale + size / 2
            ok = (uv >= 0).all(1) & (uv < size).all(1)
            for c, col in ((0, (60, 60, 255)), (1, (60, 255, 60))):
                m = ok & (cams == c)
                img[uv[m, 1].astype(int), uv[m, 0].astype(int)] = col
        out.append(img)
    v = np.hstack(out)
    cv2.putText(v, "front (cam_A view)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(v, "side", (size + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return v


ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--every", type=int, default=1, help="fuse every Nth frame group")
ap.add_argument("--refine-every", type=int, default=15, help="frame groups used for the refinement")
ap.add_argument("--max-mm", type=float, default=8, help="largest refinement translation accepted")
ap.add_argument("--max-deg", type=float, default=3, help="largest refinement rotation accepted")
ap.add_argument("--no-refine", action="store_true")
ap.add_argument("--ply-every", type=int, default=30)
ap.add_argument("--max-frames", type=int, default=0, help="stop after this many groups (0 = all)")
ap.add_argument("--out", help="output folder (default <session>/fused); tools/rig_check.py uses a scratch one")
ap.add_argument("--max-disp", type=int, default=128,
                help="SGBM range in px: 128 measures down to ~144 mm from a camera, 192 to ~96 mm (hand held close)")
args = ap.parse_args()

rig = json.load(open(rig_path(args.session)))  # the session's pinned rig, else the current one
Rb, tb = np.array(rig["cameras"]["cam_B"]["R"]), np.array(rig["cameras"]["cam_B"]["t_mm"])
groups = json.load(open(os.path.join(args.session, "groups.json")))["groups"]
sa, sb = (Segmenter(args.session, c, max_disp=args.max_disp) for c in ("cam_A", "cam_B"))
out = args.out or os.path.join(args.session, "fused")
os.makedirs(out, exist_ok=True)
cache = {}


def frame_points(g):
    k = (g["frames"]["cam_A"], g["frames"]["cam_B"])
    if k not in cache:
        a, b = sa.points(k[0]), sb.points(k[1])
        cache[k] = (clean(a) if len(a) else a, clean(b @ Rb.T + tb) if len(b) else b)
    return cache[k]


# --- per-session refinement, fitted on half the sampled frames, checked on the other half
R, t, refine = np.eye(3), np.zeros(3), {"used": False}
if not args.no_refine:
    sample = []
    for g in groups[::args.refine_every]:
        a, b = frame_points(g)
        if len(a) > 300 and len(b) > 300:
            sample.append((a, normals(a), b))
    fit, test = sample[0::2], sample[1::2]
    if len(fit) >= 5:
        Rf, tf = icp_point_to_plane([(A, N, B) for A, N, B in fit])
        before = [gap(A, B)[0] for A, _, B in test]
        after = [gap(A, B @ Rf.T + tf)[0] for A, _, B in test]
        deg, mm = angle_deg(Rf), float(np.linalg.norm(tf))
        ok = deg <= args.max_deg and mm <= args.max_mm and np.nanmedian(after) < np.nanmedian(before)
        refine = {"used": bool(ok), "rotation_deg": deg, "translation_mm": mm, "fit_frames": len(fit),
                  "test_frames": len(test), "test_gap_before_mm": float(np.nanmedian(before)),
                  "test_gap_after_mm": float(np.nanmedian(after)), "R": Rf.tolist(), "t_mm": tf.tolist()}
        print(f"refinement: {deg:.2f} deg, {mm:.2f} mm from {len(fit)} frames; on {len(test)} other frames the gap "
              f"goes {np.nanmedian(before):.2f} -> {np.nanmedian(after):.2f} mm -> {'used' if ok else 'NOT used'}")
        if ok:
            R, t = Rf, tf
    else:
        print("refinement: too few frames with both hands, skipped")
json.dump(refine, open(os.path.join(out, "refine.json"), "w"), indent=2)

# --- fuse every frame
pts, cams, offs, fids, gaps = [], [], [0], [], []
sel = groups[::args.every][: args.max_frames or None]
for i, g in enumerate(sel):
    a, b = frame_points(g)
    if len(b):
        b = b @ R.T + t
    P = np.vstack([a, b]) if len(a) + len(b) else np.zeros((0, 3))
    C = np.r_[np.zeros(len(a), np.uint8), np.ones(len(b), np.uint8)]
    pts.append(P.astype(np.float32))
    cams.append(C)
    offs.append(offs[-1] + len(P))
    fids.append((g["frames"]["cam_A"], g["frames"]["cam_B"]))
    if len(a) > 200 and len(b) > 200 and i % 5 == 0:
        gaps.append(gap(a, b) + surface_gap(a, b))
    if args.ply_every and i % args.ply_every == 0 and len(P):
        write_ply(os.path.join(out, f"{g['frames']['cam_A']}.ply"), P,
                  np.where(C[:, None] == 0, [255, 60, 60], [60, 255, 60]))
    if i % 100 == 0:
        print(f"  {i}/{len(sel)} frames", flush=True)
cache.clear()
allp = np.vstack(pts) if pts else np.zeros((0, 3))
np.savez_compressed(os.path.join(out, "hands.npz"), points=allp, camera=np.concatenate(cams), offsets=np.array(offs),
                    frames=np.array(fids))

# --- playback video (front + side), fixed framing from all points
if len(allp):
    center = np.median(allp, 0)
    scale = 480 * 0.8 / max(np.percentile(np.abs(allp - center), 98, axis=0).max() * 2, 1)
    vw = cv2.VideoWriter(os.path.join(out, "playback.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 90 / args.every, (960, 480))
    for P, C, (fa, _) in zip(pts, cams, fids):
        v = render(P, C, 480, center, scale)
        cv2.putText(v, f"frame {fa}", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        vw.write(v)
    vw.release()

g = np.array(gaps)
print(f"\n{len(sel)} frames fused, {sum(len(p) for p in pts) / max(len(pts), 1):.0f} points per frame")
if len(g):
    print(f"cam_B vs cam_A on {len(g)} sampled frames: same-surface point-to-plane gap median {np.nanmedian(g[:, 2]):.2f} mm "
          f"(N6 exit <= 3 mm; {np.median(g[:, 3]) * 100:.0f} % of near pairs are the same surface), nearest-point gap "
          f"{np.nanmedian(g[:, 0]):.2f} mm, overlap {np.median(g[:, 1]) * 100:.0f} %")
print("wrote", out)
