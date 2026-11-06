"""N3: depth from a single frame per camera (the hand moves, so no multi-frame median).

  python depth.py sessions/2026-11-06_035726 sessions/2026-11-06_035752 sessions/2026-11-06_035809

Module: Depth(unit, cam).disparity(frame_id) -> disparity map (px) on the unit's rectified left
image (stereo.Unit), optionally only inside a row / column box (roi) to save time;
Depth.points(disp, mask) -> (n, 3) mm in the unit's frame.

CLI (N3 exit check on still-hand sessions): per camera, a rough hand mask comes from LeapC's joints
(tracking.csv) projected into the rectified image (thick bone lines + palm polygon), kept where
IR-bright and not far behind the joints; N4 replaces this with real segmentation. Each test frame is
compared with the median of the --ref frames around it (drift of a hand held in the air is not error). For --n single frames it reports the
share of hand pixels with depth (exit >= 70 %), the median |depth - median depth| (exit <= 2 mm)
and the time per frame, for each SGBM setting in SETTINGS.
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from session_io import read_pair
from stereo import Cam, Unit

SETTINGS = {  # name: (block, mode, uniqueness, speckle window)
    "b5_3way": (5, cv2.STEREO_SGBM_MODE_SGBM_3WAY, 10, 100),
    "b3_3way": (3, cv2.STEREO_SGBM_MODE_SGBM_3WAY, 10, 100),
    "b7_3way": (7, cv2.STEREO_SGBM_MODE_SGBM_3WAY, 10, 100),
    "b5_hh": (5, cv2.STEREO_SGBM_MODE_HH, 10, 100),
    "b5_u5": (5, cv2.STEREO_SGBM_MODE_SGBM_3WAY, 5, 100),
}


class Depth:
    def __init__(self, unit, cam, max_disp=128, setting="b5_u5"):  # N3: most hand coverage, 0.85 mm
        self.u, self.cam, self.max_disp = unit, cam, max_disp
        self.maps = unit.raw_maps(cam, max_disp)  # cam must be built with pad >= 0; output has max_disp pad columns
        block, mode, uniq, speck = SETTINGS[setting]
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=block,
                                          P1=8 * block ** 2, P2=32 * block ** 2, uniquenessRatio=uniq,
                                          speckleWindowSize=speck, speckleRange=2, mode=mode)

    def rectify(self, frame_id):
        raw = read_pair(self.cam.dir, self.cam.by_id[frame_id], self.cam.wh)
        return [cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0) for img, (mx, my) in zip(raw, self.maps)]

    def disparity(self, frame_id, roi=None, images=None):
        """Disparity (size x size, NaN where none) on the rectified left image, and that image.

        roi = (y0, y1, x0, x1) in unpadded rectified pixels: only that box is matched."""
        L, R = images if images is not None else self.rectify(frame_id)
        n = self.u.size
        out = np.full((n, n), np.nan, np.float32)
        if roi is None:
            y0, y1, x0, x1 = 0, n, 0, n
        else:
            y0, y1, x0, x1 = roi
        p = self.max_disp
        d = self.sgbm.compute(L[y0:y1, x0:x1 + p], R[y0:y1, x0:x1 + p]).astype(np.float32)[:, p:] / 16
        d[d <= 0] = np.nan
        out[y0:y1, x0:x1] = d
        return out, L[:, p:]

    def depth(self, disp):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.u.f * self.u.baseline / disp

    def points(self, disp, mask=None):
        z = self.depth(disp)
        ok = np.isfinite(z) if mask is None else np.isfinite(z) & mask
        v, u = np.nonzero(ok)
        z = z[ok]
        return np.stack([(u - self.u.cx) / self.u.f * z, (v - self.u.cy) / self.u.f * z, z], 1)


def palm_distance(cam_dir):
    p = os.path.join(cam_dir, "tracking.csv")
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    if not rows:
        return None
    xyz = np.array([[float(r[f"palm_{a}"]) for a in "xyz"] for r in rows])
    return float(np.median(np.linalg.norm(xyz, axis=1)))


def tracking_rows(cam_dir):
    """frame_id -> list of hand rows (dicts) from tracking.csv."""
    p = os.path.join(cam_dir, "tracking.csv")
    out = {}
    for r in (csv.DictReader(open(p)) if os.path.exists(p) else []):
        out.setdefault(int(r["frame_id"]), []).append(r)
    return out


def hand_joints(row):
    """(25 joints, palm, wrist) of one tracking.csv row, LeapC tracking space (mm)."""
    J = np.array([[float(row[f"j{i:02d}_{a}"]) for a in "xyz"] for i in range(25)])
    return J, np.array([float(row[f"palm_{a}"]) for a in "xyz"]), np.array([float(row[f"wrist_{a}"]) for a in "xyz"])


def leap_to_unit(P, unit, leap_baseline):
    """LeapC tracking space -> the unit's rectified-left frame. Axes found from the data (2026-11-06):
    (x, y, z)_unit = (-x, z, y)_leap, origin at the device centre = +baseline/2 from the left imager."""
    P = np.asarray(P, float).reshape(-1, 3)
    Q = np.stack([-P[:, 0] + leap_baseline / 2, P[:, 2], P[:, 1]], 1)
    return Q @ unit.R1.T


def hand_mask(unit, row, leap_baseline, size, finger_mm=20, thumb_mm=24):
    """Rough hand mask from LeapC joints: thick lines along the bones plus the palm polygon; and the joints' median z."""
    J, palm, wrist = hand_joints(row)
    P = leap_to_unit(np.vstack([J, wrist]), unit, leap_baseline)
    z = P[:, 2]
    px = np.stack([unit.f * P[:, 0] / z + unit.cx, unit.f * P[:, 1] / z + unit.cy], 1)
    m = np.zeros((size, size), np.uint8)
    ip = lambda q: (int(round(q[0])), int(round(q[1])))  # noqa: E731
    for f in range(5):
        w = thumb_mm if f == 0 else finger_mm
        for b in range(4):
            a, c = 5 * f + b, 5 * f + b + 1
            t = max(2, int(unit.f * w / ((z[a] + z[c]) / 2)))
            cv2.line(m, ip(px[a]), ip(px[c]), 1, t)
    palm_pts = px[[25, 0, 1, 5, 6, 10, 11, 15, 16, 20, 21]]  # wrist, finger bases and knuckles
    cv2.fillConvexPoly(m, cv2.convexHull(palm_pts.astype(np.int32)), 1)
    return m > 0, float(np.median(z))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="still-hand sessions")
    ap.add_argument("--ref", type=int, default=10, help="neighbouring frames in each frame's median reference")
    ap.add_argument("--n", type=int, default=20, help="single frames tested per camera")
    ap.add_argument("--band", type=float, default=80, help="mm around the joints' median depth kept as hand")
    ap.add_argument("--min-gray", type=int, default=50)
    ap.add_argument("--max-disp", type=int, default=128)
    ap.add_argument("--settings", nargs="+", default=["b5_3way", "b7_3way", "b5_u5"])
    args = ap.parse_args()

    results = {k: [] for k in args.settings}
    for s in args.sessions:
        meta = json.load(open(os.path.join(s, "session.json")))
        groups = json.load(open(os.path.join(s, "groups.json")))["groups"]
        for name in sorted(meta["cameras"]):
            cam = Cam(s, name, 800, 1.5, pad=256)
            unit = Unit(meta["cameras"][name])
            track = tracking_rows(cam.dir)
            if not track:
                print(f"{s[-6:]} {name}: no LeapC tracking, skipped")
                continue
            leap_b = cam.baseline
            ids = [g["frames"][name] for g in groups]
            half = args.ref // 2
            test_pos = list(range(half, len(ids) - half, max(1, (len(ids) - 2 * half) // args.n)))[:args.n]
            for k in args.settings:
                dep = Depth(unit, cam, args.max_disp, k)
                for pos in test_pos:
                    i = ids[pos]
                    tid = min(track, key=lambda t: abs(t - i))
                    drawn, zj = hand_mask(unit, track[tid][0], leap_b, unit.size)
                    ys, xs = np.nonzero(drawn)
                    if len(ys) == 0:
                        continue
                    roi = (max(ys.min() - 16, 0), min(ys.max() + 17, 800), max(xs.min() - 16, 0), min(xs.max() + 17, 800))
                    # reference: median of the --ref frames around this one (about +-55 ms), not including it,
                    # so slow drift of a freely held hand doesn't count as error
                    near = [ids[j] for j in range(pos - half, pos + half + 1) if j != pos]
                    ref = np.nanmedian(np.stack([dep.disparity(j, roi=roi)[0] for j in near]), axis=0)
                    zr = dep.depth(ref)
                    imgs = dep.rectify(i)
                    t0 = time.perf_counter()
                    d, L = dep.disparity(i, images=imgs)
                    t_full = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    dr, _ = dep.disparity(i, roi=roi, images=imgs)
                    t_roi = time.perf_counter() - t0
                    # bright, and either no reference depth or reference depth near the joints (not the body behind)
                    hand = drawn & (L >= args.min_gray) & (~np.isfinite(zr) | (np.abs(zr - zj) < args.band))
                    z = dep.depth(dr)
                    cov = np.isfinite(z[hand]).mean()
                    both = hand & np.isfinite(z) & np.isfinite(zr)
                    err = np.median(np.abs(z[both] - zr[both])) if both.any() else np.nan
                    roi_same = np.nanmedian(np.abs(z[both] - dep.depth(d)[both])) if both.any() else np.nan
                    results[k].append((s[-6:], name, hand.sum(), zj, cov, err, t_full, t_roi, roi_same))
    print(f"{'setting':<9} {'session':<7} {'cam':<6} {'hand px':>7} {'palm mm':>7} {'depth %':>8} {'|dz| mm':>8} "
          f"{'full ms':>8} {'roi ms':>7}")
    for k, rows in results.items():
        arr = {}
        for r in rows:
            arr.setdefault((r[0], r[1]), []).append(r)
        for (sname, cname), rs in arr.items():
            rs = np.array([x[2:] for x in rs], float)
            print(f"{k:<9} {sname:<7} {cname:<6} {rs[0, 0]:>7.0f} {rs[0, 1]:>7.0f} {np.median(rs[:, 2]) * 100:>7.1f}% "
                  f"{np.nanmedian(rs[:, 3]):>8.2f} {np.median(rs[:, 4]) * 1e3:>8.1f} {np.median(rs[:, 5]) * 1e3:>7.1f}")
        allr = np.array([x[2:] for x in rows], float)
        print(f"{k:<9} ALL: depth on {np.median(allr[:, 2]) * 100:.1f} % of hand pixels (exit >= 70 %), "
              f"|dz| median {np.nanmedian(allr[:, 3]):.2f} mm (exit <= 2 mm), {np.median(allr[:, 4]) * 1e3:.0f} ms full / "
              f"{np.median(allr[:, 5]) * 1e3:.0f} ms hand box per frame per camera; roi vs full |dz| {np.nanmedian(allr[:, 6]):.2f} mm\n")
