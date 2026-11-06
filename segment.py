"""N4: hand mask per frame per camera.

  python segment.py --eval            IoU against the hand-painted masks in labels/ (tools/label_hands.py)

Steps for one camera frame (Segmenter.mask):
 1. LeapC's joints for this frame (cam_X/tracking.csv), mapped into the unit's rectified frame
    (depth.leap_to_unit). If this camera has no hand, the other camera's joints are moved over with
    rig.json. No joints -> empty mask.
 2. Disparity only in the box around the projected joints (depth.Depth, ~6 ms).
 3. Keep pixels that are IR-bright and whose 3D point lies within --radius mm of a bone (finger
    bones, palm triangle fan). Pixels without depth are kept if bright and inside the drawn joint mask.
 4. Cut the arm: drop points on the far side of the plane through the wrist, normal along elbow -> wrist.
 5. Close small gaps, keep the connected pieces that touch the joints, fill holes inside the hand.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from depth import Depth, hand_joints, hand_mask, leap_to_unit, tracking_rows
from stereo import Cam, Unit

LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels")


def seg_point_dist(P, A, B):
    """Distance from points P (n, 3) to segments A[k]-B[k] (m, 3): (n, m)."""
    AB = B - A
    t = np.clip(((P[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9)[None], 0, 1)
    return np.linalg.norm(P[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1)


def bones(J, wrist):
    """Segments (A, B) for the hand: finger bones and a fan over the palm from the wrist."""
    A, B = [], []
    for f in range(5):
        for b in range(4):
            A.append(J[5 * f + b])
            B.append(J[5 * f + b + 1])
        A.append(wrist)
        B.append(J[5 * f + 1])  # wrist to each knuckle covers the palm
    for f in range(1, 4):
        A.append(J[5 * f + 1])
        B.append(J[5 * (f + 1) + 1])  # across the knuckles
    return np.array(A), np.array(B)


class Segmenter:
    def __init__(self, session, name, radius=40, min_gray=45, margin=24, fill=True):
        self.session, self.name = session, name
        meta = json.load(open(os.path.join(session, "session.json")))
        self.cam = Cam(session, name, 800, 1.5, pad=256)
        self.unit = Unit(meta["cameras"][name])
        self.depth = Depth(self.unit, self.cam)
        self.track = tracking_rows(self.cam.dir)
        self.radius, self.min_gray, self.margin, self.fill = radius, min_gray, margin, fill
        # the other camera, for frames this one has no hand in
        self.other = None
        rig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rig.json")
        others = [n for n in meta["cameras"] if n != name]
        if others and os.path.exists(rig_path):
            rig = json.load(open(rig_path))
            o = others[0]
            if o in rig["cameras"] and name in rig["cameras"]:
                ocam = Cam(session, o, 800, 1.5)
                # X_name = R_name^T (R_o X_o + t_o - t_name)
                Ro, to = np.array(rig["cameras"][o]["R"]), np.array(rig["cameras"][o]["t_mm"])
                Rn, tn = np.array(rig["cameras"][name]["R"]), np.array(rig["cameras"][name]["t_mm"])
                groups = json.load(open(os.path.join(session, "groups.json")))["groups"]
                self.other = {"track": tracking_rows(ocam.dir), "unit": Unit(meta["cameras"][o]), "b": ocam.baseline,
                              "R": Rn.T @ Ro, "t": Rn.T @ (to - tn),
                              "frame": {g["frames"][name]: g["frames"][o] for g in groups}}

    def joints(self, frame_id):
        """(J (25, 3), wrist, elbow) in this unit's frame, or None. Source: own tracking, else the other camera."""
        r = self.track.get(frame_id)
        if r:
            J, _, wrist = hand_joints(r[0])
            el = np.array([float(r[0][f"elbow_{a}"]) for a in "xyz"])
            P = leap_to_unit(np.vstack([J, wrist, el]), self.unit, self.cam.baseline)
            return P[:25], P[25], P[26]
        o = self.other
        if o and frame_id in o["frame"] and o["frame"][frame_id] in o["track"]:
            r = o["track"][o["frame"][frame_id]][0]
            J, _, wrist = hand_joints(r)
            el = np.array([float(r[f"elbow_{a}"]) for a in "xyz"])
            P = leap_to_unit(np.vstack([J, wrist, el]), o["unit"], o["b"]) @ o["R"].T + o["t"]
            return P[:25], P[25], P[26]
        return None

    def project(self, P):
        u = self.unit
        return np.stack([u.f * P[:, 0] / P[:, 2] + u.cx, u.f * P[:, 1] / P[:, 2] + u.cy], 1)

    def mask(self, frame_id):
        """(hand mask (800, 800) bool, rectified left image) for one frame."""
        n = self.unit.size
        imgs = self.depth.rectify(frame_id)
        L = imgs[0][:, self.depth.max_disp:]
        jt = self.joints(frame_id)
        if jt is None:
            return np.zeros((n, n), bool), L
        J, wrist, elbow = jt
        drawn = drawn_mask(self.unit, J, wrist, n)
        ys, xs = np.nonzero(drawn)
        if len(ys) == 0:
            return np.zeros((n, n), bool), L
        m = self.margin
        roi = (max(ys.min() - m, 0), min(ys.max() + m + 1, n), max(xs.min() - m, 0), min(xs.max() + m + 1, n))
        disp, _ = self.depth.disparity(frame_id, roi=roi, images=imgs)
        y0, y1, x0, x1 = roi
        box = np.zeros((n, n), bool)
        box[y0:y1, x0:x1] = True
        bright = (L >= self.min_gray) & box
        z = self.depth.depth(disp)
        has = np.isfinite(z) & bright
        v, u = np.nonzero(has)
        zz = z[has]
        P = np.stack([(u - self.unit.cx) / self.unit.f * zz, (v - self.unit.cy) / self.unit.f * zz, zz], 1)
        A, B = bones(J, wrist)
        near = seg_point_dist(P, A, B).min(1) < self.radius
        arm_dir = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        near &= (P - wrist) @ arm_dir > -10  # keep up to 1 cm past the wrist towards the elbow
        keep = np.zeros((n, n), bool)
        keep[v[near], u[near]] = True
        # no depth: bright and inside the drawn joint mask (thin fingers often lack depth), not past the wrist
        wp = self.project(np.array([wrist, wrist + 50 * arm_dir]))
        d2 = wp[1] - wp[0]
        vv, uu = np.indices((n, n))
        past = ((uu - wp[0, 0]) * d2[0] + (vv - wp[0, 1]) * d2[1]) < 0
        keep |= bright & ~np.isfinite(z) & drawn & ~past
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        keep &= bright
        k, lab = cv2.connectedComponents(keep.astype(np.uint8))
        touch = np.unique(lab[drawn & keep])
        out = np.isin(lab, touch[touch > 0])
        if self.fill:  # holes inside the hand (dim or depth-less spots on the palm)
            ff = out.astype(np.uint8) * 255
            fl = np.zeros((n + 2, n + 2), np.uint8)
            cv2.floodFill(ff, fl, (0, 0), 128)
            out |= ff == 0
        return out, L


def drawn_mask(unit, J, wrist, size, finger_mm=20, thumb_mm=24):
    """depth.hand_mask on joints already in the unit's frame."""
    P = np.vstack([J, wrist])
    z = P[:, 2]
    px = np.stack([unit.f * P[:, 0] / z + unit.cx, unit.f * P[:, 1] / z + unit.cy], 1)
    m = np.zeros((size, size), np.uint8)
    ip = lambda q: (int(round(q[0])), int(round(q[1])))  # noqa: E731
    for f in range(5):
        w = thumb_mm if f == 0 else finger_mm
        for b in range(4):
            a, c = 5 * f + b, 5 * f + b + 1
            cv2.line(m, ip(px[a]), ip(px[c]), 1, max(2, int(unit.f * w / ((z[a] + z[c]) / 2))))
    cv2.fillConvexPoly(m, cv2.convexHull(px[[25, 0, 1, 5, 6, 10, 11, 15, 16, 20, 21]].astype(np.int32)), 1)
    return m > 0


def label_files():
    return sorted(glob.glob(os.path.join(LABELS, "*", "*", "*.png")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="IoU against labels/<session>/<cam>/<frame>.png")
    ap.add_argument("--radius", type=float, default=40)
    ap.add_argument("--min-gray", type=int, default=45)
    ap.add_argument("--exclude", nargs="*", default=[], help="session_cam pairs left out of the summary, e.g. "
                    "2026-11-06_035647_cam_A (side-on views with an unclear wrist)")
    ap.add_argument("--show", help="folder to write overlays (green = label only, red = mask only, yellow = both)")
    args = ap.parse_args()
    if not args.eval:
        raise SystemExit("nothing to do (use --eval)")
    segs, res = {}, []
    for f in label_files():
        sess, cam, fid = f.split(os.sep)[-3], f.split(os.sep)[-2], int(os.path.splitext(os.path.basename(f))[0])
        key = (sess, cam)
        if key not in segs:
            segs[key] = Segmenter(os.path.join("sessions", sess), cam, args.radius, args.min_gray)
        m, L = segs[key].mask(fid)
        lab = cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127
        inter, union = (m & lab).sum(), (m | lab).sum()
        iou = inter / union if union else 1.0
        # same, ignoring a 2 px band along the painted edge (brush edges on blurry IR outlines)
        band = cv2.morphologyEx(lab.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        u2 = ((m | lab) & ~band).sum()
        iou2 = ((m & lab) & ~band).sum() / u2 if u2 else 1.0
        res.append((sess, cam, fid, iou, lab.sum(), m.sum(), iou2))
        if args.show:
            os.makedirs(args.show, exist_ok=True)
            v = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
            v[lab & ~m] = (0, 200, 0)
            v[m & ~lab] = (0, 0, 220)
            v[m & lab] = (0, 200, 220)
            cv2.imwrite(os.path.join(args.show, f"{sess}_{cam}_{fid}.png"), v)
    if args.exclude:
        res = [r for r in res if f"{r[0]}_{r[1]}" not in args.exclude]
        print(f"left out: {args.exclude}")
    for cam in sorted({r[1] for r in res}):
        ious = np.array([r[3] for r in res if r[1] == cam])
        i2 = np.array([r[6] for r in res if r[1] == cam])
        print(f"{cam}: {len(ious)} frames, IoU median {np.median(ious):.3f}, min {ious.min():.3f}, "
              f"frames >= 0.9: {(ious >= 0.9).mean() * 100:.0f} %  (N4 exit: IoU >= 0.9) | ignoring 2 px at the "
              f"painted edge: median {np.median(i2):.3f}, min {i2.min():.3f}, >= 0.9: {(i2 >= 0.9).mean() * 100:.0f} %")
    for r in sorted(res, key=lambda r: r[3])[:5]:
        print(f"  worst: {r[0]} {r[1]} frame {r[2]}: IoU {r[3]:.3f} (label {r[4]} px, mask {r[5]} px)")
