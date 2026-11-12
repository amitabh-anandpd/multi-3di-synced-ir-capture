"""PLAN.md P2: how far are the MANO labels (and LeapC's joints) from the hand-clicked joints?

  python tools/eval_clicks.py              # every clicked frame in labels/joints/

For each clicked frame (tools/click_joints.py) MANO's 21 joints (mano/smoothed.npz, cam_A's frame) are moved into the
clicked camera (for cam_B: back through fused/refine.json and the session's rig, exactly as fuse.py fused it) and
projected into its rectified left image. LeapC's joints of that camera are projected the same way, as a baseline.
Error per joint: pixels, and mm at the joint's depth (px * Z / f), i.e. the error across the view; the depth
direction is not seen by a click. Reported per joint type, per camera and per session; writes labels/joints/eval.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_mano import MANO_TO_LEAP, load_shape  # noqa: E402
from segment import LABELS, Segmenter  # noqa: E402
from stereo import rig_path  # noqa: E402

TYPES = {"wrist": [0], "knuckle": [1, 4, 7, 10, 13], "middle joint": [2, 5, 8, 11, 14],
         "end joint": [3, 6, 9, 12, 15], "tip": [16, 17, 18, 19, 20]}


def to_camera(P, session, cam):
    """(n, 3) points in cam_A's frame -> that camera's frame."""
    if cam == "cam_A":
        return P
    rig = json.load(open(rig_path(session)))
    Rb, tb = np.array(rig["cameras"]["cam_B"]["R"]), np.array(rig["cameras"]["cam_B"]["t_mm"])
    rf = os.path.join(session, "fused", "refine.json")
    R, t = np.eye(3), np.zeros(3)
    if os.path.exists(rf):
        r = json.load(open(rf))
        if r.get("used"):
            R, t = np.array(r["R"]), np.array(r["t_mm"])
    return ((P - t) @ R - tb) @ Rb  # inverse of P_A = R (Rb P_B + tb) + t


def project(P, unit):
    return np.stack([unit.f * P[:, 0] / P[:, 2] + unit.cx, unit.f * P[:, 1] / P[:, 2] + unit.cy], 1)


rows, segs = [], {}
for f in sorted(glob.glob(os.path.join(LABELS, "joints", "*", "*", "*.json"))):
    c = json.load(open(f))
    s = os.path.join("sessions", c["session"])
    uv = np.array([p if p is not None else [np.nan, np.nan] for p in c["uv"]], float)
    z = np.load(os.path.join(s, "mano", "smoothed.npz"))
    k = np.nonzero(z["frames"].astype(int) == c["cam_A_frame"])[0]
    if not len(k):
        continue
    key = (s, c["cam"])
    if key not in segs:
        segs[key] = Segmenter(s, c["cam"])
    seg = segs[key]
    M = to_camera(z["joints"][k[0]], s, c["cam"])
    em = np.linalg.norm(project(M, seg.unit) - uv, axis=1)
    out = {"session": c["session"], "cam": c["cam"], "frame": c["frame"], "mano_px": em,
           "mano_mm": em * M[:, 2] / seg.unit.f, "leap_px": np.full(21, np.nan), "leap_mm": np.full(21, np.nan)}
    lj = seg.joints(c["frame"], side=load_shape(s)["side"])
    if lj is not None:
        L = np.vstack([lj[0], lj[1]])[MANO_TO_LEAP]
        el = np.linalg.norm(project(L, seg.unit) - uv, axis=1)
        out["leap_px"], out["leap_mm"] = el, el * L[:, 2] / seg.unit.f
    rows.append(out)
if not rows:
    raise SystemExit("no clicked frames yet: python tools/click_joints.py")


def med(rs, key, idx):
    v = np.concatenate([r[key][idx] for r in rs])
    v = v[np.isfinite(v)]
    return (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if len(v) else (np.nan, np.nan, 0)


print(f"{len(rows)} clicked frames\n\n{'joint type':<14}{'MANO mm (med / p90)':>22}{'LeapC mm (med / p90)':>24}{'joints':>8}")
summary = {}
for name, idx in TYPES.items():
    m, l = med(rows, "mano_mm", idx), med(rows, "leap_mm", idx)
    summary[name] = {"mano_mm": m[:2], "leap_mm": l[:2], "n": m[2]}
    print(f"{name:<14}{m[0]:>11.1f} / {m[1]:<8.1f}{l[0]:>13.1f} / {l[1]:<8.1f}{m[2]:>8}")
allj = list(range(21))
m, l = med(rows, "mano_mm", allj), med(rows, "leap_mm", allj)
print(f"{'all':<14}{m[0]:>11.1f} / {m[1]:<8.1f}{l[0]:>13.1f} / {l[1]:<8.1f}{m[2]:>8}")
summary["all"] = {"mano_mm": m[:2], "leap_mm": l[:2], "n": m[2]}
print("\nper camera / session (all joints, MANO median mm):")
for key in sorted({(r["session"], r["cam"]) for r in rows}):
    rs = [r for r in rows if (r["session"], r["cam"]) == key]
    print(f"  {key[0]} {key[1]}: {med(rs, 'mano_mm', allj)[0]:.1f} mm  (LeapC {med(rs, 'leap_mm', allj)[0]:.1f})  "
          f"[{len(rs)} frames]")
json.dump({"summary": summary, "frames": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                                          for r in rows]}, open(os.path.join(LABELS, "joints", "eval.json"), "w"))
print(f"\nwrote {os.path.join(LABELS, 'joints', 'eval.json')}")
