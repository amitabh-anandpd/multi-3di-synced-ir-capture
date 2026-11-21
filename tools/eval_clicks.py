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
from clicks3d import CHAIN, clicked_pairs, finger_length, palm_width, to_camera, triangulate  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--person", help="only sessions of this person in dataset.json (e.g. P02)")
ap.add_argument("--hand", choices=["left", "right"], help="only sessions of this hand")
ap.add_argument("--out", help="write the per-frame results of a filtered run here (eval.json itself stays untouched)")
args = ap.parse_args()
FILTERED = bool(args.person or args.hand)
_rows = json.load(open("dataset.json"))["sessions"] if FILTERED else {}


def wanted(session):
    """Whether a clicked session passes --person / --hand (always, without them)."""
    if not FILTERED:
        return True
    r = _rows.get(session, {})
    return (not args.person or r.get("person") == args.person) and (not args.hand or r.get("hand") == args.hand)


TYPES = {"wrist": [0], "knuckle": [1, 4, 7, 10, 13], "middle joint": [2, 5, 8, 11, 14],
         "end joint": [3, 6, 9, 12, 15], "tip": [16, 17, 18, 19, 20]}


def project(P, unit):
    return np.stack([unit.f * P[:, 0] / P[:, 2] + unit.cx, unit.f * P[:, 1] / P[:, 2] + unit.cy], 1)


rows, segs = [], {}
for f in sorted(glob.glob(os.path.join(LABELS, "joints", "*", "*", "*.json"))):
    c = json.load(open(f))
    if not wanted(c["session"]):
        continue
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
if FILTERED:  # a subset must not overwrite the full-dataset results other tools read
    if args.out:
        json.dump({"summary": summary, "frames": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                   for k, v in r.items()} for r in rows]}, open(args.out, "w"))
        print(f"\nwrote {args.out}")
    print("\n(filtered run: eval.json / eval_3d.json left as they were)")
else:
    json.dump({"summary": summary, "frames": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                               for k, v in r.items()} for r in rows]},
              open(os.path.join(LABELS, "joints", "eval.json"), "w"))
    print(f"\nwrote {os.path.join(LABELS, 'joints', 'eval.json')}")

# --- 3D: moments clicked in both cameras -> triangulated joints -> 3D error and finger lengths
tri = []
for (sess, fa), cc in sorted(clicked_pairs(wanted).items()):
    if len(cc) < 2:
        continue
    s = os.path.join("sessions", sess)
    z = np.load(os.path.join(s, "mano", "smoothed.npz"))
    k = np.nonzero(z["frames"].astype(int) == fa)[0]
    if not len(k):
        continue
    X = triangulate(s, cc["cam_A"], cc["cam_B"], segs[(s, "cam_A")].unit, segs[(s, "cam_B")].unit)
    tri.append({"session": sess, "frame": fa, "clicked": X, "mano": z["joints"][k[0]]})
if tri:
    print(f"\n3D: {len(tri)} moments clicked in both cameras (rays meeting within 15 mm)")
    e3 = np.array([np.linalg.norm(t["clicked"] - t["mano"], axis=1) for t in tri])
    for name, idx in TYPES.items():
        v = e3[:, idx].ravel()
        v = v[np.isfinite(v)]
        if len(v):
            print(f"  {name:<14} MANO vs triangulated clicks: median {np.median(v):5.1f} mm, p90 {np.percentile(v, 90):5.1f}  [{len(v)}]")
    print("  finger length, knuckle (thumb: base) to tip, median over moments where all 4 joints were clicked:")
    lengths = {}
    for f_ in CHAIN:
        lx, lm = [], []
        for t in tri:
            x = finger_length(t["clicked"], f_)
            if x is not None:
                lx.append(x)
                lm.append(finger_length(t["mano"], f_))
        if lx:
            d = np.array(lx) - np.array(lm)
            lengths[f_] = {"clicked_mm": float(np.median(lx)), "mano_mm": float(np.median(lm)),
                           "diff_mm": float(np.median(d)), "moments": len(lx)}
            print(f"    {f_:<7} clicked {np.median(lx):5.1f}  MANO {np.median(lm):5.1f}  difference {np.median(d):+5.1f} mm "
                  f"(p25 {np.percentile(d, 25):+.1f}, p75 {np.percentile(d, 75):+.1f})  [{len(lx)}]")
    # palm width as fit_mano.py --palm-mm defines it: index MCP (1) to little MCP (7)
    pw = [(palm_width(t["clicked"]), palm_width(t["mano"])) for t in tri if palm_width(t["clicked"]) is not None]
    if pw:
        px, pm = np.array(pw).T
        lengths["palm"] = {"clicked_mm": float(np.median(px)), "mano_mm": float(np.median(pm)),
                           "diff_mm": float(np.median(px - pm)), "moments": len(pw)}
        print(f"  palm width, index to little knuckle (fit_mano.py --palm-mm): clicked {np.median(px):5.1f}  "
              f"MANO {np.median(pm):5.1f}  difference {np.median(px - pm):+5.1f} mm  [{len(pw)}]")
    if not FILTERED:
        json.dump({"moments": len(tri), "finger_lengths": lengths},
                  open(os.path.join(LABELS, "joints", "eval_3d.json"), "w"), indent=1)
