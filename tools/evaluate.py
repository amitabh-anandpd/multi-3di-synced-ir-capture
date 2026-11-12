"""N8: evaluate the MANO fits per pose and per speed, against the fused points and LeapC's tracking.

  python tools/evaluate.py sessions/2026-11-06_035647 sessions/2026-11-06_035726 ...

Uses <session>/mano/smoothed.npz (fit_mano.py --post) and <session>/fused/hands.npz. Per frame:
  - point-to-mesh: fused points to a dense sampling of the MANO surface, median (mm)
  - MANO vs LeapC joints (cam_A's tracking in cam_A's frame, cam_B's via rig.json when cam_A has no hand),
    per joint type; also after removing the session's mean offset (LeapC's joints sit inside the bones,
    MANO's are its own regressed joints, so part of the difference is definition, not error)
  - hand speed: 95th-percentile MANO joint speed (mm/s)
  - jitter: joint acceleration, mm / frame^2 (frame = the fit's frame step)
Prints per session, per speed band and the worst frames; writes evaluation.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_mano import MANO_TO_LEAP, Hand, load_fused, load_shape, surface_samples  # noqa: E402
from segment import Segmenter  # noqa: E402
from session_io import load_rows  # noqa: E402

TYPES = {"wrist": [0], "knuckle (MCP; thumb CMC)": [1, 4, 7, 10, 13], "middle joint (PIP; thumb MCP)": [2, 5, 8, 11, 14],
         "end joint (DIP; thumb IP)": [3, 6, 9, 12, 15], "tip": [16, 17, 18, 19, 20]}
BANDS = [(0, 50), (50, 150), (150, 400), (400, 1e9)]

ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="+")
ap.add_argument("--every", type=int, default=2, help="evaluate every Nth fitted frame")
ap.add_argument("--out", default="evaluation.json")
args = ap.parse_args()

T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
rows, summary, hands = [], {}, {}
for s in args.sessions:
    sh = load_shape(s)  # this session's person and hand
    hand = hands.setdefault(sh["side"], Hand(sh["side"]))
    betas = torch.tensor([sh["betas"]], dtype=torch.float32)
    note = json.load(open(os.path.join(s, "session.json")))["notes"].replace("N2 ", "")
    z = np.load(os.path.join(s, "mano", "smoothed.npz"))
    fr, J = z["frames"], z["joints"]
    fused = dict(load_fused(s))
    ts = {int(r["frame_id"]): int(r["device_ts_us"]) for r in load_rows(os.path.join(s, "cam_A"))}
    seg = Segmenter(s, "cam_A")
    t = np.array([ts[int(f)] for f in fr]) / 1e6
    vel = np.linalg.norm(np.gradient(J, t, axis=0), axis=-1)  # (n, 21) mm/s
    speed = np.percentile(vel, 95, axis=1)
    acc = np.r_[np.nan, np.median(np.linalg.norm(J[2:] - 2 * J[1:-1] + J[:-2], axis=-1), axis=1), np.nan]
    with torch.no_grad():
        V, _ = hand(betas.expand(len(fr), -1), T(z["orient"]), T(z["pose"]), T(z["transl"]))
    V = V.numpy()
    per = []
    for i in range(0, len(fr), args.every):
        f = int(fr[i])
        p2m = float(np.median(cKDTree(surface_samples(V[i], hand.faces)).query(fused[f])[0]))
        lj = seg.joints(f, side=sh["side"])  # mirrored (other-side) tracking is not a fair reference
        diff = None
        if lj is not None:
            L21 = np.vstack([lj[0], lj[1]])[MANO_TO_LEAP]
            diff = J[i] - L21
        per.append((f, p2m, speed[i], acc[i], diff))
    D = np.array([p[4] for p in per if p[4] is not None])
    off = D.mean(0) if len(D) else None  # session's mean offset per joint (definition differences)
    for f, p2m, sp, ac, diff in per:
        e = np.linalg.norm(diff, axis=1) if diff is not None else None
        e0 = np.linalg.norm(diff - off, axis=1) if diff is not None else None
        rows.append({"session": s[-6:], "pose": note, "frame": f, "p2m": p2m, "speed": float(sp), "jitter": ac,
                     "leap_err": e, "leap_err_centred": e0})
    print(f"{s}: {len(per)} frames evaluated", flush=True)


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def joint_errs(rs, key):
    out = {}
    for name, idx in TYPES.items():
        out[name] = med([float(np.median(r[key][idx])) for r in rs if r[key] is not None])
    return out


print(f"\n{'session':<8}{'pose':<14}{'frames':>7}{'surface mm':>11}{'p90':>6}{'speed mm/s':>11}{'jitter':>8}"
      f"{'vs LeapC mm':>12}{'centred':>9}")
for sid in dict.fromkeys(r["session"] for r in rows):
    rs = [r for r in rows if r["session"] == sid]
    p = np.array([r["p2m"] for r in rs])
    le = med([float(np.median(r["leap_err"])) for r in rs if r["leap_err"] is not None])
    lc = med([float(np.median(r["leap_err_centred"])) for r in rs if r["leap_err_centred"] is not None])
    print(f"{sid:<8}{rs[0]['pose']:<14}{len(rs):>7}{np.median(p):>11.2f}{np.percentile(p, 90):>6.2f}"
          f"{med([r['speed'] for r in rs]):>11.0f}{med([r['jitter'] for r in rs]):>8.2f}{le:>12.1f}{lc:>9.1f}")
    summary[sid] = {"pose": rs[0]["pose"], "frames": len(rs), "point_to_mesh_mm": float(np.median(p)),
                    "point_to_mesh_p90_mm": float(np.percentile(p, 90)), "speed_mm_s": med([r["speed"] for r in rs]),
                    "jitter_mm_per_frame2": med([r["jitter"] for r in rs]), "vs_leapc_mm": le, "vs_leapc_centred_mm": lc,
                    "vs_leapc_by_joint_mm": joint_errs(rs, "leap_err"),
                    "vs_leapc_centred_by_joint_mm": joint_errs(rs, "leap_err_centred")}

print("\nby hand speed (95th-percentile joint speed):")
bands = {}
for lo, hi in BANDS:
    rs = [r for r in rows if lo <= r["speed"] < hi]
    if not rs:
        continue
    name = f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+"
    p = np.array([r["p2m"] for r in rs])
    lc = med([float(np.median(r["leap_err_centred"])) for r in rs if r["leap_err_centred"] is not None])
    print(f"  {name:>8} mm/s: {len(rs):>5} frames, surface {np.median(p):.2f} mm (p90 {np.percentile(p, 90):.2f}), "
          f"vs LeapC centred {lc:.1f} mm")
    bands[name] = {"frames": len(rs), "point_to_mesh_mm": float(np.median(p)),
                   "point_to_mesh_p90_mm": float(np.percentile(p, 90)), "vs_leapc_centred_mm": lc}

print("\nMANO vs LeapC by joint type, all sessions (raw / mean offset removed per session), mm:")
allj, allc = joint_errs(rows, "leap_err"), joint_errs(rows, "leap_err_centred")
for k in TYPES:
    print(f"  {k:<30} {allj[k]:5.1f} / {allc[k]:5.1f}")

print("\nworst frames (surface distance):")
worst = sorted(rows, key=lambda r: -r["p2m"])[:8]
for r in worst:
    print(f"  {r['session']} {r['pose']:<14} frame {r['frame']}: {r['p2m']:.2f} mm at {r['speed']:.0f} mm/s")
json.dump({"sessions": summary, "speed_bands": bands, "vs_leapc_by_joint_mm": allj,
           "vs_leapc_centred_by_joint_mm": allc,
           "worst": [{k: r[k] for k in ("session", "pose", "frame", "p2m", "speed")} for r in worst]},
          open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}")
