"""dataset.json: one row per hand session in the training set (PLAN.md P3). Used by process.py and finish.py.

  python dataset.py                 # print the table
  python dataset.py --backfill      # add rows for sessions fitted before process.py existed

Rows: person, hand, notes, fps per camera, the rig board session, rig check (same-surface gap between the cameras'
clouds), fused / fitted frames, smoothing, MANO surface error, MANO vs LeapC, the Drive zip. People are ids (P01 ...),
never names.
"""
import argparse
import glob
import json
import os

import numpy as np

PATH = "dataset.json"
RIG_GAP_MAX = 3.0  # mm, PLAN.md P1


def load():
    if os.path.exists(PATH):
        return json.load(open(PATH))
    return {"people": {"P01": "first person; MANO shape in shapes/P01_left.json"}, "sessions": {}}


def save(d):
    d["sessions"] = dict(sorted(d["sessions"].items()))
    json.dump(d, open(PATH, "w"), indent=1)


def name_of(session):
    return os.path.basename(os.path.normpath(session))


def notes_of(session):
    return json.load(open(os.path.join(session, "session.json"))).get("notes", "")


def fps(session):
    """Frames per second per camera, from the device timestamps."""
    from session_io import load_rows
    out = {}
    for c in ("cam_A", "cam_B"):
        t = np.array([int(r["device_ts_us"]) for r in load_rows(os.path.join(session, c))])
        out[c] = round(float((len(t) - 1) / ((t[-1] - t[0]) / 1e6)), 1) if len(t) > 1 else None
    return out


def update(session, **fields):
    d = load()
    row = d["sessions"].setdefault(name_of(session), {})
    row.update(fields)
    save(d)
    return row


def table(d):
    cols = ("person", "hand", "fps", "rig_gap_mm", "fit_frames", "surface_mm", "vs_leapc_centred_mm", "drive_zip_mb")
    print(f"{'session':<20}{'notes':<26}" + "".join(f"{c[:12]:>13}" for c in cols))
    for s, r in d["sessions"].items():
        f = r.get("fps") or {}
        vals = [r.get("person"), r.get("hand"), f"{f.get('cam_A')}/{f.get('cam_B')}" if f else None,
                r.get("rig_gap_mm"), r.get("fit_frames"), r.get("surface_mm"), r.get("vs_leapc_centred_mm"),
                r.get("drive_zip_mb")]
        print(f"{s:<20}{r.get('notes', '')[:25]:<26}" + "".join(
            f"{(f'{v:.2f}' if isinstance(v, float) else str(v)) if v is not None else '-':>13}" for v in vals))
    n = sum(r.get("fit_frames") or 0 for r in d["sessions"].values())
    print(f"\n{len(d['sessions'])} sessions, {n} fitted frames")


def backfill():
    """Rows for the sessions fitted before process.py: N2 (2026-11-06, rig 033648) and N9 (2026-11-11, rig 230458)."""
    # rig checks printed by fuse.py when these were fused (full sessions); the other N2 sessions' were not kept
    gaps = {"2026-11-06_035647": 3.15, "2026-11-06_035830": 2.34, "2026-11-06_035901": 2.16,
            "2026-11-11_232548": 1.72, "2026-11-11_232631": 1.81, "2026-11-11_232713": 2.32, "2026-11-11_232756": 2.42,
            "2026-11-11_232838": 2.20, "2026-11-11_232944": 2.67}
    evals = {}
    for f in ("evaluation.json", "evaluation_n9.json"):
        if os.path.exists(f):
            evals.update(json.load(open(f))["sessions"])
    for s in sorted(glob.glob("sessions/20*/mano/smoothed.npz")):
        s = os.path.dirname(os.path.dirname(s))
        name = name_of(s)
        z = np.load(os.path.join(s, "mano", "smoothed.npz"))
        e = evals.get(name[-6:], {})
        rig = "sessions/2026-11-06_033648" if name < "2026-11-11" else "sessions/2026-11-11_230458"
        zp = os.path.join("drive_upload", f"{name}.zip")
        update(s, person="P01", hand="left", notes=notes_of(s), fps=fps(s), rig_board=rig, rig_gap_mm=gaps.get(name),
               fit_frames=int(len(z["frames"])), smoothing_sigma=float(z["sigma"]),
               surface_mm=e.get("point_to_mesh_mm"), surface_p90_mm=e.get("point_to_mesh_p90_mm"),
               vs_leapc_centred_mm=e.get("vs_leapc_centred_mm"),
               drive_zip=zp if os.path.exists(zp) else None,
               drive_zip_mb=round(os.path.getsize(zp) / 1e6) if os.path.exists(zp) else None)
        print("added", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    if a.backfill:
        backfill()
    table(load())
