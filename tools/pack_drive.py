"""Pack fitted sessions for the shared Drive / train_kaggle.ipynb (two_cam mode): one zip per session.

  python tools/pack_drive.py sessions/2026-11-06_035647 sessions/2026-11-11_232548 ...

Keeps only the frame pairs that have a MANO label (mano/smoothed.npz), so the zips are a fraction of the session.
Each zip holds <session>/: session.json, groups.json (labelled groups only), cam_A|cam_B/frames.bin + frames.csv
(rewritten offsets) + tracking.csv, mano/smoothed.npz, rig.json (the rig the session was fused with) and label_info.json.
Writes drive_upload/<session>.zip (git-ignored); drag them into the Drive's sessions/ folder.
"""
import argparse
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_io import load_rows  # noqa: E402

# rig each session was fused with (rig.json changes when the cameras move)
RIGS = [("2026-11-06_000000", "sessions/2026-11-06_033648/rig.json"),
        ("2026-11-11_225000", "sessions/2026-11-11_230458/rig.json")]


def rig_for(name):
    path = None
    for start, p in RIGS:
        if name >= start:
            path = p
    return path


ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="+")
ap.add_argument("--out", default="drive_upload")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
for s in args.sessions:
    name = os.path.basename(os.path.normpath(s))
    meta = json.load(open(os.path.join(s, "session.json")))
    w, h = meta["image_size"]
    z = np.load(os.path.join(s, "mano", "smoothed.npz"))
    labelled = set(int(f) for f in z["frames"])
    g = json.load(open(os.path.join(s, "groups.json")))
    g["groups"] = [x for x in g["groups"] if x["frames"]["cam_A"] in labelled]
    want = {c: {x["frames"][c] for x in g["groups"]} for c in ("cam_A", "cam_B")}
    rig = rig_for(name)
    dst = os.path.join(args.out, f"{name}.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(os.path.join(s, "session.json"), f"{name}/session.json")
        zf.writestr(f"{name}/groups.json", json.dumps(g))
        zf.write(os.path.join(s, "mano", "smoothed.npz"), f"{name}/mano/smoothed.npz")
        if rig:
            zf.write(rig, f"{name}/rig.json")
        zf.writestr(f"{name}/label_info.json", json.dumps({
            "labels": "mano/smoothed.npz: joints (n, 21, 3) mm in cam_A's rectified-left (unit) frame, per frame id in "
                      "'frames'; orient/pose/transl/betas are the MANO parameters (left hand, flat_hand_mean, no PCA)",
            "joint_order": "MANO 16 joints + 5 tips (thumb, index, middle, ring, pinky tip as 16..20)",
            "rig": f"rig.json = cam_B in cam_A's frame, from {rig}",
            "frames": "only frame pairs with a MANO label are kept; groups.json pairs cam_A and cam_B frame ids"}, indent=1))
        for c in ("cam_A", "cam_B"):
            rows = [r for r in load_rows(os.path.join(s, c)) if int(r["frame_id"]) in want[c]]
            buf, out = io.StringIO(), io.BytesIO()
            wr = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            with open(os.path.join(s, c, "frames.bin"), "rb") as f:
                for r in rows:
                    f.seek(int(r["offset"]))
                    data = f.read(2 * w * h)
                    r = dict(r, offset=out.tell())
                    out.write(data)
                    wr.writerow(r)
            zf.writestr(f"{name}/{c}/frames.csv", buf.getvalue())
            zf.writestr(f"{name}/{c}/frames.bin", out.getvalue())
            tr = os.path.join(s, c, "tracking.csv")
            if os.path.exists(tr):
                zf.write(tr, f"{name}/{c}/tracking.csv")
    print(f"{name}: {len(g['groups'])} labelled pairs -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)", flush=True)
