"""Pack fitted sessions for the shared Drive / train_kaggle.ipynb (two_cam mode): one zip per session.

  python tools/pack_drive.py sessions/2026-11-06_035647 sessions/2026-11-11_232548 ...
  python tools/pack_drive.py --verify     # check the zips already packed against dataset.json, pack nothing

Keeps only the frame pairs that have a MANO label (mano/smoothed.npz), so the zips are a fraction of the session.
Each zip holds <session>/: session.json, groups.json (labelled groups only), cam_A|cam_B/frames.bin + frames.csv
(rewritten offsets) + tracking.csv, mano/smoothed.npz, rig.json (the rig the session was fused with) and label_info.json.
Writes drive_upload/<session>.zip (git-ignored); drag them into the Drive's sessions/ folder.
"""
import argparse
import csv
import glob
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
    pinned = os.path.join("sessions", name, "rig.json")  # pinned by process.py / tools/set_rig.py
    if os.path.exists(pinned):
        return pinned
    path = None
    for start, p in RIGS:
        if name >= start:
            path = p
    return path


def side_for(name, meta):
    """left/right, from dataset.json's `hand` (same source fit_mano.py uses), else the session notes."""
    try:
        return json.load(open("dataset.json"))["sessions"][name]["hand"]
    except (OSError, KeyError, ValueError):
        notes = (meta.get("notes") or "").lower()
        return "right" if "right" in notes else "left" if "left" in notes else None


def verify(out_dir):
    """Check every packed zip against dataset.json before it goes to the Drive.

    Returns (zips, bad, stale). `bad` is wrong data — a zip whose stated hand contradicts dataset.json would
    mirror the hand once the consumer builds a MANO mesh from it. `stale` is only a zip packed before the
    "side" key existed: its prose is still right, so it is reported but does not fail the check.
    """
    rows = json.load(open("dataset.json"))["sessions"]
    bad, stale = [], []
    zips = sorted(glob.glob(os.path.join(out_dir, "*.zip")))
    for f in zips:
        name = os.path.basename(f)[:-len(".zip")]
        row = rows.get(name)
        try:
            zf = zipfile.ZipFile(f)
            names = zf.namelist()
        except (OSError, zipfile.BadZipFile) as e:
            bad.append((name, f"unreadable: {e}"))
            continue
        if row is None:
            bad.append((name, "not in dataset.json"))
            continue
        if row.get("exclude"):
            bad.append((name, "marked exclude in dataset.json but packed for the Drive"))
        if not any(n.endswith("mano/smoothed.npz") for n in names):
            bad.append((name, "no mano/smoothed.npz (the labels)"))
        li = [n for n in names if n.endswith("label_info.json")]
        if not li:
            bad.append((name, "no label_info.json"))
            continue
        info = json.loads(zf.read(li[0]))
        hand = row.get("hand")
        if f"({hand} hand" not in info.get("labels", ""):
            bad.append((name, f"label_info says it is not a {hand} hand, but dataset.json says it is"))
        elif info.get("side") is None:
            stale.append(name)  # packed before the "side" key; prose is correct, so re-pack when convenient
        elif info.get("side") != hand:
            bad.append((name, f"label_info side {info.get('side')!r} != dataset.json hand {hand!r}"))
    return zips, bad, stale


ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="*")
ap.add_argument("--out", default="drive_upload")
ap.add_argument("--verify", action="store_true",
                help="check the zips already in --out against dataset.json and exit (packs nothing)")
args = ap.parse_args()
if args.verify:
    checked, problems, stale = verify(args.out)
    for name, why in problems:
        print(f"  WRONG  {name}: {why}")
    if stale:
        print(f"  {len(stale)} zip(s) packed before the \"side\" key (prose is correct; re-pack when convenient): "
              f"{', '.join(stale[:4])}{', ...' if len(stale) > 4 else ''}")
    print(f"{len(checked) - len({n for n, _ in problems})} of {len(checked)} zips OK in {args.out}")
    sys.exit(1 if problems else 0)
if not args.sessions:
    ap.error("give sessions to pack, or --verify")
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
    side = side_for(name, meta)
    dst = os.path.join(args.out, f"{name}.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(os.path.join(s, "session.json"), f"{name}/session.json")
        zf.writestr(f"{name}/groups.json", json.dumps(g))
        zf.write(os.path.join(s, "mano", "smoothed.npz"), f"{name}/mano/smoothed.npz")
        if rig:
            zf.write(rig, f"{name}/rig.json")
        zf.writestr(f"{name}/label_info.json", json.dumps({
            "labels": "mano/smoothed.npz: joints (n, 21, 3) mm in cam_A's rectified-left (unit) frame, per frame id in "
                      f"'frames'; orient/pose/transl/betas are the MANO parameters ({side} hand, flat_hand_mean, "
                      "no PCA)",
            "side": side,  # load MANO with is_rhand=(side == "right"); left and right are mirror images
            "joint_order": "MANO 16 joints + 5 tips (thumb, index, middle, ring, pinky tip as 16..20)",
            "flip_dist": "smoothed.npz['flip_dist'] (when present): per frame, fitted steps to the nearest palm/back "
                         "flip of the fit, -1 = the session has none. Labels near a flip are worse (vs hand-clicked "
                         "joints: 31 mm at 0-1 steps, 21 mm at 2-20, 8-13 mm beyond): drop 0-1, consider <= 20",
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
