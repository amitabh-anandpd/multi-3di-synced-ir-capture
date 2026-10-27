"""Export frames of a raw session as PNGs, for viewing or tools that want image files.

  python tools/export_png.py                         # newest session, every frame, all cameras
  python tools/export_png.py sessions/X --every 90   # ~1 frame per second

Writes <session>/<cam>/png/<frame_id>_L.png and _R.png.
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_io import load_rows, load_session, read_pair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session", nargs="?", help="session folder (default: newest)")
ap.add_argument("--cams", nargs="+", help="default: all cameras in the session")
ap.add_argument("--every", type=int, default=1, help="export every Nth frame")
args = ap.parse_args()
session = args.session or max(glob.glob(os.path.join("sessions", "20*")))

meta = load_session(session)
for cam in args.cams or sorted(meta["cameras"]):
    cam_dir = os.path.join(session, cam)
    out = os.path.join(cam_dir, "png")
    os.makedirs(out, exist_ok=True)
    rows = load_rows(cam_dir)[::args.every]
    for row in rows:
        for side, img in zip("LR", read_pair(cam_dir, row, meta["image_size"])):
            cv2.imwrite(os.path.join(out, f"{int(row['frame_id']):08d}_{side}.png"), img)
    print(f"{cam}: {len(rows)} frames -> {out}")
