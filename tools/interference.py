"""M7: does a second camera's IR hurt this camera's images? Same static scene, two sessions:

  python capture_multi.py --duration 10 --cams cam_A --notes "90 deg, A alone"     # cam_B unplugged
  python capture_multi.py --duration 10 --notes "90 deg, A with B on"
  python tools/interference.py sessions/ALONE sessions/WITH_B

Per session: mean brightness, % saturated pixels, flicker (std of the per-frame mean brightness,
as % of the mean). Writes <WITH_B>/<cam>/interference.png: alone on top, with B below.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_io import load_rows, load_session, read_pair  # noqa: E402


def stats(session, cam, every, sat):
    """(mean brightness, % saturated, flicker %, middle frame as one left|right image)."""
    cam_dir = os.path.join(session, cam)
    size = load_session(session)["image_size"]
    rows = load_rows(cam_dir)[::every]
    means, sats = [], []
    for row in rows:
        img = np.asarray(read_pair(cam_dir, row, size))
        means.append(img.mean())
        sats.append(100 * (img >= sat).mean())
    example = np.hstack(list(np.asarray(read_pair(cam_dir, rows[len(rows) // 2], size))))
    return np.mean(means), np.mean(sats), 100 * np.std(means) / np.mean(means), example


ap = argparse.ArgumentParser()
ap.add_argument("alone", help="session with only this camera on")
ap.add_argument("together", help="session with the other camera on too")
ap.add_argument("--cam", default="cam_A", help="camera being checked")
ap.add_argument("--every", type=int, default=3, help="use every Nth frame")
ap.add_argument("--sat", type=int, default=250, help="gray level counted as saturated")
args = ap.parse_args()

a = stats(args.alone, args.cam, args.every, args.sat)
b = stats(args.together, args.cam, args.every, args.sat)
print(f"{args.cam}{'':<7}{'brightness':>12}{'saturated %':>13}{'flicker %':>11}")
print(f"{'alone':<12}{a[0]:>12.1f}{a[1]:>13.2f}{a[2]:>11.2f}")
print(f"{'with other':<12}{b[0]:>12.1f}{b[1]:>13.2f}{b[2]:>11.2f}")
print(f"{'change':<12}{b[0] - a[0]:>+12.1f}{b[1] - a[1]:>+13.2f}{b[2] - a[2]:>+11.2f}")

out = os.path.join(args.together, args.cam, "interference.png")
cv2.imwrite(out, np.vstack([a[3], b[3]]))
print(f"wrote {out}")
