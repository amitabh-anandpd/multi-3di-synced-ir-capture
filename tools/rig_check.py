"""P1: is rig.json still right? Run it on the first hand recording of every sitting, before recording more.

  python tools/rig_check.py                  # the newest session that is not a board session
  python tools/rig_check.py sessions/2026-11-11_232548

Pairs the frames if needed, fuses ~150 frame groups into a scratch folder (the session's own fused/ is not touched)
and prints the same-surface gap between the two cameras' hand clouds and the frame rate. Gap <= 3 mm: OK, record on.
More: a camera moved since the board session in rig.json -> record a board session and run calib_rig.py on it.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json  # noqa: E402

import numpy as np  # noqa: E402

import dataset  # noqa: E402
from stereo import rig_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("session", nargs="?")
ap.add_argument("--groups", type=int, default=150, help="frame groups to fuse (every 3rd)")
ap.add_argument("--max-disp", type=int, default=128, help="passed to fuse.py (192 for a hand held close)")
args = ap.parse_args()

s = args.session
if not s:
    hands = [p for p in sorted(glob.glob("sessions/20*")) if os.path.exists(os.path.join(p, "session.json"))
             and "board" not in dataset.notes_of(p).lower()]
    if not hands:
        sys.exit("no hand session found")
    s = hands[-1]
print(f"{dataset.name_of(s)}  '{dataset.notes_of(s)}'")
if not os.path.exists(os.path.join(s, "groups.json")):
    subprocess.run([sys.executable, "pair_frames.py", s], capture_output=True, check=True)
fps = dataset.fps(s)
tmp = tempfile.mkdtemp(prefix="rig_check_")
dist = None
try:
    out = subprocess.run([sys.executable, "fuse.py", s, "--every", "3", "--no-refine", "--ply-every", "0",
                          "--max-frames", str(args.groups), "--out", tmp, "--max-disp", str(args.max_disp)],
                         capture_output=True, text=True)
    hp = os.path.join(tmp, "hands.npz")
    if os.path.exists(hp):  # how far the hand was from each camera (both clouds are in cam_A's frame)
        z = np.load(hp)
        rig = json.load(open(rig_path(s)))
        cb = np.array(rig["cameras"]["cam_B"]["t_mm"])  # cam_B's centre in cam_A's frame
        P = z["points"]
        dist = (float(np.median(np.linalg.norm(P, axis=1))), float(np.median(np.linalg.norm(P - cb, axis=1))))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
g = re.search(r"same-surface point-to-plane gap median ([\d.]+) mm.*overlap (\d+) %", out.stdout)
if not g:
    sys.exit("fuse.py gave no gap (is a hand in both views?)\n" + out.stdout[-800:] + out.stderr[-1500:])
gap, overlap = float(g.group(1)), int(g.group(2))
ok = gap <= dataset.RIG_GAP_MAX
print(f"rig check: gap {gap:.2f} mm, overlap {overlap} %  ->  {'OK' if ok else 'FAILED'}")
if dist:
    print(f"hand distance: {dist[0]:.0f} mm from cam_A, {dist[1]:.0f} mm from cam_B")
print(f"fps: cam_A {fps['cam_A']}, cam_B {fps['cam_B']}"
      + ("  (below 85: close the Ultraleap Control Panel, plug the laptop in)" if min(v or 0 for v in fps.values()) < 85 else ""))
if not ok:
    print("A camera has probably moved: record a board session (40 s, hold each pose still), then\n"
          "  python pair_frames.py <board session>\n  python calib_rig.py <board session>\nand check again.")
