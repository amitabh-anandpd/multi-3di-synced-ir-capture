"""A person's hand measurements from their clicked joints, and the MANO shape fitted to them, before any MANO fit.

  python tools/click_shape.py --person P03 --hand left           # measure, and print the shape-fit command
  python tools/click_shape.py --person P03 --hand left --fit     # measure and fit shapes/P03_left.json

Clicks come from `python tools/click_joints.py --person P03 --hand left --still` (frames clicked in both cameras).
Each moment is triangulated (tools/clicks3d.py); finger lengths (MCP / thumb CMC to tip) and palm width (index to
little MCP) are medians over the moments where every joint of that measurement was clicked. These are the numbers
fit_mano.py --shape takes as --click-lengths / --thumb-mm / --palm-mm.

Why clicks and not a ruler: a ruler measures skin landmarks (the knuckles' outer width, the visible finger base),
MANO's joints are joint centres inside the hand. P02's ruler palm of 85 mm against a clicked 69-74 mm shortened
every finger 3-10 mm (STATUS.md, 2026-11-19). Never constrain a shape with ruler or tracing numbers.

Exit code 2 when there are too few clicked moments, so process.py can tell "click more" from a failure.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clicks3d import clicked_pairs, finger_length, palm_width, triangulate  # noqa: E402
from stereo import Unit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--person", required=True)
ap.add_argument("--hand", required=True, choices=["left", "right"])
ap.add_argument("--min-moments", type=int, default=5,
                help="fewest moments per measurement before it is trusted (P02 had 9-14 per finger)")
ap.add_argument("--fit", action="store_true", help="run fit_mano.py --shape with these measurements")
args = ap.parse_args()

rows = json.load(open("dataset.json"))["sessions"]
mine = {s for s, r in rows.items() if r.get("person") == args.person and r.get("hand") == args.hand
        and not r.get("exclude")}
if not mine:
    sys.exit(f"no {args.person} {args.hand} sessions in dataset.json (run process.py on them first)")

units, X = {}, []
for (sess, fa), cc in sorted(clicked_pairs(lambda s: s in mine).items()):
    if len(cc) < 2:
        continue
    path = os.path.join("sessions", sess)
    if sess not in units:
        serials = json.load(open(os.path.join(path, "session.json")))["cameras"]
        units[sess] = (Unit(serials["cam_A"]), Unit(serials["cam_B"]))
    X.append(triangulate(path, cc["cam_A"], cc["cam_B"], *units[sess]))
print(f"{args.person} {args.hand}: {len(X)} moments clicked in both cameras, from {len(units)} sessions")

meas, short = {}, []
for name in ("index", "middle", "ring", "little", "thumb", "palm"):
    v = [palm_width(x) if name == "palm" else finger_length(x, name) for x in X]
    v = [a for a in v if a is not None]
    if len(v) < args.min_moments:
        short.append(f"{name} ({len(v)})")
        continue
    meas[name] = float(np.median(v))
    print(f"  {name:<7} {meas[name]:6.1f} mm   p25 {np.percentile(v, 25):6.1f}  p75 {np.percentile(v, 75):6.1f}  "
          f"[{len(v)} moments]")
if short:
    print(f"\ntoo few moments (need {args.min_moments}) for: {', '.join(short)}. Click more frames where these joints "
          f"are visible in BOTH cameras (open or spread hand):\n"
          f"  python tools/click_joints.py --person {args.person} --hand {args.hand} --still --per 4")
    sys.exit(2)

stills = sorted(os.path.join("sessions", s) for s in mine if "still" in (rows[s].get("notes") or "").lower()
                and os.path.exists(os.path.join("sessions", s, "fused", "hands.npz")))
if not stills:
    sys.exit(f"no fused still sessions of {args.person} {args.hand} to fit the shape to")
cmd = [sys.executable, "-u", "fit_mano.py", "--shape", *stills, "--person", args.person,
       "--click-lengths", *(f"{meas[f]:.1f}" for f in ("index", "middle", "ring", "little")),
       "--thumb-mm", f"{meas['thumb']:.1f}", "--palm-mm", f"{meas['palm']:.1f}"]
print("\nshape fit:\n  " + " ".join(cmd[1:]))
if args.fit:
    out = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join(x for x in out.stdout.splitlines() if x.startswith(("shape", "MANO", "palm", "wrote")))
          or out.stderr[-1500:])
    sys.exit(out.returncode)
