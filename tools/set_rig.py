"""Give hand sessions the rig of a board session (or any rig.json): copied into each session as its own rig.json.

  python tools/set_rig.py sessions/2026-11-12_120000 sessions/2026-11-12_103347 sessions/2026-11-12_103432
  python tools/set_rig.py sessions/2026-11-12_120000 --dates 2026-11-12_1033 2026-11-12_1041

The first argument is a board session that calib_rig.py has run on (its rig.json) or a rig.json file. With --dates
FROM TO, every hand session whose name sorts between FROM and TO gets it. The rig a session had is kept as
rig_prev.json. Fused clouds made with the old rig are now stale: run process.py on the sessions again.

Rule (PLAN.md P1): a hand session is only as good as a board recorded while the cameras were in the same place, so
record a board at the start and the end of every sitting and after any bump; then nothing ever has to be re-recorded.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo import pin_rig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rig", help="board session folder (with rig.json from calib_rig.py) or a rig.json file")
ap.add_argument("sessions", nargs="*")
ap.add_argument("--dates", nargs=2, metavar=("FROM", "TO"), help="all hand sessions with names in [FROM, TO]")
args = ap.parse_args()

src = args.rig if args.rig.endswith(".json") else os.path.join(args.rig, "rig.json")
if not os.path.exists(src):
    sys.exit(f"{src} not found: run calib_rig.py on the board session first")
rig = json.load(open(src))
sessions = list(args.sessions)
if args.dates:
    for p in sorted(glob.glob("sessions/20*")):
        n = os.path.basename(p)
        if args.dates[0] <= n <= args.dates[1] + "~" and os.path.exists(os.path.join(p, "session.json")):
            if "board" not in json.load(open(os.path.join(p, "session.json"))).get("notes", "").lower():
                sessions.append(p)
if not sessions:
    sys.exit("no sessions given")
print(f"rig from {rig.get('board_session', src)}: {rig.get('views')} views, corner residual "
      f"{rig.get('corner_residual_median_mm', float('nan')):.2f} mm")
for s in sessions:
    pin_rig(s, src)
    print(f"  {s}: rig set (old one kept as rig_prev.json if it had one)")
print("next: python process.py " + " ".join(sessions) + "   (re-fuses with this rig)")
