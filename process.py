"""P3 step 1: recorded hand sessions -> Kaggle-ready fit inputs, in one command.

  python process.py --latest 6                 # the 6 newest sessions (no folder names to type)
  python process.py sessions/2026-11-11_232548 sessions/2026-11-11_232631
  options: --person P01 --hand left   (written to dataset.json; default P01 / left)
           --force                    (pack even if the rig check fails)
  a new person or hand, e.g. after recording P02's right hand (still poses with "still" in the notes + motion):
    python process.py --latest 13 --person P02 --hand right
  fits shapes/P02_right.json from the still sessions first, then packs everything with it.

Per session: pair_frames.py -> fps per camera -> fuse.py (every 3rd frame; every 2nd if the notes say "fast") ->
rig check (same-surface gap between the two cameras' clouds <= 3 mm, else the cameras moved: recalibrate) ->
fit_batch.py --pack into fit_inputs/. Every result goes into dataset.json. Then upload fit_inputs/*.npz (not the
*_params.npz) to Kaggle and run fit_kaggle.ipynb; afterwards: python finish.py
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import dataset
from stereo import pin_rig

ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="*")
ap.add_argument("--latest", type=int, help="take the N newest sessions")
ap.add_argument("--person", default="P01")
ap.add_argument("--hand", default="left")
ap.add_argument("--force", action="store_true", help="pack even when the rig check fails")
ap.add_argument("--refine", action="store_true",
                help="let fuse.py correct the rig from the hand points (up to 6 deg / 20 mm; kept only if it lowers "
                     "the gap on frames it was not fitted to). For a hand held where the board was not: 2026-11-12 "
                     "right hand 6.4 -> 1.8 mm")
args = ap.parse_args()

sessions = list(args.sessions)
if args.latest:
    sessions += sorted(glob.glob("sessions/20*"))[-args.latest:]
if not sessions:
    sys.exit("give sessions or --latest N")
packed, failed, fused = [], [], []
for s in sessions:
    t0 = time.time()
    name, notes = dataset.name_of(s), dataset.notes_of(s)
    if "board" in notes.lower():
        print(f"{name}: '{notes}' is a board session, skipped")
        continue
    print(f"== {name}  '{notes}'", flush=True)
    out = subprocess.run([sys.executable, "pair_frames.py", s], capture_output=True, text=True)
    if out.returncode:
        print(out.stdout[-500:], out.stderr[-1500:])
        failed.append(name)
        continue
    kept = re.search(r"groups kept: .*", out.stdout)
    fps = dataset.fps(s)
    print(f"   {kept.group(0) if kept else ''};  fps cam_A {fps['cam_A']}, cam_B {fps['cam_B']}"
          + ("   (below 85: Ultraleap Control Panel open?)" if min(v or 0 for v in fps.values()) < 85 else ""))
    # pin the rig: a session keeps the rig it was fused with (tools/set_rig.py gives it another one)
    if not os.path.exists(os.path.join(s, "rig.json")):
        pin_rig(s)
    rig = json.load(open(os.path.join(s, "rig.json")))
    every = 2 if "fast" in notes.lower() else 3
    refine = ["--max-mm", "20", "--max-deg", "6"] if args.refine else ["--no-refine"]
    out = subprocess.run([sys.executable, "fuse.py", s, "--every", str(every), *refine],
                         capture_output=True, text=True)
    rf = re.search(r"refinement: .*", out.stdout)
    if rf:
        print("   " + rf.group(0))
    if out.returncode:
        print(out.stdout[-500:], out.stderr[-1500:])
        failed.append(name)
        continue
    m = re.search(r"(\d+) frames fused, (\d+) points per frame", out.stdout)
    g = re.search(r"same-surface point-to-plane gap median ([\d.]+) mm.*overlap (\d+) %", out.stdout)
    gap = float(g.group(1)) if g else None
    ok = gap is not None and gap <= dataset.RIG_GAP_MAX
    print(f"   fused {m.group(1) if m else '?'} frames (every {every}), {m.group(2) if m else '?'} points/frame; "
          f"rig check: gap {gap} mm, overlap {g.group(2) if g else '?'} % -> {'OK' if ok else 'FAILED'}", flush=True)
    dataset.update(s, person=args.person, hand=args.hand, notes=notes, fps=fps, rig_board=rig.get("board_session"),
                   rig_gap_mm=gap, rig_overlap_pct=int(g.group(2)) if g else None,
                   fused_frames=int(m.group(1)) if m else None, fused_every=every,
                   rig_refined=bool(rf and rf.group(0).endswith("-> used")))
    if not ok and not args.force:
        print(f"   NOT packed: the cameras' clouds disagree by {gap} mm (> {dataset.RIG_GAP_MAX}). If the cameras "
              f"moved, record a board session and run calib_rig.py on it, then process.py again.")
        failed.append(name)
        continue
    fused.append((s, t0))

# a new person or hand: fit the MANO shape from this batch's still sessions first (fit_mano.py --shape, ~1-2 min)
shape = os.path.join("shapes", f"{args.person}_{args.hand}.json")
if fused and not os.path.exists(shape):
    still = [s for s, _ in fused if "still" in dataset.notes_of(s).lower()]
    if not still:
        sys.exit(f"\nno hand shape {shape} yet, and none of these sessions is a still pose. Record 3-5 still poses "
                 f"(notes containing 'still', e.g. \"P02 fist still\") and run process.py on them together with these.")
    print(f"\nno {shape} yet: fitting {args.person}'s {args.hand} hand shape from {len(still)} still sessions", flush=True)
    out = subprocess.run([sys.executable, "fit_mano.py", "--shape", *still, "--person", args.person],
                         capture_output=True, text=True)
    print("\n".join(x for x in out.stdout.splitlines() if x.startswith(("shape", "MANO", "wrote"))) or out.stderr[-1500:])
    if not os.path.exists(shape):
        sys.exit(f"shape fit did not write {shape} (LeapC may call this hand the other side: check --hand)")

for s, t0 in fused:
    name = dataset.name_of(s)
    out = subprocess.run([sys.executable, "fit_batch.py", "--pack", s], capture_output=True, text=True)
    line = [x for x in out.stdout.splitlines() if "->" in x]
    print("   " + (line[-1] if line else out.stderr[-800:]))
    if out.returncode:
        failed.append(name)
        continue
    dataset.update(s, fit_input=os.path.join("fit_inputs", f"{name}.npz"))
    packed.append(name)
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)

print(f"\npacked {len(packed)}: {packed}" + (f"\nNOT packed {len(failed)}: {failed}" if failed else ""))
if packed:
    print("next: upload these fit_inputs/<session>.npz files as a Kaggle dataset, run fit_kaggle.ipynb, download "
          "params.zip into fit_inputs/, then: python finish.py")
