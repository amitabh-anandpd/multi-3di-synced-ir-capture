"""P3 step 1: recorded hand sessions -> Kaggle-ready fit inputs, in one command.

  python process.py --latest 6                 # the 6 newest sessions (no folder names to type)
  python process.py sessions/2026-11-11_232548 sessions/2026-11-11_232631
  person and hand come from each session's notes ("P02 left wrist rotation"); --person P02 --hand right override
  them, and a session whose notes name a different person or hand is refused. There is no default: a session
  packed as the wrong person or hand gets another hand's MANO shape (2026-11-20: two P02 takes came out as P01 left).
           --force                    (pack even if the rig check fails)
  a new person or hand (still poses with "still" in the notes + motion), e.g. P03's left hand:
    python process.py --latest 13            # fuses everything, then stops: P03 has no hand shape and no clicks
    python tools/click_joints.py --person P03 --hand left --still --per 3      # ~30 frames, ~30 min
    python process.py --no-fuse --latest 13  # shape from the clicks (tools/click_shape.py --fit), then packs
  So a new person's shape comes from the clicks before the first fit, and the sessions go to Kaggle once. A shape is
  never fitted without clicks unless --shape-without-clicks is given (2026-11-19: P02's ruler-constrained shape put
  its fingers 3-10 mm short, and every label with it).

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
ap.add_argument("--person", help="override the person named in the notes (e.g. P02)")
ap.add_argument("--hand", choices=["left", "right"], help="override the hand named in the notes")
ap.add_argument("--force", action="store_true", help="pack even when the rig check fails")
ap.add_argument("--no-fuse", action="store_true",
                help="reuse each session's existing fused clouds and rig check (after clicking): only shape + pack")
ap.add_argument("--shape-without-clicks", action="store_true",
                help="fit a missing hand shape from the point clouds alone (no clicks): only for a quick test")
ap.add_argument("--refine", action="store_true",
                help="let fuse.py correct the rig from the hand points (up to 6 deg / 20 mm; kept only if it lowers "
                     "the gap on frames it was not fitted to). For a hand held where the board was not: 2026-11-12 "
                     "right hand 6.4 -> 1.8 mm")
args = ap.parse_args()

def who(notes):
    """(person, hand, problem) for a session: from --person / --hand, else from its notes. Never a silent default."""
    np_ = re.search(r"\b(P\d+)\b", notes)
    nh = re.search(r"\b(left|right)\b", notes.lower())
    in_notes = (np_.group(1) if np_ else None, nh.group(1) if nh else None)
    person, hand = args.person or in_notes[0], args.hand or in_notes[1]
    for flag, given, noted in (("--person", args.person, in_notes[0]), ("--hand", args.hand, in_notes[1])):
        if given and noted and given != noted:
            return person, hand, f"{flag} {given}, but the notes say {noted}"
    if not person or not hand:
        return person, hand, "the notes name no person or hand: pass --person / --hand"
    return person, hand, None


sessions = list(args.sessions)
if args.latest:
    sessions += sorted(glob.glob("sessions/20*"))[-args.latest:]
if not sessions:
    sys.exit("give sessions or --latest N")
packed, failed, fused, who_of = [], [], [], {}
for s in sessions:
    t0 = time.time()
    name, notes = dataset.name_of(s), dataset.notes_of(s)
    if "board" in notes.lower():
        print(f"{name}: '{notes}' is a board session, skipped")
        continue
    person, hand, problem = who(notes)
    if problem:
        print(f"== {name}  '{notes}'\n   NOT processed: {problem}")
        failed.append(name)
        continue
    who_of[s] = (person, hand)
    print(f"== {name}  '{notes}'  ({person} {hand})", flush=True)
    row = dataset.load()["sessions"].get(name, {})
    if args.no_fuse and os.path.exists(os.path.join(s, "fused", "hands.npz")) and row.get("rig_gap_mm") is not None:
        gap = row["rig_gap_mm"]
        ok = gap <= dataset.RIG_GAP_MAX
        print(f"   already fused ({row.get('fused_frames')} frames); rig check: gap {gap} mm -> {'OK' if ok else 'FAILED'}")
        dataset.update(s, person=person, hand=hand)
        if not ok and not args.force:
            print(f"   NOT packed: rig gap {gap} mm > {dataset.RIG_GAP_MAX} (use --force, or re-run without --no-fuse)")
            failed.append(name)
            continue
        fused.append((s, t0))
        continue
    if args.no_fuse:
        print("   not fused yet: fusing it now")
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
    dataset.update(s, person=person, hand=hand, notes=notes, fps=fps, rig_board=rig.get("board_session"),
                   rig_gap_mm=gap, rig_overlap_pct=int(g.group(2)) if g else None,
                   fused_frames=int(m.group(1)) if m else None, fused_every=every,
                   rig_refined=bool(rf and rf.group(0).endswith("-> used")))
    if not ok and not args.force:
        print(f"   NOT packed: the cameras' clouds disagree by {gap} mm (> {dataset.RIG_GAP_MAX}). If the cameras "
              f"moved, record a board session and run calib_rig.py on it, then process.py again.")
        failed.append(name)
        continue
    fused.append((s, t0))

# a new person or hand: fit the MANO shape first (~1-2 min), from the person's clicked joints (tools/click_shape.py).
# One shape per person and hand, so a batch that mixes them gets each fitted from its own stills. Without enough
# clicks the sessions of that person and hand are held back, not packed with a guessed shape.
waiting = {}  # (person, hand) -> sessions held back until the hand is clicked
for (person, hand) in sorted({who_of[s] for s, _ in fused}):
    shape = os.path.join("shapes", f"{person}_{hand}.json")
    if os.path.exists(shape):
        continue
    mine = [s for s, _ in fused if who_of[s] == (person, hand)]
    still = [s for s in mine if "still" in dataset.notes_of(s).lower()]
    if not still:
        sys.exit(f"\nno hand shape {shape} yet, and none of these sessions is a still pose. Record 3-5 still poses "
                 f"(notes containing 'still', e.g. \"P02 fist still\") and run process.py on them together with these.")
    if args.shape_without_clicks:
        print(f"\nno {shape} yet: fitting {person}'s {hand} hand shape from {len(still)} still sessions WITHOUT "
              f"clicks (--shape-without-clicks: finger lengths are not constrained)", flush=True)
        out = subprocess.run([sys.executable, "fit_mano.py", "--shape", *still, "--person", person],
                             capture_output=True, text=True)
    else:
        print(f"\nno {shape} yet: fitting {person}'s {hand} hand shape from the clicked joints", flush=True)
        out = subprocess.run([sys.executable, "tools/click_shape.py", "--person", person, "--hand", hand, "--fit"],
                             capture_output=True, text=True)
        if out.returncode == 2 or "moments clicked" not in out.stdout or not os.path.exists(shape):
            print(out.stdout.strip() or out.stderr[-1500:])
            waiting[(person, hand)] = mine
            continue
    print("\n".join(x for x in out.stdout.splitlines()
                    if x.startswith(("shape", "MANO", "palm", "wrote", "  ", "P"))) or out.stderr[-1500:])
    if not os.path.exists(shape):
        sys.exit(f"shape fit did not write {shape} (LeapC may call this hand the other side: check the notes)")
held = {s for ss in waiting.values() for s in ss}
fused = [(s, t0) for s, t0 in fused if s not in held]

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
for (person, hand), ss in waiting.items():
    print(f"\nWAITING for clicks, {person} {hand} ({len(ss)} sessions fused, not packed): no hand shape yet, and it is "
          f"fitted from clicked joints, not guessed. Click the still poses (~30 frames, skip any with fingers hidden):\n"
          f"  python tools/click_joints.py --person {person} --hand {hand} --still --per 3\n"
          f"then pack them (the fusing is reused):\n"
          f"  python process.py --no-fuse {' '.join(ss)}")
if packed:
    print("next: upload these fit_inputs/<session>.npz files as a Kaggle dataset, run fit_kaggle.ipynb, download "
          "params.zip into fit_inputs/, then: python finish.py")
