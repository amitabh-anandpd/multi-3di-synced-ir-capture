"""P6: free disk by deleting a person's raw IR frames, once their labels are safely packed.

  python tools/prune_raw.py P01              # dry run: say what would go, delete nothing
  python tools/prune_raw.py P01 --delete     # actually delete

Deletes, per session of that person: cam_*/frames.bin (the raw IR images) and fused/*.ply + fused/*.mp4
(previews). KEEPS fused/hands.npz, tracking.csv, groups.json, session.json, rig.json and mano/ — so the
session can still be re-packed for Kaggle (fit_batch.py --pack), have its shape refitted and be re-evaluated
(tools/evaluate.py). All of those read fused/hands.npz, not frames.bin.

What you give up is re-FUSING (a different stride or segmentation) and re-packing the Drive zip, because
tools/pack_drive.py copies the labelled frames out of frames.bin. So a session is only prunable once its
drive_upload/<session>.zip exists and passes tools/pack_drive.py --verify; --delete refuses otherwise.

That check covers the local zip only. Confirm the copies on the shared Drive open before deleting, since
after this the zip is the sole copy of those frames.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEEP = "fused/hands.npz, tracking.csv, groups.json, session.json, rig.json, mano/"


def targets_for(session):
    """The files this tool would delete in one session folder."""
    out = []
    for root, _, files in os.walk(session):
        for f in files:
            if f == "frames.bin" or f.endswith((".ply", ".mp4")):
                out.append(os.path.join(root, f))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("person", help="person id in dataset.json, e.g. P01")
ap.add_argument("--delete", action="store_true", help="actually delete (default: dry run)")
ap.add_argument("--force", action="store_true", help="delete even for sessions with no verified Drive zip")
args = ap.parse_args()

rows = json.load(open("dataset.json"))["sessions"]
mine = {k: v for k, v in rows.items() if v.get("person") == args.person}
if not mine:
    sys.exit(f"no sessions for {args.person} in dataset.json")

# A session is safe to prune only if its labelled frames already live in a verified Drive zip. An excluded
# session has no zip and no labels worth keeping, so its raw frames are prunable once it is not being re-fitted.
unsafe = []
for name, row in mine.items():
    if row.get("exclude"):
        continue
    zp = os.path.join("drive_upload", f"{name}.zip")
    if not os.path.exists(zp):
        unsafe.append((name, "no drive_upload zip"))

if not args.force and not unsafe:
    v = subprocess.run([sys.executable, "tools/pack_drive.py", "--verify"], capture_output=True, text=True)
    if v.returncode:
        print(v.stdout.strip())
        sys.exit("drive_upload/ failed verification: fix those zips before pruning anything")

total, planned = 0, {}
for name in sorted(mine):
    d = os.path.join("sessions", name)
    if not os.path.isdir(d):
        continue
    if not args.force and any(name == u for u, _ in unsafe):
        continue
    fs = targets_for(d)
    if fs:
        planned[name] = fs
        total += sum(os.path.getsize(f) for f in fs)

for name, _ in unsafe:
    print(f"  SKIP {name}: no verified Drive zip (use --force to prune anyway)")
n = sum(len(v) for v in planned.values())
print(f"{'DELETING' if args.delete else 'DRY RUN'}: {n} files, {total / 1e9:.2f} GB "
      f"across {len(planned)} of {len(mine)} {args.person} sessions")
print(f"keeping: {KEEP}")
if not args.delete:
    for name in list(planned)[:3]:
        print(f"   {name}: {len(planned[name])} files")
    print(f"\nre-run with --delete to remove them. The Drive zips become the only copy of these frames:\n"
          f"check they open on the shared Drive first.")
    sys.exit(0)

freed = 0
for name, fs in planned.items():
    for f in fs:
        try:
            freed += os.path.getsize(f)
            os.remove(f)
        except OSError as e:
            print(f"  could not delete {f}: {e}")
print(f"freed {freed / 1e9:.2f} GB; {args.person}'s sessions keep {KEEP}")
