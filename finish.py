"""P3 step 2: Kaggle's params.zip -> smoothed, evaluated, Drive-ready sessions, in one command.

  python finish.py                          # the newest params*.zip in fit_inputs/ or Downloads
  python finish.py fit_inputs/params.zip

Unzips the *_params.npz into fit_inputs/, copies each to <session>/mano/params.npz, smooths over time
(fit_mano.py --post: sigma 2 frames if the notes say "still", else 1), evaluates (tools/evaluate.py ->
fit_inputs/evaluation_<time>.json), packs for the Drive (tools/pack_drive.py -> drive_upload/<session>.zip) and
fills in dataset.json. Then drag the new drive_upload/*.zip into the shared Drive's sessions/ folder.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

import dataset

ap = argparse.ArgumentParser()
ap.add_argument("zip", nargs="?")
args = ap.parse_args()

zp = args.zip
if not zp:
    cands = glob.glob("fit_inputs/params*.zip") + glob.glob(os.path.expanduser("~/Downloads/params*.zip"))
    if not cands:
        sys.exit("no params.zip in fit_inputs/ or Downloads: download it from the Kaggle notebook's output first")
    zp = max(cands, key=os.path.getmtime)
print(f"using {zp} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(zp)))})")
os.makedirs("fit_inputs", exist_ok=True)
with zipfile.ZipFile(zp) as z:
    names = [n for n in z.namelist() if n.endswith("_params.npz")]
    z.extractall("fit_inputs", members=names)

sessions = []
for n in names:
    name = os.path.basename(n)[:-len("_params.npz")]
    s = os.path.join("sessions", name)
    if not os.path.isdir(s):
        print(f"{name}: no such session folder here, skipped")
        continue
    os.makedirs(os.path.join(s, "mano"), exist_ok=True)
    shutil.copy(os.path.join("fit_inputs", os.path.basename(n)), os.path.join(s, "mano", "params.npz"))
    sigma = 2 if "still" in dataset.notes_of(s).lower() else 1
    out = subprocess.run([sys.executable, "fit_mano.py", s, "--post", str(sigma)], capture_output=True, text=True)
    if out.returncode:
        print(f"{name}: smoothing failed\n{out.stderr[-1500:]}")
        continue
    sm = re.search(r"smoothed, sigma .*?point-to-mesh median ([\d.]+) mm; jitter median ([\d.]+)", out.stdout)
    sw = re.search(r"orientation switches > 90 deg at (\d+) steps", out.stdout)
    frames = len(np.load(os.path.join(s, "mano", "smoothed.npz"))["frames"])
    print(f"{name}: {frames} frames, smoothed (sigma {sigma}): point-to-mesh {sm.group(1) if sm else '?'} mm, "
          f"jitter {sm.group(2) if sm else '?'}" + (f", {sw.group(1)} palm/back switches" if sw else ""), flush=True)
    dataset.update(s, fit_frames=frames, smoothing_sigma=sigma, palm_back_switches=int(sw.group(1)) if sw else 0)
    sessions.append(s)

if sessions:
    ev = os.path.join("fit_inputs", f"evaluation_{time.strftime('%Y%m%d_%H%M')}.json")
    out = subprocess.run([sys.executable, "tools/evaluate.py", *sessions, "--out", ev], capture_output=True, text=True)
    if out.returncode:
        print("evaluation failed\n" + out.stderr[-1500:])
    else:
        print("\n" + "\n".join(line for line in out.stdout.splitlines() if "evaluated" not in line))
        e = json.load(open(ev))["sessions"]
        for s in sessions:
            r = e.get(dataset.name_of(s)[-6:], {})
            dataset.update(s, surface_mm=r.get("point_to_mesh_mm"), surface_p90_mm=r.get("point_to_mesh_p90_mm"),
                           vs_leapc_centred_mm=r.get("vs_leapc_centred_mm"), evaluation=ev)
    out = subprocess.run([sys.executable, "tools/pack_drive.py", *sessions], capture_output=True, text=True)
    print(out.stdout + (out.stderr[-1500:] if out.returncode else ""))
    for s in sessions:
        p = os.path.join("drive_upload", f"{dataset.name_of(s)}.zip")
        if os.path.exists(p):
            dataset.update(s, drive_zip=p, drive_zip_mb=round(os.path.getsize(p) / 1e6))
    print("next: drag these drive_upload/*.zip into the shared Drive's sessions/ folder")
