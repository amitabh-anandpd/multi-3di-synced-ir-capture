"""Record raw stereo IR from all (or some) 3Dis, at the same time or one after another.

  python capture_multi.py --duration 60
  python capture_multi.py --cams cam_A --notes "cam_A alone"
  python capture_multi.py --sequential --duration 10 --notes "static scene"

The poll loop only copies frames onto a per-camera queue; one writer thread per
camera appends them to cam_X/frames.bin, so slow disk writes show up as queue_depth,
not dropped frames. One file per camera: creating ~360 PNGs/s cost ~15-20 % FPS (M4).
Read frames back with session_io.py; export PNGs with tools/export_png.py.
Ctrl+C stops early; queued frames are still written.
LeapC's own hand tracking goes to cam_X/tracking.csv (one row per hand per tracking frame,
columns in leapc.HAND_COLUMNS, LeapC tracking space in mm).

--sequential (M8, static scenes only): records each camera in turn with the others off, so
their IR LEDs can't reach it. LeapC can't switch off one device (LeapSetPause pauses all), so
the others are disabled in Windows with pnputil (needs an Administrator terminal) and
re-enabled afterwards. Without admin rights it asks you to unplug / replug instead.
Same session layout; pair_frames.py pairs a sequential session by time since each camera's start.
"""
import argparse
import csv
import ctypes
import datetime
import json
import os
import queue
import re
import subprocess
import threading
import time
from ctypes import byref

import numpy as np

import leapc
from list_devices import CONFIG

VERSION_TXT = r"C:\Program Files\Ultraleap\version.txt"


def writer(cam_dir, q):
    """frames.bin: per frame, left then right image, raw uint8 (size in session.json image_size)."""
    with open(os.path.join(cam_dir, "frames.csv"), "w", newline="") as f, \
            open(os.path.join(cam_dir, "frames.bin"), "wb") as b:
        w = csv.writer(f)
        w.writerow(["frame_id", "device_ts_us", "host_ts_us", "matrix_version", "offset", "queue_depth"])
        while True:
            item = q.get()
            if item is None:
                return
            frame_id, dev_ts, host_ts, mver, left, right, depth = item
            offset = b.tell()
            b.write(left)
            b.write(right)
            w.writerow([frame_id, dev_ts, host_ts, mver, offset, depth])


def connect_wait(serials, timeout_s):
    """leapc.connect until every serial is there (a re-enabled device takes a few seconds to come back)."""
    deadline = time.monotonic() + timeout_s
    while True:
        conn, devices = leapc.connect(serials=serials)
        missing = set(serials) - {d["serial"] for d in devices}
        if not missing or time.monotonic() > deadline:
            return conn, devices, missing
        leapc.close(conn, devices)
        time.sleep(1)


def record(names, session, duration, meta, wait_s=5.0):
    """Record cameras `names` into session/<name>/ for `duration` s. Adds them to meta. False if aborted."""
    name_of = {cameras[n]: n for n in names}
    conn, devices, missing = connect_wait(list(name_of), wait_s)
    if missing:
        leapc.close(conn, devices)
        print(f"ERROR: not connected: {[name_of[s] for s in missing]}")
        return False
    cams = {}  # device id -> state
    for d in devices:
        name = name_of[d["serial"]]
        cam_dir = os.path.join(session, name)
        os.makedirs(cam_dir)
        q = queue.Queue()
        t = threading.Thread(target=writer, args=(cam_dir, q))
        t.start()
        cams[d["id"]] = {"name": name, "dir": cam_dir, "q": q, "thread": t, "mver": None, "n": 0, "max_q": 0,
                         "handle": d["handle"], "track": []}
        meta["cameras"][name] = d["serial"]
        meta["baseline_um"][name] = d["baseline_um"]
    print(f"Recording {sorted(c['name'] for c in cams.values())} for {duration:g}s -> {session}  (Ctrl+C to stop)")

    msg = leapc.LEAP_CONNECTION_MESSAGE()
    t_start = time.monotonic()
    t_end, checked, ok = t_start + duration, False, True
    meta.setdefault("camera_start", {}).update(
        {c["name"]: datetime.datetime.now().isoformat(timespec="seconds") for c in cams.values()})
    try:
        while time.monotonic() < t_end:
            if not checked and time.monotonic() - t_start > 2:
                silent = [c["name"] for c in cams.values() if c["n"] == 0]
                if silent:  # service has the device but no video (seen after a bad replug)
                    print(f"ERROR: no frames from {silent} after 2 s. Replug it or restart the "
                          "Ultraleap Tracking Service, then retry.")
                    meta["notes"] += f" | ABORTED: no frames from {silent}"
                    ok = False
                    break
                checked = True
            if leapc.lib.LeapPollConnection(conn, 100, byref(msg)) != leapc.RS_SUCCESS:
                continue
            if msg.type == leapc.EVT_DEVICE_LOST:
                print("WARNING: a device was lost during capture")
            if msg.type == leapc.EVT_TRACKING and msg.device_id in cams:  # LeapC's own hand tracking
                fid, ts, rows = leapc.hands(msg)
                cams[msg.device_id]["track"] += [[fid, ts, leapc.lib.LeapGetNow()] + r for r in rows]
                continue
            if msg.type != leapc.EVT_IMAGE or msg.device_id not in cams:
                continue
            host_ts = leapc.lib.LeapGetNow()
            c = cams[msg.device_id]
            ev = leapc.image_event(msg)
            left, right = ev.image
            p = left.properties
            mver = left.matrix_version
            if c["name"] not in meta["calibration"]:
                meta["calibration"][c["name"]] = leapc.calibration(conn, c["handle"])
            if mver != c["mver"]:  # new distortion map: save it (pointer is valid only until the next poll)
                np.savez(os.path.join(c["dir"], f"distortion_v{mver}.npz"),
                         left=leapc.distortion(left), right=leapc.distortion(right))
                c["mver"] = mver
            imgs = [leapc.image_bytes(i) for i in (left, right)]  # copies: LeapC's buffer is reused next poll
            meta.setdefault("image_size", [p.width, p.height])  # LeapC memory: read it now, not after close
            depth = c["q"].qsize()
            c["max_q"] = max(c["max_q"], depth)
            c["q"].put((ev.info.frame_id, ev.info.timestamp, host_ts, mver, *imgs, depth))
            c["n"] += 1
    except KeyboardInterrupt:
        print("Stopped early.")
        ok = False
    leapc.close(conn, devices)

    print("Writing queued frames...")
    for c in cams.values():
        c["q"].put(None)
    for c in cams.values():
        c["thread"].join()
        with open(os.path.join(c["dir"], "tracking.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "device_ts_us", "host_ts_us"] + leapc.HAND_COLUMNS)
            w.writerows(c["track"])
        print(f"{c['name']}: {c['n']} frames, max queue depth {c['max_q']}, "
              f"{len({r[0] for r in c['track']})} tracking frames with a hand")
        meta["frames"][c["name"]] = c["n"]
    return ok


def pnp_ids(serials):
    """Windows device instance id of each 3Di, found by its serial (also lists disabled devices)."""
    out = subprocess.run(["pnputil", "/enum-devices", "/class", "USB"], capture_output=True, text=True).stdout
    ids = re.findall(r"Instance ID:\s+(\S+)", out)
    return {s: next((i for i in ids if i.upper().endswith("\\" + s.upper())), None) for s in serials}


def set_enabled(instance_id, on):
    r = subprocess.run(["pnputil", "/enable-device" if on else "/disable-device", instance_id],
                       capture_output=True, text=True)
    if r.returncode != 0 and "already" not in r.stdout:  # "Device is already enabled." also exits non-zero
        raise RuntimeError(f"pnputil failed for {instance_id}:\n{r.stdout}{r.stderr}")


ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=10, help="seconds (per camera with --sequential)")
ap.add_argument("--cams", nargs="+", help="camera names from config.json (default: all)")
ap.add_argument("--notes", default="")
ap.add_argument("--sequential", action="store_true", help="one camera at a time, the others off (static scenes)")
ap.add_argument("--isolate", choices=["auto", "pnputil", "prompt"], default="auto",
                help="--sequential: how to turn the other cameras off (auto = pnputil if admin, else prompt)")
args = ap.parse_args()

tasks = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True).stdout
if "Ultraleap Control Panel" in tasks:  # its live view halved our FPS (49-55 vs 88-90, 2026-10-27)
    print("WARNING: Ultraleap Control Panel is running and slows capture to ~50 FPS. Close it (and its tray icon).")

cameras = json.load(open(CONFIG))["cameras"]  # name -> serial
names = args.cams or sorted(cameras)
unknown = set(names) - set(cameras)
if unknown:
    raise SystemExit(f"Not in config.json: {sorted(unknown)}. Run list_devices.py.")

session = os.path.join("sessions", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
meta = {
    "cameras": {},
    "mode": "sequential" if args.sequential else "simultaneous",
    "sdk_version": open(VERSION_TXT).read().strip() if os.path.exists(VERSION_TXT) else None,
    "start": datetime.datetime.now().isoformat(timespec="seconds"),
    "duration_s": args.duration,
    "notes": args.notes,
    "timestamps": "device_ts_us and host_ts_us both in the LeapGetNow() microsecond clock",
    "frame_format": "raw_v1: cam_X/frames.bin, per frame left then right uint8 image_size images, "
                    "byte offset in frames.csv",
    "baseline_um": {},
    "calibration": {},  # LeapC factory calibration per camera, read at its first frame
    "frames": {},
}

if not args.sequential:
    record(names, session, args.duration, meta)
else:
    if len(names) < 2:
        raise SystemExit("--sequential needs two or more cameras.")
    isolate = args.isolate
    if isolate == "auto":
        isolate = "pnputil" if ctypes.windll.shell32.IsUserAnAdmin() else "prompt"
    meta["isolate"] = isolate
    print(f"Sequential capture of {names}, {args.duration:g} s each, others off by "
          + ("pnputil (Windows device disable)." if isolate == "pnputil" else "unplugging (you will be asked)."))
    if isolate == "pnputil":
        ids = pnp_ids([cameras[n] for n in names])
        if None in ids.values():
            raise SystemExit(f"No Windows device found for {[n for n in names if ids[cameras[n]] is None]}.")
        try:
            for n in names:
                for m in names:
                    set_enabled(ids[cameras[m]], m == n)
                time.sleep(2)  # let the service drop the disabled devices
                if not record([n], session, args.duration, meta, wait_s=20):
                    break
        finally:
            print("Re-enabling all cameras...")
            for m in names:
                set_enabled(ids[cameras[m]], True)
    else:
        for n in names:
            others = ", ".join(m for m in names if m != n)
            input(f"\nPlug in ONLY {n} (serial ...{cameras[n][14:18]}); unplug {others}. Then press Enter.")
            if not record([n], session, args.duration, meta, wait_s=20):
                break
        print("\nPlug all cameras back in.")

meta["stop"] = datetime.datetime.now().isoformat(timespec="seconds")
if meta["cameras"]:
    with open(os.path.join(session, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done:", session)
