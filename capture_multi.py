"""Record raw stereo IR from all (or some) 3Dis at the same time.

  python capture_multi.py --duration 60
  python capture_multi.py --cams cam_A --notes "cam_A alone"

The poll loop only copies frames onto a per-camera queue; one writer thread per
camera saves PNGs, so slow disk writes show up as queue_depth, not dropped frames.
Ctrl+C stops early; queued frames are still written.
"""
import argparse
import csv
import datetime
import json
import os
import queue
import threading
import time
from ctypes import byref

import cv2
import numpy as np

import leapc
from list_devices import CONFIG

VERSION_TXT = r"C:\Program Files\Ultraleap\version.txt"


def writer(cam_dir, q):
    with open(os.path.join(cam_dir, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "device_ts_us", "host_ts_us", "matrix_version", "left", "right", "queue_depth"])
        while True:
            item = q.get()
            if item is None:
                return
            frame_id, dev_ts, host_ts, mver, left, right, depth = item
            names = [f"{frame_id:08d}_L.png", f"{frame_id:08d}_R.png"]
            for name, img in zip(names, (left, right)):
                cv2.imwrite(os.path.join(cam_dir, "frames", name), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            w.writerow([frame_id, dev_ts, host_ts, mver, *names, depth])


ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=10, help="seconds")
ap.add_argument("--cams", nargs="+", help="camera names from config.json (default: all)")
ap.add_argument("--notes", default="")
args = ap.parse_args()

cameras = json.load(open(CONFIG))["cameras"]  # name -> serial
names = args.cams or sorted(cameras)
unknown = set(names) - set(cameras)
if unknown:
    raise SystemExit(f"Not in config.json: {sorted(unknown)}. Run list_devices.py.")
name_of = {cameras[n]: n for n in names}

conn, devices = leapc.connect(serials=list(name_of))
missing = set(name_of) - {d["serial"] for d in devices}
if missing:
    leapc.close(conn, devices)
    raise SystemExit(f"Not connected: {[name_of[s] for s in missing]}")

session = os.path.join("sessions", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
cams = {}  # device id -> state
for d in devices:
    name = name_of[d["serial"]]
    cam_dir = os.path.join(session, name)
    os.makedirs(os.path.join(cam_dir, "frames"))
    q = queue.Queue()
    t = threading.Thread(target=writer, args=(cam_dir, q))
    t.start()
    cams[d["id"]] = {"name": name, "dir": cam_dir, "q": q, "thread": t, "mver": None, "n": 0, "max_q": 0}

meta = {
    "cameras": {name_of[d["serial"]]: d["serial"] for d in devices},
    "sdk_version": open(VERSION_TXT).read().strip() if os.path.exists(VERSION_TXT) else None,
    "start": datetime.datetime.now().isoformat(timespec="seconds"),
    "duration_s": args.duration,
    "notes": args.notes,
    "timestamps": "device_ts_us and host_ts_us both in the LeapGetNow() microsecond clock",
}
print(f"Recording {sorted(meta['cameras'])} for {args.duration:g}s -> {session}  (Ctrl+C to stop)")

msg = leapc.LEAP_CONNECTION_MESSAGE()
t_start = time.monotonic()
t_end, checked = t_start + args.duration, False
try:
    while time.monotonic() < t_end:
        if not checked and time.monotonic() - t_start > 2:
            silent = [c["name"] for c in cams.values() if c["n"] == 0]
            if silent:  # service has the device but no video (seen after a bad replug)
                print(f"ERROR: no frames from {silent} after 2 s. Replug it or restart the "
                      "Ultraleap Tracking Service, then retry.")
                meta["notes"] += f" | ABORTED: no frames from {silent}"
                break
            checked = True
        if leapc.lib.LeapPollConnection(conn, 100, byref(msg)) != leapc.RS_SUCCESS:
            continue
        if msg.type == leapc.EVT_DEVICE_LOST:
            print("WARNING: a device was lost during capture")
        if msg.type != leapc.EVT_IMAGE or msg.device_id not in cams:
            continue
        host_ts = leapc.lib.LeapGetNow()
        c = cams[msg.device_id]
        ev = leapc.image_event(msg)
        left, right = ev.image
        p = left.properties
        mver = left.matrix_version
        if mver != c["mver"]:  # new distortion map: save it (pointer is valid only until the next poll)
            np.savez(os.path.join(c["dir"], f"distortion_v{mver}.npz"),
                     left=leapc.distortion(left), right=leapc.distortion(right))
            c["mver"] = mver
        imgs = [np.frombuffer(leapc.image_bytes(i), np.uint8).reshape(p.height, p.width) for i in (left, right)]
        meta.setdefault("image_size", [p.width, p.height])  # LeapC memory: read it now, not after close
        depth = c["q"].qsize()
        c["max_q"] = max(c["max_q"], depth)
        c["q"].put((ev.info.frame_id, ev.info.timestamp, host_ts, mver, *imgs, depth))
        c["n"] += 1
except KeyboardInterrupt:
    print("Stopped early.")
meta["stop"] = datetime.datetime.now().isoformat(timespec="seconds")
leapc.close(conn, devices)

print("Writing queued frames...")
for c in cams.values():
    c["q"].put(None)
for c in cams.values():
    c["thread"].join()
    print(f"{c['name']}: {c['n']} frames, max queue depth {c['max_q']}")
meta["frames"] = {c["name"]: c["n"] for c in cams.values()}
with open(os.path.join(session, "session.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Done:", session)
