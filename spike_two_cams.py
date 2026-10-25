"""M1 spike: do all connected 3Dis stream IR images at the same time?

  python spike_two_cams.py                    # all devices together
  python spike_two_cams.py --serial <SERIAL>  # one device alone, for comparison

Prints per-device FPS, frame_id gaps, device-ts jitter and service latency.
Saves the first L/R frame of each device to sessions/spike/ for a visual check.
"""
import argparse
import ctypes
import os
import time

import cv2
import numpy as np

import leapc

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=10)
ap.add_argument("--serial", help="enable images on this device only")
args = ap.parse_args()

conn = leapc.connect()
devices = leapc.open_devices(conn)
print(f"{len(devices)} device(s):", ", ".join(f"{d['serial']} (id {d['id']}, pid 0x{d['pid']:X})" for d in devices))
if not devices:
    raise SystemExit("No devices found.")

by_id = {}
for d in devices:
    if args.serial and d["serial"] != args.serial:
        continue
    want = leapc.POLICY_BACKGROUND_FRAMES | leapc.POLICY_IMAGES
    leapc.check(leapc.lib.LeapSetPolicyFlagsEx(conn, d["handle"], want, 0), "LeapSetPolicyFlagsEx")
    d.update(frame_ids=[], dev_ts=[], host_ts=[], saved=False, policy=0)
    by_id[d["id"]] = d

os.makedirs("sessions/spike", exist_ok=True)
msg = leapc.LEAP_CONNECTION_MESSAGE()
other = {}
t_end = time.monotonic() + args.seconds
while time.monotonic() < t_end:
    if leapc.lib.LeapPollConnection(conn, 100, ctypes.byref(msg)) != leapc.RS_SUCCESS:
        continue
    if msg.type == leapc.EVT_POLICY and msg.device_id in by_id:
        by_id[msg.device_id]["policy"] = leapc.policy_event_flags(msg)
    if msg.type != leapc.EVT_IMAGE:
        other[msg.type] = other.get(msg.type, 0) + 1
        continue
    host_ts = leapc.lib.LeapGetNow()
    d = by_id.get(msg.device_id)
    if d is None:
        other["image_from_unknown_device_%d" % msg.device_id] = other.get("image_from_unknown_device_%d" % msg.device_id, 0) + 1
        continue
    ev = leapc.image_event(msg)
    d["frame_ids"].append(ev.info.frame_id)
    d["dev_ts"].append(ev.info.timestamp)
    d["host_ts"].append(host_ts)
    if not d["saved"]:
        for side, img in zip("LR", ev.image):
            p = img.properties
            arr = np.frombuffer(leapc.image_bytes(img), np.uint8).reshape(p.height, p.width * p.bpp)
            cv2.imwrite(f"sessions/spike/{d['serial']}_{side}.png", arr)
        p = ev.image[0].properties
        print(f"{d['serial']}: first image {p.width}x{p.height} bpp={p.bpp} format=0x{p.format:X} "
              f"matrix_version={ev.image[0].matrix_version}")
        d["saved"] = True

print(f"\n{'serial':<36}{'frames':>8}{'fps':>8}{'id gaps':>9}{'dropped':>9}{'jitter ms':>11}{'latency ms':>12}")
for d in by_id.values():
    if len(d["frame_ids"]) < 2:
        print(f"{d['serial']:<36}{len(d['frame_ids']):>8}   <- NO STREAM")
        continue
    ids, dev, host = (np.array(d[k], np.int64) for k in ("frame_ids", "dev_ts", "host_ts"))
    gaps = np.diff(ids)
    fps = (len(ids) - 1) / ((dev[-1] - dev[0]) / 1e6)
    jitter = np.diff(dev).std() / 1e3
    latency = np.median(host - dev) / 1e3
    print(f"{d['serial']:<36}{len(ids):>8}{fps:>8.1f}{int((gaps > 1).sum()):>9}{int((gaps - 1).clip(0).sum()):>9}"
          f"{jitter:>11.2f}{latency:>12.2f}")
print("other events:", other)
for d in by_id.values():
    if not d["policy"] & leapc.POLICY_BACKGROUND_FRAMES:
        print(f"WARNING {d['serial']}: background frames refused -> Control Panel > enable 'Allow Background Apps'")
    if not d["policy"] & leapc.POLICY_IMAGES:
        print(f"WARNING {d['serial']}: images policy refused -> Control Panel > enable 'Allow Images'")

leapc.close(conn, devices)
