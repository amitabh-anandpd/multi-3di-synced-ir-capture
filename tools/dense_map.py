"""N0: dense rectification map per 3Di from LeapC itself, instead of the 64x64 distortion map.

  python tools/dense_map.py            (cameras plugged in; Control Panel closed)

For every pixel of the virtual pinhole view (stereo.rectify_maps: --size, --slope, --pad columns on
the left) asks LeapRectilinearToPixelEx for the raw pixel of that ray, per imager. Writes
units/<serial>_dense_<size>_<slope>_<pad>.npz (left_x, left_y, right_x, right_y, float32) and prints
how far the 64x64-map version is from it. stereo.Cam uses the dense map when the file exists.
"""
import argparse
import os
import sys
import time
from ctypes import byref

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import leapc  # noqa: E402
from stereo import rectify_maps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=800)
ap.add_argument("--slope", type=float, default=1.5)
ap.add_argument("--pad", type=int, default=256, help="extra columns on the left (SGBM max disparity)")
args = ap.parse_args()

conn, devices = leapc.connect()
if not devices:
    raise SystemExit("No 3Di connected.")
# LeapC's mapping is valid only after the device has sent an image; keep one distortion map to compare.
msg, seen = leapc.LEAP_CONNECTION_MESSAGE(), {}
deadline = time.monotonic() + 10
while len(seen) < len(devices) and time.monotonic() < deadline:
    if leapc.lib.LeapPollConnection(conn, 100, byref(msg)) != leapc.RS_SUCCESS or msg.type != leapc.EVT_IMAGE:
        continue
    if msg.device_id not in seen:
        ev = leapc.image_event(msg)
        p = ev.image[0].properties
        seen[msg.device_id] = {"wh": (p.width, p.height), "mver": ev.image[0].matrix_version,
                               "coarse": [leapc.distortion(i).copy() for i in ev.image]}

f = args.size / 2 / args.slope
ax = (np.arange(-args.pad, args.size) + 0.5 - args.size / 2) / f
ay = (np.arange(args.size) + 0.5 - args.size / 2) / f
os.makedirs("units", exist_ok=True)
for d in devices:
    if d["id"] not in seen:
        print(f"{d['serial']}: no image received, skipped")
        continue
    s = seen[d["id"]]
    out = {"matrix_version": s["mver"]}
    t0 = time.monotonic()
    for k, side in ((1, "left"), (2, "right")):  # eLeapPerspectiveType_stereo_left / right
        mx = np.empty((len(ay), len(ax)), np.float32)
        my = np.empty_like(mx)
        for i, y in enumerate(ay):
            for j, x in enumerate(ax):
                v = leapc.lib.LeapRectilinearToPixelEx(conn, d["handle"], k, leapc.LEAP_VECTOR(x, y, 1.0))
                mx[i, j], my[i, j] = v.x, v.y
        out[f"{side}_x"], out[f"{side}_y"] = mx, my
        cx, cy, _ = rectify_maps(s["coarse"][k - 1], args.size, args.slope, s["wh"], args.pad)
        ok = np.isfinite(mx) & (mx >= 0) & (mx < s["wh"][0]) & (my >= 0) & (my < s["wh"][1])
        diff = np.hypot(cx - mx, cy - my)[ok]
        cen = np.zeros_like(ok)
        cen[:, args.pad:][args.size // 4:3 * args.size // 4, args.size // 4:3 * args.size // 4] = True
        print(f"{d['serial']} {side}: 64x64 map vs LeapC, raw px: median {np.median(diff):.2f}, p95 "
              f"{np.percentile(diff, 95):.2f}, centre half median {np.median(np.hypot(cx - mx, cy - my)[ok & cen]):.2f} "
              f"(mean offset x {np.mean((cx - mx)[ok]):+.2f}, y {np.mean((cy - my)[ok]):+.2f})")
    path = os.path.join("units", f"{d['serial']}_dense_{args.size}_{args.slope}_{args.pad}.npz")
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({time.monotonic() - t0:.0f} s)")
leapc.close(conn, devices)
