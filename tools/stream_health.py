"""M4: per-camera stream health of one or more sessions, from frames.csv.

  python tools/stream_health.py                       # newest session
  python tools/stream_health.py sessions/A sessions/B # compare runs (e.g. alone vs together)

FPS and jitter come from device timestamps; drops from frame_id gaps.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def health(cam_dir):
    rows = list(csv.DictReader(open(os.path.join(cam_dir, "frames.csv"))))
    ids, dev, host, q = (np.array([int(r[k]) for r in rows], np.int64)
                         for k in ("frame_id", "device_ts_us", "host_ts_us", "queue_depth"))
    gaps, dt = np.diff(ids), np.diff(dev) / 1e3  # dt in ms
    return {
        "frames": len(ids),
        "fps": (len(ids) - 1) / ((dev[-1] - dev[0]) / 1e6),
        "gap_events": int((gaps > 1).sum()),
        "dropped": int((gaps - 1).clip(0).sum()),
        "drop_pct": 100 * (gaps - 1).clip(0).sum() / (ids[-1] - ids[0]),
        "jitter_ms": dt.std(),
        "max_dt_ms": dt.max(),
        "latency_ms": np.median(host - dev) / 1e3,
        "max_queue": int(q.max()),
    }


ap = argparse.ArgumentParser()
ap.add_argument("sessions", nargs="*", help="session folders (default: newest)")
args = ap.parse_args()
sessions = args.sessions or [max(glob.glob(os.path.join("sessions", "20*")))]

print(f"{'session':<20}{'cam':<7}{'frames':>7}{'fps':>7}{'gaps':>6}{'drop':>6}{'drop%':>7}"
      f"{'jit ms':>8}{'max dt':>8}{'lat ms':>8}{'max q':>7}  notes")
for s in sessions:
    meta = json.load(open(os.path.join(s, "session.json")))
    for cam in sorted(meta["cameras"]):
        h = health(os.path.join(s, cam))
        print(f"{os.path.basename(os.path.normpath(s)):<20}{cam:<7}{h['frames']:>7}{h['fps']:>7.1f}"
              f"{h['gap_events']:>6}{h['dropped']:>6}{h['drop_pct']:>7.2f}{h['jitter_ms']:>8.2f}"
              f"{h['max_dt_ms']:>8.1f}{h['latency_ms']:>8.1f}{h['max_queue']:>7}  {meta.get('notes', '')}")
