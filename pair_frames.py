"""M5: group frames across cameras by timestamp.

  python pair_frames.py                      # newest session, reference = first camera
  python pair_frames.py sessions/X --ref cam_B

For each reference frame, takes the nearest-timestamp frame of every other camera
(device_ts, all in the LeapGetNow() clock). A group is kept only if every gap is within
--max-gap periods (default half a frame period) and no frame is used twice.
Writes <session>/groups.json and prints the gap distribution.
"""
import argparse
import glob
import json
import os

import numpy as np

from session_io import load_rows, load_session

ap = argparse.ArgumentParser()
ap.add_argument("session", nargs="?", help="session folder (default: newest)")
ap.add_argument("--ref", help="reference camera (default: first in session)")
ap.add_argument("--max-gap", type=float, default=0.5, help="max |gap| as a fraction of the frame period")
args = ap.parse_args()
session = args.session or max(glob.glob(os.path.join("sessions", "20*")))

meta = load_session(session)
cams = sorted(meta["cameras"])
ref = args.ref or cams[0]
others = [c for c in cams if c != ref]
if not others:
    raise SystemExit(f"{session} has only {ref}; pairing needs two or more cameras.")
ts = {c: np.array([int(r["device_ts_us"]) for r in load_rows(os.path.join(session, c))], np.int64) for c in cams}
ids = {c: np.array([int(r["frame_id"]) for r in load_rows(os.path.join(session, c))], np.int64) for c in cams}

period = np.median(np.diff(ts[ref]))
max_gap = args.max_gap * period
keep = np.ones(len(ts[ref]), bool)
match, gap = {}, {}
for c in others:
    j = np.clip(np.searchsorted(ts[c], ts[ref]), 1, len(ts[c]) - 1)
    j -= (ts[ref] - ts[c][j - 1]) < (ts[c][j] - ts[ref])  # nearest of the two neighbours
    g = ts[c][j] - ts[ref]
    ok = np.abs(g) <= max_gap
    # one frame of c may be nearest to two reference frames (after a drop): keep the closer one
    order = np.argsort(np.abs(g))
    first = np.zeros(len(j), bool)
    _, uniq = np.unique(j[order], return_index=True)
    first[order[uniq]] = True
    keep &= ok & first
    match[c], gap[c] = j, g

groups = [{"t_us": int(ts[ref][i]), "frames": {ref: int(ids[ref][i]), **{c: int(ids[c][match[c][i]]) for c in others}},
           "gap_us": {c: int(gap[c][i]) for c in others}} for i in np.flatnonzero(keep)]
with open(os.path.join(session, "groups.json"), "w") as f:
    json.dump({"reference": ref, "period_us": float(period), "max_gap_us": float(max_gap),
               "frames": {c: len(ts[c]) for c in cams}, "groups": groups}, f)

print(f"{session}: reference {ref}, period {period / 1e3:.2f} ms, max gap {max_gap / 1e3:.2f} ms")
print(f"groups kept: {keep.sum()} of {len(keep)} reference frames ({100 * keep.mean():.1f} %)")
t = (ts[ref] - ts[ref][0]) / 1e6
for c in others:
    g = gap[c][keep] / 1e3
    a = np.abs(g)
    drift = np.polyfit(t[keep], g, 1)[0]  # ms per second: clocks/frame rates not locked if nonzero
    print(f"\n{c} - {ref} gap (ms): median {np.median(g):+.2f}, |gap| mean {a.mean():.2f}, p95 {np.percentile(a, 95):.2f}, "
          f"max {a.max():.2f}, std {g.std():.2f}, drift {drift:+.3f} ms/s")
    edges = np.linspace(-max_gap / 1e3, max_gap / 1e3, 13)
    hist, _ = np.histogram(g, edges)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        print(f"  {lo:+6.2f} .. {hi:+6.2f}  {n:6d}  {'#' * int(50 * n / max(hist.max(), 1))}")
print(f"\nwrote {os.path.join(session, 'groups.json')}")
