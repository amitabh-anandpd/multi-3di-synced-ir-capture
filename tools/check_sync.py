"""M6: physical sync check. Record a sharp event (hand slapping the desk, a clap) seen by
both cameras, run pair_frames.py, then:

  python tools/check_sync.py                  # newest session
  python tools/check_sync.py sessions/X --k 8

Per camera: motion energy = mean |frame - previous frame| in the most-changed cell of a
12x12 grid (left image, 1/4 size). Both cameras must see the event, close up (~30 cm).
Events = onsets in the reference camera (first frame of a run above median + k * MAD).
Around each event (+-half_window s) the two motion signals, aligned by groups.json, are
cross-correlated: best lag of 0 or +-1 group with r >= --min-r passes. Onset times alone are
not compared: each camera sees the hand enter its view at a different moment.
Also cross-correlates the whole motion signals.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_io import load_rows, load_session, read_pair  # noqa: E402


def motion(cam_dir, image_size):
    rows = load_rows(cam_dir)
    ids = np.array([int(r["frame_id"]) for r in rows], np.int64)
    ts = np.array([int(r["device_ts_us"]) for r in rows], np.int64)
    e, prev = np.zeros(len(rows)), None
    for i, r in enumerate(rows):
        img = read_pair(cam_dir, r, image_size)[0][::4, ::4].astype(np.int16)
        if prev is not None:  # strongest change in any of 12x12 cells, so a local hand motion stands out
            d = np.abs(img - prev)
            h, w = (d.shape[0] // 12) * 12, (d.shape[1] // 12) * 12
            e[i] = d[:h, :w].reshape(12, h // 12, 12, w // 12).mean(axis=(1, 3)).max()
        prev = img
    return ids, ts, e


def onsets(e, ts, k, gap_us=500_000):
    med = np.median(e)
    thr = med + k * max(np.median(np.abs(e - med)), 1e-3)
    above = e > thr
    out = []
    for i in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
        if not out or ts[i] - ts[out[-1]] > gap_us:  # one onset per event
            out.append(i)
    return np.array(out, int), thr


def best_lag(a, b, max_lag=8):
    """Lag (in samples) maximising the normalised correlation of b against a; positive = b later."""
    a, b = a - a.mean(), b - b.mean()
    lags = range(-max_lag, max_lag + 1)
    r = [np.sum(a[max(0, -L):len(a) - max(0, L)] * b[max(0, L):len(b) - max(0, -L)])
         / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) for L in lags]
    return list(lags)[int(np.argmax(r))], max(r), r[max_lag]


ap = argparse.ArgumentParser()
ap.add_argument("session", nargs="?", help="session folder (default: newest)")
ap.add_argument("--k", type=float, default=8, help="threshold = median + k * MAD of motion energy")
ap.add_argument("--half-window", type=float, default=0.75, help="s of motion either side of an event")
ap.add_argument("--min-r", type=float, default=0.5, help="min correlation for an event to count")
args = ap.parse_args()
session = args.session or max(glob.glob(os.path.join("sessions", "20*")))

meta = load_session(session)
gpath = os.path.join(session, "groups.json")
if not os.path.exists(gpath):
    raise SystemExit(f"No groups.json in {session}. Run: python pair_frames.py {session}")
G = json.load(open(gpath))
ref = G["reference"]
cams = sorted(meta["cameras"])
group_of = {c: {g["frames"][c]: n for n, g in enumerate(G["groups"])} for c in cams}  # frame_id -> group

sig = {c: motion(os.path.join(session, c), meta["image_size"]) for c in cams}
ev = {}
for c in cams:
    ids, ts, e = sig[c]
    ev[c], thr = onsets(e, ts, args.k)
    print(f"{c}: {len(ev[c])} event(s), threshold {thr:.2f}, onsets at "
          + ", ".join(f"{(ts[i] - ts[0]) / 1e6:.2f}s" for i in ev[c]))

period_ms = G["period_us"] / 1e3
half = int(round(args.half_window * 1e3 / period_ms))  # groups either side of an event
for c in (c for c in cams if c != ref):
    # both motion signals on the common group timeline
    row = {c2: {int(f): n for n, f in enumerate(sig[c2][0])} for c2 in (ref, c)}
    a = np.array([sig[ref][2][row[ref][g["frames"][ref]]] for g in G["groups"]])
    b = np.array([sig[c][2][row[c][g["frames"][c]]] for g in G["groups"]])

    print(f"\n{c} vs {ref}  (1 group = {period_ms:.2f} ms, window +-{args.half_window:g} s)")
    print(f"{'t (s)':>8}{'lag':>6}{'r':>7}  result  (lag in groups; + = {c} later)")
    ok_all = []
    for i in ev[ref]:
        t_s = (sig[ref][1][i] - sig[ref][1][0]) / 1e6
        g0 = group_of[ref].get(int(sig[ref][0][i]))
        if g0 is None:  # onset frame was dropped from pairing: use the nearest group
            g0 = int(np.argmin(np.abs(np.array([g["t_us"] for g in G["groups"]]) - sig[ref][1][i])))
        lo, hi = max(0, g0 - half), min(len(a), g0 + half + 1)
        lag, r, _ = best_lag(a[lo:hi], b[lo:hi])
        if r < args.min_r:
            print(f"{t_s:8.2f}{lag:+6d}{r:7.2f}  unclear (motion not shared by both views)")
            continue
        ok = abs(lag) <= 1
        ok_all.append(ok)
        print(f"{t_s:8.2f}{lag:+6d}{r:7.2f}  {'PASS' if ok else 'FAIL'}")
    if ok_all:
        print(f"events passing (same or adjacent group): {sum(ok_all)}/{len(ok_all)}")

    lag, r, r0 = best_lag(a, b, 5)
    print(f"whole-session motion cross-correlation: best lag {lag:+d} group(s), r = {r:.2f} (lag 0: {r0:.2f})")
