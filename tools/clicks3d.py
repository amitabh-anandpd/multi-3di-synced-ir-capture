"""Hand-clicked joints (tools/click_joints.py) -> 3D, shared by tools/eval_clicks.py and tools/click_shape.py.

A joint clicked in both cameras at one moment is triangulated in cam_A's rectified-left (unit) frame, the frame the
fused clouds and MANO labels live in. No MANO is needed, so a person's hand can be measured before any fit.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment import LABELS  # noqa: E402
from stereo import rig_path  # noqa: E402

# MCP (thumb: CMC) ... tip in the MANO 21-joint order (fit_mano.FINGERS; "little" = fit_mano's "pinky")
CHAIN = {"thumb": [13, 14, 15, 16], "index": [1, 2, 3, 17], "middle": [4, 5, 6, 18], "ring": [10, 11, 12, 19],
         "little": [7, 8, 9, 20]}
PALM = (1, 7)  # index MCP to little MCP, as fit_mano.py --palm-mm defines palm width
MAX_RAY_GAP = 15.0  # mm: two clicks of one joint are only trusted when their rays nearly meet


def to_camera(P, session, cam):
    """(n, 3) points in cam_A's frame -> that camera's frame, exactly as fuse.py fused the session."""
    if cam == "cam_A":
        return P
    rig = json.load(open(rig_path(session)))
    Rb, tb = np.array(rig["cameras"]["cam_B"]["R"]), np.array(rig["cameras"]["cam_B"]["t_mm"])
    rf = os.path.join(session, "fused", "refine.json")
    R, t = np.eye(3), np.zeros(3)
    if os.path.exists(rf):
        r = json.load(open(rf))
        if r.get("used"):
            R, t = np.array(r["R"]), np.array(r["t_mm"])
    return ((P - t) @ R - tb) @ Rb  # inverse of P_A = R (Rb P_B + tb) + t


def clicked_pairs(keep=lambda session: True):
    """{(session, cam_A frame): {cam: click json}} for every clicked frame whose session passes `keep`."""
    pairs = {}
    for f in glob.glob(os.path.join(LABELS, "joints", "*", "*", "*.json")):
        c = json.load(open(f))
        if keep(c["session"]):
            pairs.setdefault((c["session"], c["cam_A_frame"]), {})[c["cam"]] = c
    return pairs


def triangulate(session, click_a, click_b, unit_a, unit_b):
    """(21, 3) joints in cam_A's frame from one moment clicked in both cameras; NaN where a joint was hidden in
    either view or the two rays miss each other by more than MAX_RAY_GAP. `session` is the sessions/<name> path."""
    # cam_B's centre and its rays in cam_A's frame: to_camera is rigid, P_B = A P_A + c; read A, c off 4 points
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]) * 100
    Bm = to_camera(base, session, "cam_B")
    A = (Bm[1:] - Bm[0]).T / 100  # P_B = A @ P_A + c  ->  P_A = A^T (P_B - c)
    oB = A.T @ (-Bm[0])
    X = np.full((21, 3), np.nan)
    for j in range(21):
        pa, pb = click_a["uv"][j], click_b["uv"][j]
        if pa is None or pb is None:
            continue
        da = np.array([(pa[0] - unit_a.cx) / unit_a.f, (pa[1] - unit_a.cy) / unit_a.f, 1.0])
        db = A.T @ np.array([(pb[0] - unit_b.cx) / unit_b.f, (pb[1] - unit_b.cy) / unit_b.f, 1.0])
        lam = np.linalg.lstsq(np.stack([da, -db], 1), oB, rcond=None)[0]
        qa, qb = lam[0] * da, oB + lam[1] * db
        if np.linalg.norm(qa - qb) < MAX_RAY_GAP and lam[0] > 0 and lam[1] > 0:  # the two rays nearly meet
            X[j] = (qa + qb) / 2
    return X


def finger_length(X, finger):
    """MCP (thumb: CMC) to tip along the bones, or None if any of the 4 joints is missing."""
    ch = CHAIN[finger]
    if not np.isfinite(X[ch]).all():
        return None
    return float(sum(np.linalg.norm(X[a] - X[b]) for a, b in zip(ch, ch[1:])))


def palm_width(X):
    """Index MCP to little MCP, or None if either is missing."""
    return float(np.linalg.norm(X[PALM[0]] - X[PALM[1]])) if np.isfinite(X[list(PALM)]).all() else None
