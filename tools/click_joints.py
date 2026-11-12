"""PLAN.md P2: click the 21 hand joints in IR frames, as ground truth for the MANO labels.

  python tools/click_joints.py                  # 2 frames per camera from every fitted session in dataset.json
  python tools/click_joints.py --per 3
  python tools/click_joints.py --list           # only show which frames would come up

Shows the rectified left IR image of one camera, zoomed on the hand. Click the joints in the order shown at the top:
wrist, then thumb (base, middle knuckle, end knuckle, tip), index, middle, ring, little (knuckle, middle joint, end
joint, tip). Click the CENTRE of the joint as you see it (where the bone bends), and the very end of the fingertip.
The MANO fit is NOT shown, so the clicks don't lean towards it.

  left click        place the current joint (then the next one is asked)
  right click / x   this joint is hidden (behind the hand / out of view): skip it
  z / BACKSPACE     undo the last joint
  SPACE / ENTER     save and go to the next frame (when all 21 are placed or skipped)
  s                 skip this frame (blurry, can't tell)      b   back one frame
  + / -             zoom in / out                              ESC stop (saved frames are kept)

Saves labels/joints/<session>/<cam>/<frame>.json: 21 (u, v) in MANO joint order (null = hidden), pixels of the
800 x 800 rectified left image (segment.py / depth.py), so MANO's joints can be projected onto it exactly.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment import LABELS, Segmenter  # noqa: E402

OUT = os.path.join(LABELS, "joints")
# click order: (name, MANO 21-joint index); fit_mano.FINGERS gives MANO's order
CLICK = [("wrist", 0),
         ("THUMB base", 13), ("THUMB middle knuckle", 14), ("THUMB end knuckle", 15), ("THUMB tip", 16),
         ("INDEX knuckle", 1), ("INDEX middle joint", 2), ("INDEX end joint", 3), ("INDEX tip", 17),
         ("MIDDLE knuckle", 4), ("MIDDLE middle joint", 5), ("MIDDLE end joint", 6), ("MIDDLE tip", 18),
         ("RING knuckle", 10), ("RING middle joint", 11), ("RING end joint", 12), ("RING tip", 19),
         ("LITTLE knuckle", 7), ("LITTLE middle joint", 8), ("LITTLE end joint", 9), ("LITTLE tip", 20)]
COLORS = {"wrist": (255, 255, 255), "THUMB": (0, 0, 255), "INDEX": (0, 200, 0), "MIDDLE": (255, 120, 0),
          "RING": (0, 220, 220), "LITTLE": (220, 0, 220)}

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="*", help="default: every fitted session in dataset.json")
ap.add_argument("--per", type=int, default=2, help="frames per session per camera")
ap.add_argument("--redo", action="store_true", help="also show frames already clicked")
ap.add_argument("--list", action="store_true", help="only list the frames")
args = ap.parse_args()

sessions = args.sessions
if not sessions:
    d = json.load(open("dataset.json"))["sessions"]
    sessions = [s for s, r in d.items() if r.get("fit_frames") and os.path.exists(f"sessions/{s}/mano/smoothed.npz")]
print(f"Loading frames from {len(sessions)} sessions...")
items = []  # (session, cam, frame_id of that camera)
for s in sessions:
    path = os.path.join("sessions", s)
    fitted = np.load(os.path.join(path, "mano", "smoothed.npz"))["frames"].astype(int)
    a2b = {g["frames"]["cam_A"]: g["frames"]["cam_B"] for g in json.load(open(os.path.join(path, "groups.json")))["groups"]}
    for k, frac in enumerate(np.linspace(0.2, 0.8, args.per)):
        fa = int(fitted[int(frac * (len(fitted) - 1))])
        for cam in ("cam_A", "cam_B"):
            fid = fa if cam == "cam_A" else a2b.get(fa)
            if fid is None:
                continue
            if not args.redo and os.path.exists(os.path.join(OUT, s, cam, f"{fid}.json")):
                continue
            items.append((s, cam, int(fid), int(fa)))
print(f"{len(items)} frames to click (about 1 minute each)")
if args.list:
    for it in items:
        print(" ", *it[:3])
    raise SystemExit
if not items:
    raise SystemExit("Nothing left to click.")

segs = {}
state = {"uv": [], "scale": 2.0, "off": (0, 0)}


def on_mouse(ev, x, y, flags, _):
    if len(state["uv"]) >= len(CLICK):
        return
    if ev == cv2.EVENT_LBUTTONDOWN:
        oy, ox = state["off"]
        state["uv"].append((x / state["scale"] + ox, y / state["scale"] + oy))
    elif ev == cv2.EVENT_RBUTTONDOWN:
        state["uv"].append(None)


def view_box(seg, fid):
    """Hand box from the segmentation (or the joints), with context, at least 260 px each way."""
    m, L = seg.mask(fid)
    ys, xs = np.nonzero(m) if m is not None and m.any() else (np.array([300, 500]), np.array([300, 500]))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    hh, hw = max((ys.max() - ys.min()) // 2 + 60, 130), max((xs.max() - xs.min()) // 2 + 60, 130)
    return L, (max(cy - hh, 0), min(cy + hh, L.shape[0]), max(cx - hw, 0), min(cx + hw, L.shape[1]))


win = "click joints"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, on_mouse)
i = 0
while 0 <= i < len(items):
    s, cam, fid, fa = items[i]
    if (s, cam) not in segs:
        segs[(s, cam)] = Segmenter(os.path.join("sessions", s), cam)
    L, (y0, y1, x0, x1) = view_box(segs[(s, cam)], fid)
    out = os.path.join(OUT, s, cam, f"{fid}.json")
    state["uv"] = []
    if os.path.exists(out):  # redo: start from the saved clicks, in click order
        saved = json.load(open(out))["uv"]
        state["uv"] = [tuple(saved[m]) if saved[m] is not None else None for _, m in CLICK]
    state["scale"] = min(900 / (y1 - y0), 1400 / (x1 - x0), 4.0)
    state["off"] = (y0, x0)
    while True:
        crop = cv2.cvtColor(cv2.equalizeHist(L[y0:y1, x0:x1]), cv2.COLOR_GRAY2BGR)
        view = cv2.resize(crop, None, fx=state["scale"], fy=state["scale"], interpolation=cv2.INTER_LINEAR)
        pts = {}
        for k, p in enumerate(state["uv"]):
            if p is None:
                continue
            name = CLICK[k][0]
            col = COLORS[name.split()[0]]
            q = (int((p[0] - x0) * state["scale"]), int((p[1] - y0) * state["scale"]))
            pts[k] = q
            prev = 0 if k in (1, 5, 9, 13, 17) else k - 1  # bones back to the previous joint / the wrist
            if k and prev in pts:
                cv2.line(view, pts[prev], q, col, 1)
            cv2.circle(view, q, 4, col, -1)
        n = len(state["uv"])
        ask = CLICK[n][0] if n < len(CLICK) else "all done: SPACE to save"
        col = COLORS[ask.split()[0]] if n < len(CLICK) else (0, 255, 0)
        cv2.rectangle(view, (0, 0), (view.shape[1], 52), (0, 0, 0), -1)
        cv2.putText(view, f"{n + 1 if n < len(CLICK) else n}/21  click: {ask}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, col, 2)
        cv2.putText(view, f"{i + 1}/{len(items)} {s} {cam} {fid}   right-click/x hidden  z undo  SPACE save  s skip  "
                          "b back  +/- zoom  ESC quit", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            i = -1
            break
        if k in (32, 13) and len(state["uv"]) == len(CLICK):
            uv = [None] * 21
            for (name, m), p in zip(CLICK, state["uv"]):
                uv[m] = [round(float(p[0]), 2), round(float(p[1]), 2)] if p is not None else None
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({"session": s, "cam": cam, "frame": fid, "cam_A_frame": fa, "uv": uv,
                       "image": "rectified left, 800 px (segment.Segmenter / depth.Depth)",
                       "order": "MANO 21: wrist, index 1-3, middle 4-6, little 7-9, ring 10-12, thumb 13-15, "
                                "tips thumb/index/middle/ring/little 16-20"}, open(out, "w"), indent=1)
            i += 1
            break
        if k in (ord("x"),) and len(state["uv"]) < len(CLICK):
            state["uv"].append(None)
        if k in (ord("z"), 8) and state["uv"]:
            state["uv"].pop()
        if k == ord("s"):
            i += 1
            break
        if k == ord("b"):
            i = max(0, i - 1)
            break
        if k in (ord("+"), ord("=")):
            state["scale"] *= 1.2
        if k == ord("-"):
            state["scale"] /= 1.2
cv2.destroyAllWindows()
done = sum(1 for s, cam, fid, _ in items if os.path.exists(os.path.join(OUT, s, cam, f"{fid}.json")))
print(f"Clicked {done} of {len(items)} frames in this run. Run again to continue.")
