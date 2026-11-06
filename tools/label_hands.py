"""N4: paint the true hand in sample frames, to score segment.py against.

  python tools/label_hands.py

Shows ~20 frames per camera from the N2 hand sessions (rectified left image, zoomed on the hand).
The starting mask is only "IR-bright near the hand", not segment.py's mask, so the score isn't
biased towards it. Paint the hand up to the wrist crease; leave out the arm, body and background.

  left mouse drag    paint hand (or erase, in erase mode)   right mouse drag   erase
  e                  switch between PAINT and ERASE mode (for touchpads)
  [  /  ]            smaller / bigger brush
  f                  fill holes inside the painted hand
  r                  reset to the starting mask           c   clear everything
  SPACE or ENTER     save and go to the next frame        b   back one frame
  s                  skip this frame (not saved, not scored) when you can't tell what is hand
  ESC                stop (saved frames are kept; run again to continue)

Saves labels/<session>/<cam>/<frame>.png (800 x 800, white = hand).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment import LABELS, Segmenter, drawn_mask  # noqa: E402

N2 = ["2026-11-06_035647", "2026-11-06_035726", "2026-11-06_035752", "2026-11-06_035809", "2026-11-06_035830",
      "2026-11-06_035901", "2026-11-06_035942"]

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="+", default=N2)
ap.add_argument("--per", type=int, default=3, help="frames per session per camera")
ap.add_argument("--redo", action="store_true", help="also show frames already labelled")
args = ap.parse_args()

print("Loading frames...")
items = []  # (session, cam, frame_id, segmenter)
for s in args.sessions:
    path = os.path.join("sessions", s)
    groups = json.load(open(os.path.join(path, "groups.json")))["groups"]
    for cam in ("cam_A", "cam_B"):
        seg = Segmenter(path, cam)
        for frac in np.linspace(0.15, 0.85, args.per):
            pos = int(len(groups) * frac)
            # step forward to the first frame with a hand (at most ~1 s)
            fid = next((groups[j]["frames"][cam] for j in range(pos, min(pos + 90, len(groups)))
                        if seg.joints(groups[j]["frames"][cam]) is not None), None)
            if fid is None:
                continue
            if not args.redo and os.path.exists(os.path.join(LABELS, s, cam, f"{fid}.png")):
                continue
            items.append((s, cam, fid, seg))
print(f"{len(items)} frames to label")
if not items:
    raise SystemExit("Nothing left to label.")

state = {"brush": 8, "down": 0, "mask": None, "scale": 1.0, "off": (0, 0), "erase": False}


def start_mask(seg, fid, L):
    J, wrist, _ = seg.joints(fid)
    d = drawn_mask(seg.unit, J, wrist, L.shape[0])
    ys, xs = np.nonzero(d)
    box = np.zeros_like(d)
    box[max(ys.min() - 30, 0):ys.max() + 31, max(xs.min() - 30, 0):xs.max() + 31] = True
    # view: the hand box plus context, at least 260 px each way so a side-on hand isn't blown up
    cy, cx, hh, hw = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, (ys.max() - ys.min()) // 2 + 90, (xs.max() - xs.min()) // 2 + 90
    hh, hw = max(hh, 130), max(hw, 130)
    return ((L >= 45) & box).astype(np.uint8), (max(cy - hh, 0), min(cy + hh, 800), max(cx - hw, 0), min(cx + hw, 800))


def on_mouse(ev, x, y, flags, _):
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        right = ev == cv2.EVENT_RBUTTONDOWN
        state["down"] = 2 if (right or state["erase"]) else 1  # e toggles erase for the left button / touchpad
    elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state["down"] = 0
    if state["down"] and ev in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        oy, ox = state["off"]
        px, py = int(x / state["scale"]) + ox, int(y / state["scale"]) + oy
        cv2.circle(state["mask"], (px, py), state["brush"], 1 if state["down"] == 1 else 0, -1)


win = "label hands"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, on_mouse)
i = 0
while 0 <= i < len(items):
    s, cam, fid, seg = items[i]
    _, L = seg.mask(fid)
    out = os.path.join(LABELS, s, cam, f"{fid}.png")
    m0, (y0, y1, x0, x1) = start_mask(seg, fid, L)
    state["mask"] = (cv2.imread(out, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if os.path.exists(out) else m0.copy()
    state["scale"] = min(900 / (y1 - y0), 1400 / (x1 - x0), 3.0)
    state["off"] = (y0, x0)
    while True:
        crop = cv2.cvtColor(L[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        mk = state["mask"][y0:y1, x0:x1] > 0
        crop[mk] = (0.55 * crop[mk] + [0, 110, 0]).astype(np.uint8)
        view = cv2.resize(crop, None, fx=state["scale"], fy=state["scale"], interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(cv2.resize(mk.astype(np.uint8), (view.shape[1], view.shape[0]),
                                             interpolation=cv2.INTER_NEAREST), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(view, cnt, -1, (0, 255, 0), 1)
        mode = "ERASE" if state["erase"] else "PAINT"
        cv2.putText(view, f"MODE: {mode}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if state["erase"] else (0, 255, 0), 2)
        cv2.putText(view, f"{i + 1}/{len(items)}  {s} {cam} frame {fid}  brush {state['brush']}  "
                          "  e: mode  [ ] brush  f fill  r reset  c clear  SPACE save  s skip  b back  ESC quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            i = -1
            break
        if k in (32, 13):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cv2.imwrite(out, state["mask"] * 255)
            i += 1
            break
        if k == ord("b"):
            i = max(0, i - 1)
            break
        if k == ord("s"):
            i += 1
            break
        if k == ord("e"):
            state["erase"] = not state["erase"]
        if k == ord("["):
            state["brush"] = max(1, state["brush"] - 1)
        if k == ord("]"):
            state["brush"] += 1
        if k == ord("r"):
            state["mask"] = m0.copy()
        if k == ord("c"):
            state["mask"][:] = 0
        if k == ord("f"):
            ff = state["mask"].copy() * 255
            h, w = ff.shape
            fl = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, fl, (0, 0), 128)
            state["mask"][ff == 0] = 1
cv2.destroyAllWindows()
done = sum(1 for s, cam, fid, _ in items if os.path.exists(os.path.join(LABELS, s, cam, f"{fid}.png")))
print(f"Labelled {done} of {len(items)} frames in this run. Run again to continue; segment.py --eval scores them.")
