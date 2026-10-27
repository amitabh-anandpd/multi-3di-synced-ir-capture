"""Read frames back from a session folder written by capture_multi.py.

Handles both layouts: raw_v1 (cam_X/frames.bin + offset column) and the older
per-frame PNGs (cam_X/frames/*_L.png, left/right columns).
"""
import csv
import json
import os

import cv2
import numpy as np


def load_session(session):
    return json.load(open(os.path.join(session, "session.json")))


def load_rows(cam_dir):
    with open(os.path.join(cam_dir, "frames.csv")) as f:
        return list(csv.DictReader(f))


def read_pair(cam_dir, row, image_size):
    """(left, right) uint8 images of one frames.csv row."""
    if "offset" in row:
        w, h = image_size
        with open(os.path.join(cam_dir, "frames.bin"), "rb") as f:
            f.seek(int(row["offset"]))
            buf = f.read(2 * w * h)
        return np.frombuffer(buf, np.uint8).reshape(2, h, w)
    return [cv2.imread(os.path.join(cam_dir, "frames", row[k]), cv2.IMREAD_GRAYSCALE) for k in ("left", "right")]
