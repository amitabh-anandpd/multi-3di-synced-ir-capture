"""Shared stereo helpers: LeapC-map rectification, checkerboard corners per 3Di.

Rectified image: a virtual pinhole view of +-slope ray slopes, size x size px, f = size / 2 / slope,
built for each imager from LeapC's 64x64 distortion map (see leapc.distortion). In OpenCV pixel
coordinates its camera matrix is K = [[f, 0, size/2 - 0.5], [0, f, size/2 - 0.5], [0, 0, 1]].
"""
import os

import cv2
import numpy as np

from session_io import load_rows, load_session, read_pair

GRID = (9, 6)  # inner corners of a 10 x 7 squares board


def rectify_maps(dmap, size, slope, raw_wh, pad=0):
    """Pixel maps (map_x, map_y) from a virtual pinhole view of +-slope rays into the raw image, and f.

    pad adds that many columns on the left (SGBM needs room for max_disp)."""
    f = size / 2 / slope
    ax = (np.arange(-pad, size) + 0.5 - size / 2) / f  # ray slope per output pixel
    ay = (np.arange(size) + 0.5 - size / 2) / f
    rx, ry = np.meshgrid(ax, ay)
    gc = ((rx + 4) / 8 * 63).astype(np.float32)  # grid column / row (see leapc.distortion)
    gr = ((4 - ry) / 8 * 63).astype(np.float32)
    nx = cv2.remap(np.ascontiguousarray(dmap[..., 0]), gc, gr, cv2.INTER_LINEAR)
    ny = cv2.remap(np.ascontiguousarray(dmap[..., 1]), gc, gr, cv2.INTER_LINEAR)
    return nx * raw_wh[0] - 0.5, ny * raw_wh[1] + 0.5, f


def dense_path(serial, size, slope, pad=256):
    """Dense LeapC map written by tools/dense_map.py (units/ next to this file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "units", f"{serial}_dense_{size}_{slope}_{pad}.npz")


def pinhole_K(size, slope):
    f = size / 2 / slope
    return np.array([[f, 0, size / 2 - 0.5], [0, f, size / 2 - 0.5], [0, 0, 1]], np.float64)


class Cam:
    """One camera of a session: raw frames -> LeapC-rectified L/R images."""

    def __init__(self, session, name, size, slope, pad=0, use_dense=True):
        self.meta = load_session(session)
        self.name = name
        self.serial = self.meta["cameras"][name]
        self.dir = os.path.join(session, name)
        self.rows = load_rows(self.dir)
        self.by_id = {int(r["frame_id"]): r for r in self.rows}
        self.wh = tuple(self.meta["image_size"])
        cal = self.meta["calibration"][name]
        self.baseline = cal["right"]["extrinsic"][0][3] - cal["left"]["extrinsic"][0][3]
        versions = {r["matrix_version"] for r in self.rows}
        if len(versions) > 1:
            print(f"{name}: distortion map changed during the session {sorted(versions)}, using the first")
        self.matrix_version = self.rows[0]["matrix_version"]
        self.maps = []
        dense = dense_path(self.serial, size, slope)
        if use_dense and os.path.exists(dense):  # exact LeapC mapping (tools/dense_map.py), ~1.2 px better
            dm = np.load(dense)
            full = int(os.path.basename(dense).rsplit("_", 1)[1].split(".")[0])  # pad it was built with
            if pad > full:
                raise ValueError(f"{dense} has {full} pad columns, {pad} asked")
            for side in ("left", "right"):
                self.maps.append((dm[f"{side}_x"][:, full - pad:], dm[f"{side}_y"][:, full - pad:]))
            self.f = size / 2 / slope
            self.dense = True
        else:
            dm = np.load(os.path.join(self.dir, f"distortion_v{self.matrix_version}.npz"))
            for side in ("left", "right"):
                mx, my, self.f = rectify_maps(dm[side], size, slope, self.wh, pad)
                self.maps.append((mx, my))
            self.dense = False
        self.c = size / 2

    def rect(self, frame_id):
        lr = read_pair(self.dir, self.by_id[frame_id], self.wh)
        return [cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0) for img, (mx, my) in zip(lr, self.maps)]


def find(img):
    """Subpixel checkerboard corners (54, 2) or None."""
    # no CALIB_CB_FAST_CHECK: it rejects these dim IR boards
    ok, pts = cv2.findChessboardCorners(img, GRID, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    pts = cv2.cornerSubPix(img, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return pts.reshape(-1, 2)


def board_2d(cam, frame_id, max_row_err=5):
    """Matching corners on the rectified L and R images, ((54, 2), (54, 2)), or None."""
    L, R = cam.rect(frame_id)
    pl = find(L)
    if pl is None:
        return None
    pr = find(R)
    if pr is None:
        return None
    if np.abs(pl[:, 1] - pr[::-1, 1]).mean() < np.abs(pl[:, 1] - pr[:, 1]).mean():
        pr = pr[::-1]  # L and R found the grid in opposite orders
    if np.abs(pl[:, 1] - pr[:, 1]).max() > max_row_err or (pl[:, 0] - pr[:, 0] <= 0).any():
        return None
    return pl, pr


def board_object_points(square):
    """Board corners in the board's own frame, mm, in findChessboardCorners order."""
    g = np.zeros((GRID[0] * GRID[1], 3), np.float32)
    g[:, :2] = np.mgrid[0:GRID[0], 0:GRID[1]].T.reshape(-1, 2) * square
    return g
