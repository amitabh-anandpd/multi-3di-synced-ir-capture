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
        # the sector-based detector finds boards seen at a steep angle or partly blurred (2026-11-11: cam_B 56 of 76
        # frames vs 11); its corners are already subpixel-accurate
        ok, pts = cv2.findChessboardCornersSB(img, GRID, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        return pts.reshape(-1, 2) if ok else None
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


def unit_path(serial):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "units", f"{serial}.json")


class Unit:
    """One 3Di with its N0 stereo calibration (units/<serial>.json, calib_unit.py).

    Works on the LeapC pinhole images of stereo.Cam. Its own frame is the rectified left imager:
    x right, y down, z forward, mm; disparity d gives z = f * baseline / d.
    """

    def __init__(self, serial):
        import json
        u = json.load(open(unit_path(serial)))
        self.serial, self.size, self.slope = serial, u["size"], u["slope"]
        self.K1, self.K2 = np.array(u["K_left"]), np.array(u["K_right"])
        self.D1 = np.array(u.get("D_left", np.zeros(5)))
        self.D2 = np.array(u.get("D_right", np.zeros(5)))
        R, T = np.array(u["R"]), np.array(u["T_mm"]).reshape(3, 1)
        self.R1, self.R2, self.P1, self.P2, *_ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (self.size, self.size), R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
        self.f, self.cx, self.cy = self.P1[0, 0], self.P1[0, 2], self.P1[1, 2]
        self.baseline = -self.P2[0, 3] / self.P2[0, 0]

    def triangulate(self, pl, pr):
        """Corners (n, 2) on the LeapC pinhole L and R images -> (n, 3) mm in this unit's frame."""
        ul = cv2.undistortPoints(pl.reshape(-1, 1, 2).astype(np.float64), self.K1, self.D1, R=self.R1, P=self.P1)
        ur = cv2.undistortPoints(pr.reshape(-1, 1, 2).astype(np.float64), self.K2, self.D2, R=self.R2, P=self.P2)
        ul, ur = ul.reshape(-1, 2), ur.reshape(-1, 2)
        z = self.f * self.baseline / (ul[:, 0] - ur[:, 0])
        return np.stack([(ul[:, 0] - self.cx) / self.f * z, ((ul[:, 1] + ur[:, 1]) / 2 - self.cy) / self.f * z, z], 1)

    def raw_maps(self, cam, pad):
        """Raw-image pixel maps for this unit's rectified L and R views, with `pad` columns on the left.

        cam: stereo.Cam built with the same size / slope and pad >= 0 (its maps start at pinhole x = -cam_pad)."""
        cam_pad = cam.maps[0][0].shape[1] - self.size
        out = []
        for K, D, Rr, P, (mx, my) in ((self.K1, self.D1, self.R1, self.P1, cam.maps[0]),
                                      (self.K2, self.D2, self.R2, self.P2, cam.maps[1])):
            Pp = P.copy()
            Pp[0, 2] += pad  # padded output: column j shows rectified x = j - pad
            px, py = cv2.initUndistortRectifyMap(K, D, Rr, Pp[:, :3], (self.size + pad, self.size), cv2.CV_32FC1)
            out.append((cv2.remap(mx, px + cam_pad, py, cv2.INTER_LINEAR, borderValue=-1),
                        cv2.remap(my, px + cam_pad, py, cv2.INTER_LINEAR, borderValue=-1)))
        return out

    def cloud(self, cam, frame_ids, max_disp=256, zmin=150, zmax=600, min_gray=40, min_hits=0.3, block=5):
        """Median-of-frames SGBM point cloud (n, 3) mm in this unit's frame, and the gray value of each point."""
        maps = self.raw_maps(cam, max_disp)
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=block, P1=8 * block ** 2,
                                     P2=32 * block ** 2, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disps = []
        for i in frame_ids:
            raw = read_pair(cam.dir, cam.by_id[i], cam.wh)
            L, R = [cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0) for img, (mx, my) in zip(raw, maps)]
            d = sgbm.compute(L, R).astype(np.float32)[:, max_disp:] / 16
            d[d <= 0] = np.nan
            disps.append(d)
            ref = L[:, max_disp:]
        stack = np.stack(disps)
        keep = np.isfinite(stack).mean(0) >= min_hits
        disp = np.zeros_like(disps[0])
        disp[keep] = np.nanmedian(stack[:, keep], axis=0)
        v, u = np.indices(disp.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.f * self.baseline / disp
        ok = (disp > 0) & (z > zmin) & (z < zmax) & (ref >= min_gray)
        pts = np.stack([(u - self.cx) / self.f * z, (v - self.cy) / self.f * z, z], -1)[ok]
        return pts, ref[ok]


ROOT = os.path.dirname(os.path.abspath(__file__))


def rig_path(session=None):
    """The rig a session was recorded with: <session>/rig.json when pinned (process.py pins the current one before
    fusing; tools/set_rig.py assigns another; board sessions hold their own calibration), else ./rig.json."""
    if session is not None:
        p = os.path.join(session, "rig.json")
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "rig.json")


def pin_rig(session, src=None):
    """Copy a rig (default the current ./rig.json) into the session, keeping the one it had as rig_prev.json."""
    import json
    import shutil
    src = src or os.path.join(ROOT, "rig.json")
    dst = os.path.join(session, "rig.json")
    if os.path.exists(dst):
        if os.path.abspath(dst) == os.path.abspath(src):
            return dst
        shutil.copy(dst, os.path.join(session, "rig_prev.json"))
    shutil.copy(src, dst)
    return json.load(open(dst)).get("board_session")


def kabsch(src, dst):
    """R, t minimising |R @ src + t - dst|."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    s = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    R = vt.T @ s @ u.T
    return R, cd - R @ cs


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def write_ply(path, xyz, rgb):
    xyz = xyz * [1, -1, -1]  # flip y / z so viewers show it upright (as depth_preview.py)
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(fh, np.hstack([xyz, rgb]), fmt="%.2f %.2f %.2f %d %d %d")
