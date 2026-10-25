"""M3b throwaway: one frame of one camera -> stereo depth -> .ply point cloud.

  python depth_preview.py sessions/<session> --cam cam_A [--frame N]

Undistorts both fisheye views into one shared virtual pinhole camera using the
saved distortion map. The two imagers are parallel, offset only along x
(extrinsics: +-baseline/2), so the pair comes out rectified. Then StereoSGBM ->
depth = f * baseline / disparity.
Writes <session>/<cam>/preview/ : rectified L/R, disparity PNG, points.ply.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np


def rectify_maps(dmap, size, slope, raw_wh):
    """Pixel maps (map_x, map_y) from a virtual pinhole view of +-slope rays into the raw image."""
    f = size / 2 / slope
    a = (np.arange(size) + 0.5 - size / 2) / f  # ray slope per output pixel
    rx, ry = np.meshgrid(a, a)
    gc = ((rx + 4) / 8 * 63).astype(np.float32)  # grid column / row (see leapc.distortion)
    gr = ((4 - ry) / 8 * 63).astype(np.float32)
    nx = cv2.remap(np.ascontiguousarray(dmap[..., 0]), gc, gr, cv2.INTER_LINEAR)
    ny = cv2.remap(np.ascontiguousarray(dmap[..., 1]), gc, gr, cv2.INTER_LINEAR)
    return nx * raw_wh[0] - 0.5, ny * raw_wh[1] + 0.5, f


def write_ply(path, xyz, gray):
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n"
                 f"element vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), g in zip(xyz, gray):
            fh.write(f"{x:.2f} {y:.2f} {z:.2f} {g} {g} {g}\n")


ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--cam", default="cam_A")
ap.add_argument("--frame", type=int, help="row in frames.csv (default: middle)")
ap.add_argument("--size", type=int, default=400, help="rectified image size, px")
ap.add_argument("--slope", type=float, default=1.0, help="half field of view as ray slope (1.0 = +-45 deg)")
ap.add_argument("--max-disp", type=int, default=128, help="multiple of 16")
ap.add_argument("--zmin", type=float, default=80, help="mm")
ap.add_argument("--zmax", type=float, default=1000, help="mm")
ap.add_argument("--min-gray", type=int, default=25, help="skip pixels darker than this (not lit by the IR LEDs)")
args = ap.parse_args()

cam_dir = os.path.join(args.session, args.cam)
meta = json.load(open(os.path.join(args.session, "session.json")))
rows = list(csv.DictReader(open(os.path.join(cam_dir, "frames.csv"))))
row = rows[len(rows) // 2 if args.frame is None else args.frame]
if args.cam not in meta.get("calibration", {}):
    raise SystemExit("session.json has no calibration (recorded before 2026-10-26). Capture a new session.")
cal = meta["calibration"][args.cam]
baseline_mm = cal["right"]["extrinsic"][0][3] - cal["left"]["extrinsic"][0][3]  # device-info baseline is rounded to mm
dmaps = np.load(os.path.join(cam_dir, f"distortion_v{row['matrix_version']}.npz"))
raw = [cv2.imread(os.path.join(cam_dir, "frames", row[k]), cv2.IMREAD_GRAYSCALE) for k in ("left", "right")]
wh = raw[0].shape[::-1]

rect = []
for img, side in zip(raw, ("left", "right")):
    mx, my, f = rectify_maps(dmaps[side], args.size, args.slope, wh)
    rect.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0))

sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=args.max_disp, blockSize=5,
                             P1=8 * 25, P2=32 * 25, uniquenessRatio=10, speckleWindowSize=100,
                             speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sgbm.compute(rect[0], rect[1]).astype(np.float32) / 16

c = args.size / 2
v, u = np.indices(disp.shape)
with np.errstate(divide="ignore", invalid="ignore"):
    z = f * baseline_mm / disp
ok = (disp > 0) & (z > args.zmin) & (z < args.zmax) & (rect[0] >= args.min_gray)
x = (u + 0.5 - c) / f * z
y = (v + 0.5 - c) / f * z
xyz = np.stack([x[ok], -y[ok], -z[ok]], axis=1)  # flip y/z so viewers show it upright, facing the camera

out = os.path.join(cam_dir, "preview")
os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, "rect_L.png"), rect[0])
cv2.imwrite(os.path.join(out, "rect_R.png"), rect[1])
pair = np.hstack(rect)
pair[::40] = 255  # epipolar check: a feature should sit on the same row in both halves
cv2.imwrite(os.path.join(out, "rect_pair_lines.png"), pair)
dvis = cv2.applyColorMap(np.clip(disp / args.max_disp * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
dvis[disp <= 0] = 0
cv2.imwrite(os.path.join(out, "disparity.png"), dvis)
write_ply(os.path.join(out, "points.ply"), xyz, rect[0][ok])

print(f"frame {row['frame_id']}, baseline {baseline_mm:.2f} mm, virtual f {f:.1f} px")
print(f"valid disparity {ok.mean() * 100:.1f}% of pixels, {ok.sum()} points, "
      f"depth median {np.median(z[ok]):.0f} mm (range {z[ok].min():.0f}-{z[ok].max():.0f})" if ok.any() else "no valid points")
print("wrote", out)
