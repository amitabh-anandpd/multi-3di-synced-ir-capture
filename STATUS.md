# Status

## Works
- `leapc.py`: ctypes LeapC access, multi-device, no compiled bindings (SDK 6.2.0).
- `spike_two_cams.py`: both 3Dis stream IR images at the same time.
- `list_devices.py`: stable names (cam_A = …6195, cam_B = …6230). Survived swapping the USB ports.
- `capture_multi.py`: simultaneous capture of both cameras into a session folder.

## M3 capture (2026-10-26, 60 s, both cameras, one Intel USB 3.2 controller, no hub)

| camera | frames | FPS | dropped | max writer queue | latency |
|---|---|---|---|---|---|
| cam_A | 5383 | 89.7 | 0 | 3 | 8.0 ms |
| cam_B | 5383 | 89.7 | 0 | 5 | 8.0 ms |

Disk: ~0.85 GB per camera per minute (PNG level 1).

## M3b depth preview (2026-10-26)
- `depth_preview.py` works end to end: distortion map -> rectified 400×400 virtual pinhole pair
  (±45°, f = 200 px) -> SGBM -> `.ply`. Rows line up in the rectified pair.
- The distortion map layout was verified against LeapC's `LeapRectilinearToPixelEx`: 0.6 px median
  error (see `leapc.distortion`).
- Calibration per unit (now saved in session.json): imagers parallel, offset only in x.
  Baselines from the extrinsics: cam_A 63.86 mm, cam_B 65.75 mm. Device-info baselines are rounded (63 / 65 mm).
- LeapC's `K` and `D` don't behave as a pixel-space OpenCV model (f ≈ 1.28, c ≈ −2). They're saved but not used.
- First desk scene: an arm at ~31 cm came out as a clean surface. Dark or far areas have no IR return, so no depth.
- Hand (cam_B, `sessions/2026-10-26_032559`, `--zmin 150 --zmax 600 --min-gray 40` + Open3D statistical
  outlier removal): palm, four fingers and thumb come out as one dense, clean surface. Skin under the IR
  LEDs matches well; smooth white plastic and the plain desk match poorly (9–16 % of pixels valid per frame).
- Limits seen: nothing closer than ~103 mm (128 px max disparity), the left 128 px columns never get depth
  (SGBM), and far/dark areas have no IR return.
- Size check on the hand: ruler 150 mm, cloud 165.5 mm (CloudCompare point picking, units are mm) -> +10 %.
  Lateral size is X = (u - c) * baseline / disparity, so the virtual f cancels; the error comes from
  disparity or point picking, not the rectification focal length. Likely cause: SGBM edge fattening
  (5 px block, ~2-3 px per edge, ~2 mm/px at 400 mm) on an edge-to-edge measurement.
  No global scale correction applied. To confirm: a rigid flat object at ~30 cm, measured between marks
  inside its face rather than at its silhouette.

## Measured
2026-10-26, SDK 6.2.0, 2 × 3Di, 10 s runs:

| run | camera | FPS | frame_id gaps | ts jitter | service→host latency |
|---|---|---|---|---|---|
| together | …6195 (id 1) | 89.3 | 7 | 1.03 ms | 9.5 ms |
| together | …6230 (id 2) | 89.3 | 7 | 1.05 ms | 10.0 ms |
| alone | …6195 | 89.3 | 7 | 1.01 ms | 8.4 ms |
| alone | …6230 | 89.3 | 7 | 0.98 ms | 8.3 ms |

Images: 384×384 per eye, 1 byte/pixel, IR format.

## Open issues
- A bad USB connection made cam_B silent. Control Panel warning: "USB hub or low powered port".
  Service log: "Broken device detected, missing frame data". Fixed by reseating the cable and restarting.
  capture_multi.py now aborts if a camera sends nothing in 2 s.
- Disk use (~1.7 GB/min for 2 cams) limits long sessions. Fine for bursts.
- The 64×64 distortion map is coarse (0.6 px error). For phase 2 depth, consider saving a dense
  `LeapRectilinearToPixelEx` table at capture time, as the old repo's `rect_lut.npz` did.
- Sessions recorded before 2026-10-26 00:53 have no calibration in session.json.

## Resolved
- 7 single-frame ID gaps per 10 s in M1: gone in the M3 run (0 drops). Probably the same failing USB link.

## Log
- 2026-10-26: M1 passed. Root cause of the earlier zero-frame result: devices have to be opened
  from device events inside the poll loop (subscribe, then set the policy). Enumerating with
  LeapGetDeviceList after polling gave no frames. No Control Panel "background apps" setting is needed.
- 2026-10-26: M2 passed. Names stayed stable after unplugging and swapping the ports.
- 2026-10-26: M3 passed after cam_B's USB fault was fixed (reseat + restart). 60 s, 0 drops on both cameras.
- 2026-10-26: M3b pipeline working. Known-object size check pending.
- 2026-10-26: First clean hand point cloud from cam_B. cam_A not detected in this sitting.
  `depth_preview.py` now defaults to the newest session.
- 2026-10-26: M3b passed (roughly): hand size +10 % vs. ruler. Rigid-object check still to do.
- 2026-10-26: `depth_preview.py` density work, on the hand session (`--size 800 --max-disp 256 --zmin 150
  --zmax 600 --min-gray 40`): 1 frame 47k points; `--frames 10 --min-hits 0.3` (per-pixel median of 10
  frames) 78k points with the wall and hand filled in as continuous surfaces. Also: right image rendered
  128 px wider so the left strip gets depth; `--block`/`--speckle` flags (smaller values keep thin
  objects but add noise). Thin dark objects (pen) still mostly missing; glossy desk gives a mirrored ghost.
