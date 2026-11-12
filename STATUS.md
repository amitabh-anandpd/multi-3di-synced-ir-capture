# Status

## Works
- `leapc.py`: ctypes LeapC access, multi-device, no compiled bindings (SDK 6.2.0).
- `spike_two_cams.py`: both 3Dis stream IR images at the same time.
- `list_devices.py`: stable names (cam_A = …6195, cam_B = …6230). Survived swapping the USB ports.
- `capture_multi.py`: simultaneous capture of both cameras into a session folder. Since 2026-10-27
  frames go to `cam_X/frames.bin` (raw, offsets in `frames.csv`); `session_io.py` reads both this and
  the older PNG sessions; `tools/export_png.py` exports PNGs.
- `tools/stream_health.py`: FPS, drops, jitter, latency, queue depth per camera, for one or more sessions.

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

## M4 stream health (2026-10-27, `tools/stream_health.py`) — passed

After switching capture to one raw `frames.bin` per camera (plugged in):

| run | cam_A FPS | cam_B FPS | drops | jitter | latency | max queue |
|---|---|---|---|---|---|---|
| A alone, 20 s | 90.0 | — | 0 | 0.27 ms | 9.4 ms | 96 |
| B alone, 20 s | — | 90.0 | 0 | 0.40 ms | 8.1 ms | 166 |
| together, 20 s | 89.2 | 89.2 | 0 / 1 | 1.0 ms | 11 ms | 97 / 48 |
| together, 60 s | 89.1 | 89.3 | 2 / 1 | 1.0 ms | 12.5 ms | 107 / 92 |

Together vs. alone: −1 % FPS, 0.03 % frames dropped. Disk: ~3 GB/min for two cameras (raw, uncompressed).

Before the fix, with one PNG per image (20 s runs):

| run | cam_A FPS | cam_B FPS | writer queue |
|---|---|---|---|
| alone, AC, empty scene | 89.7 | 89.2 | 12 / 253 |
| together, PNG level 1, person in cam_B view | 60.8 | 60.3 | ~500 |
| together, PNG level 0, empty scene | 70.4 | 70.3 | 8 |
| together, PNG level 0, written outside OneDrive | 76.3 | 75.8 | 6 |
| together, image writes stubbed out | 89.4 | 89.4 | 0 |

No frame_id gaps in any run: the cameras slow down, frames aren't dropped. Cause: file creation
(~360 PNGs/s), which starves the tracking service. A person in view costs a further ~9 % (hand tracking).
On battery everything is worse. Fixed by writing one raw file per camera (above).

## M5 frame pairing (2026-10-27, `pair_frames.py`) — passed

Nearest device timestamp to each cam_A frame, groups kept only if |gap| ≤ half a period (5.55 ms) and
no frame is reused.

| session | FPS | groups kept | median gap | mean \|gap\| | p95 \|gap\| | max |
|---|---|---|---|---|---|---|
| 122505 (60 s, raw) | 89 | 99.3 % | −0.05 ms | 0.54 ms | 3.28 ms | 5.54 ms |
| 122334 (20 s, raw) | 89 | 99.6 % | −0.64 ms | 0.72 ms | 1.19 ms | 5.24 ms |
| 121816 (PNG era, slowed to 74 FPS) | 74 | 91.3 % | +0.02 ms | 3.14 ms | 5.78 ms | 6.38 ms |

- At full rate the two cameras' frames arrive nearly in step: 85–99 % of pairs within 1 ms per 5 s
  window, with the median drifting only ~0.003 ms/s (−0.14 → +0.03 ms over 60 s).
- Outliers are mostly isolated single frames (206 runs, median length 1): timestamp jitter, not
  a phase change. In 122505 they grow after 45 s (within 1 ms: 98 % → 53 %).
- Whether device_ts reflects exposure time or arrival time is unknown; M6 (physical event) decides
  how far these gaps can be trusted.

## M6 physical sync (2026-10-27, `tools/check_sync.py`) — passed

Session 2026-10-27_154818: both cameras aimed at one desk spot (~20–30 cm, aimed with
`tools/live_view.py`), hand slapping it; 83.5 FPS on both, 91.6 % of frames grouped.

| event (s) | best lag (groups, + = cam_B later) | r |
|---|---|---|
| 0.13 | 0 | 0.79 |
| 0.93 | +1 | 0.72 |
| 2.28 | +1 | 0.70 |
| 2.83 | 0 | 0.75 |
| 3.97 | 0 | 0.69 |

5/5 in the same or adjacent group (1 group = 11.4 ms); whole-session best lag 0 (r = 0.57).
So groups.json pairs frames taken within about one frame of each other, and device_ts can be
trusted to that level. Method: motion energy per camera (most-changed cell of a 12×12 grid),
cross-correlated ±0.75 s around each event. Comparing threshold-crossing times instead gave
20–90 ms differences, because each camera sees the hand enter its view at a different moment.

## Hand-in-view runs and camera phase (2026-11-04)

| session | setup | FPS | drops | groups kept | median gap | \|gap\| p95 |
|---|---|---|---|---|---|---|
| 214824 (30 s) | 180°, ~35 cm apart, still fist | 90.0 / 90.0 | 0 / 0 | 99.9 % | +1.13 ms | 1.43 ms |
| 220133 (60 s) | 90°, 35 cm from the spot, moving open hand | 89.8 / 89.8 | 0 / 0 | 95.0 % | −5.14 ms | 5.44 ms |

- A moving hand didn't lower FPS here, so it doesn't explain the 83.5 FPS of the M6 session.
- The cameras' relative timing differs between sessions, so the small gaps in M5 can't be assumed. Cause not
  known yet (2 sessions, and the rig moved between them). 220133 sits near the worst case of half a period
  (5.55 ms), steady within ~0.2 ms over 60 s.
- Most of 220133's lost groups come from that offset sitting on the half-period limit, not from bad frames.
  By gap alone, 97.4 % fall within `--max-gap 0.5` and 99.9 % within 0.55.
- `check_sync.py` on 220133: 4/4 clear events within one group, but 6 of 10 were unclear (r below 0.5). Whole-session
  best lag 0 (r = 0.62). At 11.1 ms per group this can't tell a 1 ms offset from a 5 ms one.
- The Control Panel and tray were running during 220133 and FPS stayed at 89.8, so the panel running doesn't always
  slow capture.

## M7 interference at 90°, cam_A with cam_B on (2026-11-04, `tools/interference.py`)

Sessions 221801 (alone) and 221843 (with B), dark flask on the desk, 35 cm from each camera, gap +4.04 ms:

| | brightness | saturated | flicker | valid depth (mean of 6 samples) |
|---|---|---|---|---|
| alone | 27.4 | 0.00 % | 0.04 % | 3.3 % |
| with B | 27.5 | 0.00 % | 0.04 % | 4.2 % |

- cam_B's LEDs appear in cam_A's image as two small bright spots left of the target, with reflections in the desk.
  The whole-image numbers miss them, so the tool now also reports the share of strongly changed pixels.
- No glare on the target or desk elsewhere. The extra depth with B on is unexplained; these spots may add points.
- An earlier pair (220836 / 220955, clear bottle) showed 10 % less depth with B on, but the scene shifted between runs.
- Gap by session: +1.13, −5.14, −2.26, +4.04, −0.25 ms. The last two had no replug between them (as instructed, not
  verified), so a new capture alone appears to change it. No pattern, and no way to set it yet.
- Run 3 (222531, gap −0.25 ms): no glare at the closest gap so far, brightness +0.3, LED spots still visible. Its
  0.722 % changed pixels come mostly from the flask and probably cam_A shifting after run 1, so it can't give a
  fine comparison against the alone run.
- Not tested yet: cam_B with cam_A on.

## M7 interference at 180°, cam_A with cam_B on (2026-11-04)

Sessions 223209 (alone) and 223251 (with B), ~70 cm apart, flask between them, gap −4.57 ms: brightness 25.4 both,
0 % saturated, flicker 0.04 % both, 0.001 % of pixels changed, valid depth 3.5 % alone vs 3.7 % with B.

Not a worst-case result: the flask sits near the centre of both views, so it probably blocks the direct line between
the cameras. Direct LED exposure at 180° is untested. cam_B with cam_A on is also untested.

## M7 interference at 90°, cam_B with cam_A on (2026-11-06)

Raised, tilted-down rig. Sessions 015040 (B alone) and 015119 (with A), white boxes on the desk: brightness 15.1 → 15.3,
0 % saturated, flicker 0.64 → 1.08 %, 0.088 % of pixels changed by > 30 gray levels. No visible glare on the scene.
(Earlier alone run 014719: +0.3 brightness, 0.321 % changed.) The together run had 791 / 783 frames in 10 s (~79 FPS).
Both directions at 90° are now tested: interference is negligible.

## M7b merge check at 90° (2026-11-05, `tools/merge_check.py`) — passed on a flat board with depth scale correction

Board session 004009 (hand-held 7 × 10 board, 25 mm squares, 90 s), object session 004211 (10 s; it shows the checkerboard standing on the desk, not a box, so the
gap is measured on a flat board; recheck with a real 3D object), rig at 90°,
~33 cm from the spot. Board seen by both cameras in 175 of 1269 sampled groups, 117 used (board moved < 0.3 mm
between the two cameras' exposures), board positions spanning 151 × 67 × 139 mm.

| | cam_A square size | cam_B square size | corner residual (median / p95) | object overlap | gap (median / p90) |
|---|---|---|---|---|---|
| raw stereo depth | 23.99 mm (−4.0 %) | 22.13 mm (−11.5 %) | 7.57 / 15.42 mm | 69.7 % | 6.82 / 11.05 mm |
| `--scale-fix` (×1.042, ×1.130) | 25 | 25 | 3.53 / 9.04 mm | 68.5 % | **3.90** / 9.42 mm |

Transform cam_B → cam_A: rotation 93.6°, translation 367 mm (scaled).
- The stereo depth of both cameras reads short at ~30 cm near the edge of the view, cam_B by 11.5 %. The M3b phone
  check (−1 to −4 %) was near the centre. The scale fix comes from the board session and is applied unchanged to
  the separate object session. The cause (baseline, coarse 64 × 64 distortion map, ~2 px row error between L and R) is not known.
- Earlier attempts failed: 235105 (board propped on the desk, one pose, gap 9 mm) and 002558 (board faced cam_B and sat
  at the edge of cam_A's view; seen by both in 5 groups).
- Session 005553 (dark metal flask): 9.8k / 6.3k points, 22.5 % overlap, gap 7.98 mm median. Dark and shiny
  objects give sparse IR stereo depth.
- Raised, tilted-down rig (2026-11-06): board 011406, object 011610 (again the board propped on the desk, person in
  cam_B's view), ~40 cm from the cameras. 44 views used, rotation 90.2°, corner residual 6.23 / 15.13 mm, overlap
  29.2 %, gap **4.76** / 12.16 mm median / p90 with `--scale-fix`. cam_B still reads 11.8 % short, cam_A 3.2 %.
  Session 000248 (Control Panel open, ~47 FPS, board ~45 cm and rarely still) gave only 3 usable views.
- Session 012845 ("book", same calibration 011406): the scene was the board standing on the desk plus a metal can
  in front, no book. Overlap 20.5 %, gap 6.48 / 12.89 mm.
- Session 013241 (small white cardboard boxes on the desk, ~45 cm, calibration 011406, person faintly in cam_B):
  first real 3D object. 84k / 52k points, overlap 32.8 %, gap **5.09** / 12.12 mm median / p90 with `--scale-fix`.
  At the 5 mm target; the overlap includes the desk. Closer placement (20–25 cm) not tried yet.
- The board corners are found only with OpenCV's `CALIB_CB_FAST_CHECK` turned off and with a ±56° rectified view.

## Stereo depth error per device (2026-11-06, `merge_check.py --depth-fit`)

Fitted from board square size (lateral size is right, depth is not): disparity = f · b_eff / z_true + d0.

| | effective baseline (nominal) | disparity offset at f = 266.7 px | board 004009 / 011406 |
|---|---|---|---|
| cam_A (…6195) | 71.1 / 70.9 mm (63.9) | −3.82 / −3.18 px | fit residual 0.40 / 0.32 px |
| cam_B (…6230) | 70.4 / 70.8 mm (65.8) | +2.73 / +2.38 px | fit residual 0.17 / 0.25 px |

- The same values come out on two different rigs, so they belong to the devices. The error depends on depth
  (square-size error vs depth correlation +0.71 to +0.90 for cam_A, −0.70 to −0.85 for cam_B), so a pure scale can't
  remove it. A disparity offset of ~3 px is ~0.6° of toe between each unit's L and R imagers, which LeapC's
  extrinsics (identity rotation) don't report; the ~2 px L/R row mismatch also points at imager rotation.
- After the fit, board squares measure within −2 to +1 % (p10–p90) on both cameras.
- Merge gap (median, mm): 004211 raw 6.82 / scale 3.87 / depth fit 3.51; 013241 7.09 / 5.45 / 4.46;
  013839 6.96 / 5.39 / 4.99. (`--scale-fix` now uses every board view, not only still ones, so its numbers differ
  slightly from the M7b section.)
- Remaining corner residual 3–5 mm median. A full per-unit stereo calibration from the board (rotation between L and R,
  dense distortion map) is the proper fix for phase 2.

## M8 sequential mode (2026-11-06, `capture_multi.py --sequential`) — passed

- LeapC 6.2.0 has no per-device LED or pause control (`LeapSetPause` pauses the whole service; checked the full
  function list in LeapC.h). Instead the other 3Dis are disabled in Windows with `pnputil /disable-device`
  (instance id `USB\VID_2936&PID_1202\<serial>`, needs an Administrator terminal) and re-enabled afterwards
  (also on error / Ctrl+C). Without admin rights it prompts to unplug / replug. `pnputil` exits non-zero for
  "Device is already enabled", which is treated as success.
- `pair_frames.py` pairs a sequential session by time since each camera's first frame.
- Session 021241 (white boxes, 10 s per camera, admin terminal): 898 / 837 frames; both devices back to OK afterwards.
  cam_B flicker 0.66 % vs 0.64 % unplugged (015040) and 1.08 % with cam_A on (015119); 0.053 % of pixels changed vs
  the unplugged run (lowest of all pairs), brightness +1.2. So cam_A's LEDs were off.
- Merged with board 011406 and `--depth-fit`: overlap 35.7 %, gap **4.35** / 12.18 mm median / p90.

## N0 per-unit stereo calibration (2026-11-06, `tools/dense_map.py`, `calib_unit.py`) — rows pass, size not yet

- The 64×64 LeapC distortion map is 1.2 raw px (median) from LeapC's own `LeapRectilinearToPixelEx`, with a
  +0.6 to +0.75 px vertical bias on the right imagers. `tools/dense_map.py` builds the exact map per imager from the
  live device into `units/<serial>_dense_800_1.5_256.npz`; `stereo.Cam` uses it when present. With it, the L/R row
  error drops from 0.7–1.8 px to 0.18–0.26 px median even without any stereo fit.
- `calib_unit.py` (cv2.stereoCalibrate on the board, fit 011406, test 004009, 25 mm squares) finds a rotation between
  each unit's imagers, mostly about the vertical axis (toe-in): cam_A 0.16–0.30°, cam_B 0.56–0.68°; baseline 69.5–70.2 mm
  (LeapC says 63.9 / 65.8); fitted f 254–260 px vs 266.7 from LeapC.

| test session 004009 | squares p10 / median / p90 | row error median / p95 |
|---|---|---|
| cam_A LeapC parallel | −4.5 / −3.3 / −1.7 % | 0.26 / 0.93 px |
| cam_A calibrated (f, centre fitted) | +0.4 / +1.0 / +1.8 % | 0.20 / 0.84 px |
| cam_B LeapC parallel | −11.8 / −10.9 / −9.5 % | 0.18 / 0.54 px |
| cam_B calibrated (f, centre fitted) | +0.1 / +0.5 / +1.4 % | 0.17 / 0.65 px |

- Reprojection rms stays ~1 px with every model (fixed K, free K, extra k1/k2, coarse or dense map), so what is left is
  not lens distortion. A ~+1 % size bias remains on both units; it does not depend on the square value used (that
  scales everything). Likely causes: a slightly bent board (it is paper on card, hand-held; triangulated corners sit
  2.4–3.7 mm rms off a plane, about what 0.3 px of disparity noise gives at 40 cm) or corner-finding noise on the
  dim IR images.
- `units/<serial>.json` holds the free-K fit (K, R, T per unit).

## N1 rig calibration (2026-11-06, `calib_rig.py` → `rig.json`) — code done, exit not met at ~40 cm

Board 011406 with the N0 unit calibrations: 260 views seen by both, 85 still views used, rotation 87.1°.

| | corner residual median / p95 | gap 013241 | gap 021241 (sequential) | gap 013839 |
|---|---|---|---|---|
| `--depth-fit` (M7b tool) | 4.85 / 13.03 mm | 4.46 | 4.35 | 4.99 |
| units + rig (N1) | **3.82** / 10.99 mm | 5.13 | 5.30 | 4.74 |
| units + rig, +1 % size bias removed | 3.63 / 11.02 mm | 5.07 | 5.23 | 4.69 |

- The rig itself is precise: fitting on random halves of the views gives rigs 0.09° / 0.21 mm apart at the board.
- Corner residual grows with distance (2.86 mm under 35 cm, 3.24 at 35–45, 4.77 beyond 45 cm), as triangulation noise
  does (≈ z² / (f·b) ≈ 10 mm per px of disparity at 44 cm). The 2 mm target is below the noise at this distance.
- Gap noise floor: one camera against itself (10 frames vs 10 others) gives 1.65–2.29 mm median. The two cameras sit
  ~3 mm above that; not explained yet (SGBM on surfaces seen at a slant, per-unit accuracy). SGBM clouds agree with
  the triangulated corners within 1 mm (checked on 8 board frames), so the cloud path is consistent.
- Next test at the N2 distance (~25 cm) with a rigid board: noise should drop ~3× (z²).

### N1 at ~25 cm (2026-11-06, board 033648 on a rigid backing, boxes 033854) — current rig.json

- 243 views seen by both, 97 used, positions spanning 208 × 140 × 142 mm, rotation 89.6°, translation 270 mm.
- Corner residual **2.64** / 9.38 mm median / p95 (3.82 at ~40 cm; target 2). Boxes gap **4.80** / 12.58 mm; noise
  floor (each camera against itself) 1.63 / 1.90 mm at ~250 mm.
- The ~3 mm excess over the noise floor did not shrink with distance, so it is systematic. The desk, seen by both near
  the lower edge of their views where the board never went, comes out as two planes 5.7° apart (cam_B's desk points
  sit 9.25 mm IQR off cam_A's plane vs 1.6–1.95 mm about their own). The gap on and off the desk is the same (4.9 / 4.8).
- Units refitted with LeapC's focal fixed (rotation + baseline only): residual 3.94, gap 5.24, so the free-focal units
  stay.
- Board 032231 (cameras not moved, board small and seen side-on by cam_A) gave only 7 still views in one spot; its rig
  was 2.2° off. `calib_rig.py` now refuses to make a rig from < 20 views or < 50 mm of board positions the current
  `rig.json` unless `--force`.
- Not resolved: accuracy away from where the board was. Options: board views covering the edges of both views and a
  joint bundle adjustment of all four imagers, or an ICP refinement of the rig on the scene (N6).

## N2 hand sessions (2026-11-06, rig 033648) — recorded, FPS below target

`capture_multi.py` now writes LeapC's hand tracking to `cam_X/tracking.csv` (25 joints + palm, wrist, elbow per hand;
struct sizes checked against LeapC.h 6.2.0).

| session | pose | FPS A / B | hand tracked A / B | palm distance A / B (LeapC) |
|---|---|---|---|---|
| 035647 | open, still | 87.5 / 87.5 | 0 / 96 % | – / 321 mm |
| 035726 | fist, still | 85.6 / 85.5 | 100 / 100 % | 200 / 252 mm |
| 035752 | pinch, still | 82.6 / 82.9 | 100 / 100 % | 200 / 258 mm |
| 035809 | spread, still | 85.1 / 84.6 | 100 / 100 % | 201 / 253 mm |
| 035830 | point, still | 79.2 / 79.0 | 100 / 100 % | 273 / 345 mm |
| 035901 | slow motion, 20 s | 87.3 / 87.6 | 100 / 100 % | 261 / 326 mm |
| 035942 | fast motion, 20 s | 85.4 / 85.4 | 100 / 100 % | 253 / 319 mm |

- No frame-id gaps (stream_health): the cameras delivered fewer frames, nothing was dropped. Plugged in (battery
  status 2). Cause of 79–87 FPS not known yet (exit asks ≥ 88). Timestamps are exact, so pairing is unaffected.
- The hand is visible in both views in all sessions (contact sheet checked); the person's head and body are in view too.
- cam_A's LeapC tracking found no hand in "open still" although the hand is in its image.

## N3 per-frame depth (2026-11-06, `depth.py`) — passed

Still-hand sessions 035726, 035752, 035809, 035830, 035647 (cam_A of 035647 skipped: no LeapC tracking), 20 frames per
camera. Hand mask: LeapC joints projected into each unit's rectified image (bone lines 20 mm wide, thumb 24, palm
polygon), kept where IR-bright and not > 80 mm behind the joints. The LeapC tracking space maps to a unit's frame as
(x, y, z)_unit = (−x, z, y)_leap with the origin at +baseline/2 (found by fitting joint depth to the SGBM depth;
joints land on pixels with gray 102–114). Each frame is compared with the median of its 10 neighbours (±55 ms), since
a hand held in the air drifts over 10 s (against one mid-session median the error read 1–47 mm).

| SGBM setting | depth on hand pixels | single-frame vs local median, median abs | time per frame per camera, full / hand box |
|---|---|---|---|
| block 5, 3-way, uniqueness 10 | 78.4 % | 0.84 mm | 85 / 6 ms |
| block 7, 3-way, uniqueness 10 | 78.1 % | 0.73 mm | 84 / 6 ms |
| **block 5, 3-way, uniqueness 5** (default) | **84.6 %** | 0.85 mm | 86 / 6 ms |

- Matching only the hand box gives the same depth (0.00 mm difference) in ~6 ms instead of ~85 ms.
- cam_A gets depth on fewer hand pixels (56–74 %) than cam_B (92–99 %); the hand is closer to cam_A (≈180–200 mm).
- This measures single-frame noise, not accuracy (accuracy is the rig / unit calibration, N0–N1).

## N4 hand segmentation (2026-11-07, `segment.py`, labels from `tools/label_hands.py`) — strict IoU 0.86, not 0.9

42 hand-painted frames (21 per camera, 3 per N2 session) in `labels/`. The labeller starts from "IR-bright near the
hand", not from segment.py's mask. Mask: LeapC joints (other camera's via rig.json when this one has no hand) ->
depth in the joint box -> bright pixels within 40 mm of a bone -> arm cut 10 mm past the wrist -> holes filled.

| | IoU median (min) | frames ≥ 0.9 | ignoring 2 px along the painted edge: median (min) | ≥ 0.9 |
|---|---|---|---|---|
| cam_A, all 21 | 0.848 (0.479) | 24 % | 0.906 (0.489) | 57 % |
| cam_A, without the 3 side-on open-hand frames | 0.862 (0.763) | 28 % | 0.913 (0.800) | 67 % |
| cam_B, 21 | 0.867 (0.667) | 10 % | 0.913 (0.732) | 62 % |

- 60–66 % of the disagreement lies within 3 px of the painted edge, on both sides; growing the mask by 1–4 px lowers
  IoU. Bone radius 30/40/50 mm and brightness 25/35/45 change the median by < 0.03 (40 mm, 45 chosen).
- Worst frames: cam_A seeing the open hand side-on (035647), where the wrist position is unclear in the image, and
  the small pointing / open hand in cam_B (3–5k px, so edge pixels weigh a lot).
- 110–210 ms per frame per camera, mostly the full-image rectification (only the hand box is matched).

## N5 motion between the cameras' exposures (2026-11-07, `tools/motion_gap.py`) — passed, no correction needed

Single-frame hand clouds (segment.py) of both cameras merged with rig.json (033648), every 9th group of six N2 sessions
(705 groups). Hand motion between the exposures = 95th-percentile joint speed (LeapC) × |cam_B − cam_A gap|.

| session | pose | speed mm/s | gap ms | motion mm | hand cloud gap mm | overlap |
|---|---|---|---|---|---|---|
| 035726 | fist | 16 | 3.25 | 0.05 | 4.14 | 89 % |
| 035752 | pinch | 7 | 3.41 | 0.02 | 4.16 | 85 % |
| 035809 | spread | 10 | 3.47 | 0.03 | 4.62 | 90 % |
| 035830 | point | 13 | 3.41 | 0.03 | 4.14 | 88 % |
| 035901 | slow motion | 30 | 3.93 | 0.11 | 3.77 | 97 % |
| 035942 | fast motion | 377 | 4.26 | 1.57 | 4.01 | 95 % |

- By motion: < 0.25 mm 4.12 mm (507 groups), 1–2 mm 3.98 (75), 2–4 mm 4.01 (44), > 4 mm 4.26 (6). Still 4.12 vs moving
  3.95 mm; correlation with motion −0.09. The ~4 mm calibration gap hides motion up to ~4 mm, so no time correction now.
- Caveats: nearest-point distance barely sees motion along the surface; the fastest recording moves only 1.6 mm (median)
  between exposures. Revisit after the calibration gap shrinks, or for faster motion.

## N6 fused hand cloud per frame (2026-11-07, `fuse.py`) — passed (just)

| session | frames | points / frame | same-surface point-to-plane gap | nearest-point gap | overlap |
|---|---|---|---|---|---|
| 035809 spread, still (every 3rd) | 268 | 45 701 | **2.93 mm** | 4.66 mm | 89 % |
| 035942 fast motion | 1607 | 32 397 | **2.27 mm** | 4.05 mm | 95 % |

- Single-frame noise floor (a camera against its next frame): 0.48–1.00 mm, so the ~4 mm nearest-point gap is not noise.
- At 90° the cameras mostly see different sides of the hand; only 14–16 % of near cam_B/cam_A point pairs are the same
  surface (normals within 30°). Nearest-point distance pairs points across a finger's thickness, so the exit uses
  point-to-plane distance on same-surface pairs.
- A per-session ICP refinement (fit on half the frames, checked on the other half) did not help: still hand 9.4 mm /
  0.9° for 4.70 → 4.36 mm, fast motion 3.5 mm / 1.9° for 4.03 → 4.12 mm. Not used; the rig error is not a rigid offset.
- Playback (fused/playback.mp4, front = cam_A view and side views) shows one hand, no doubling, in the fast-motion
  session; cam_A sees this hand nearly edge-on and its points form depth layers.

## N7 MANO fit (2026-11-07, `fit_mano.py`) — accuracy passes; finger lengths only roughly checkable

Setup: MANO downloaded by the user into `mano/` (git-ignored, licence); `tools/convert_mano.py` turns the chumpy
pickles into plain numpy (chumpy does not install on Python 3.11). torch 2.13 (CPU) + smplx.
Hand: left (cam_A's LeapC says left in every frame; cam_B calls it right in some fist / pinch frames).

- Shape (betas) from 12 frames of fist, pinch and spread (fused): point-to-mesh 2.60 mm median (27 s) → `mano_shape.json`.
- Per frame: start from the previous frame (first frame from LeapC joints), point-to-plane at the nearest MANO
  vertex + weak LeapC joint pull + pose prior + smoothness; ~0.9 s per frame on CPU. Offline Gaussian smoothing of
  pose / orientation / position (`--post SIGMA`).

| session | frames | point-to-mesh as fitted | smoothed | jitter (joint acceleration) median / p95, as fitted → smoothed |
|---|---|---|---|---|
| 035809 spread, still (every 3rd frame) | 268 | 2.43 mm | 2.41 mm (σ 2) | 2.13 / 8.66 → 0.10 / 0.36 mm/frame² |
| 035942 fast motion (every 2nd frame) | 804 | 3.07 mm | 3.17 mm (σ 1) | 4.35 / 15.37 → 1.70 / 6.14 mm/frame² |

σ 2 on fast motion: 3.51 mm (lags the motion). Exit: ≤ 3 mm still ✓, ≤ 5 mm motion ✓. Overlay videos:
`<session>/mano/overlay.mp4`; the mesh sits on the points in front and side views.

Finger lengths. The user measured on the palm side, from the crease at the finger base to the tip: index 55, middle
63, ring 60, pinky 50, thumb 35 mm (thumb: from its last joint), palm width 78 mm (outside edges). MANO has joints, not
creases; with the crease taken halfway along the proximal phalanx: index 65 (+10), middle 66 (+3), ring 65 (+5),
pinky 50 (0); thumb last joint → tip 37 (+2). Moving the assumed crease by a quarter of the bone changes these by
~8 mm, so ruler lengths can't check better than ~±5 mm. MANO knuckle-centre width 64 mm (not comparable to 78 outside).
MANO MCP → tip: 82 / 83 / 79 / 61 mm; LeapC's own: 70 / 75 / 67 / 57 mm.

### N7 open question: fingertip length (2026-11-07)

- The fused cloud reaches past MANO's fingertips: points assigned to each MANO distal segment extend (p95 along the
  finger) index 9, middle 4–9, ring 3–7, pinky 0–4, thumb ~11 mm. Both cameras agree (per camera within ~2 mm), and
  the fingers are 73–83° from cam_A's view direction, so these are not view-ray streaks.
- Fit settings are not the cause: trim 0.95, no LeapC joint pull and a weaker shape prior change MANO's lengths by
  < 4 mm and held-out accuracy by 0.01 mm (3.05 vs 3.06).
- Either the cloud's fingertips are inflated (stereo edge smear + IR glow around bright skin; the ruler and LeapC both
  suggest shorter fingers than the cloud) or MANO's 10 shape parameters can't lengthen one finger on its own.
- Phone tests (2026-11-07_042950 front: dark glass, no depth; _043714 back): inconclusive, because the edge position
  depends on the brightness threshold (back median 70, bright shirt behind it, the phone's side visible to cam_A).
- Clean test still to do: white paper on a dark book cover (sharp bright/dark edge, exact A4 size).

## N8 evaluation (2026-11-07, `tools/evaluate.py` → `evaluation.json`)

All seven N2 sessions fused (every 3rd frame; fast motion every frame) and fitted (fast motion every 2nd fused frame),
smoothed with σ 2 (still) / σ 1 (motion); every 2nd fitted frame evaluated (1338 frames). `fit_mano.py` now takes
cam_B's LeapC joints via rig.json when cam_A has no hand (open still).

| session | pose | frames | surface mm (median / p90) | hand speed mm/s | jitter mm/frame² | vs LeapC raw / offset removed, mm |
|---|---|---|---|---|---|---|
| 035647 | open, still | 135 | 2.70 / **3.28** | 48 | 0.25 | 21.8 / **16.0** |
| 035726 | fist, still | 135 | 2.51 / 2.79 | 38 | 0.24 | 19.5 / 9.3 |
| 035752 | pinch, still | 130 | 2.48 / 2.56 | 31 | 0.17 | 14.8 / 2.0 |
| 035809 | spread, still | 134 | 2.41 / 2.50 | 16 | 0.09 | 9.7 / 1.7 |
| 035830 | point, still | 121 | 2.37 / 2.48 | 32 | 0.21 | 17.9 / 3.1 |
| 035901 | slow motion | 281 | 2.72 / 2.94 | 56 | 0.64 | 12.6 / 7.3 |
| 035942 | fast motion | 402 | 3.14 / 3.68 | 399 | 1.69 | 14.1 / 11.3 |

By hand speed (95th-percentile joint speed): 0–50 mm/s 2.47 mm (641 frames), 50–150 2.75 (328), 150–400 3.16 (170),
400+ 3.12 (199); vs LeapC (offset removed) 4.1 / 8.8 / 11.9 / 10.0 mm.

MANO vs LeapC by joint type, raw / offset removed: wrist 20.1 / 5.8, knuckles 12.3 / 5.9, middle joints 14.9 / 8.2,
end joints 15.8 / 8.9, tips 15.9 / 8.9 mm. The raw offset is mostly definition (where each model puts its joints).

Where it is weakest:
- Open hand, still: the only pose over 3 mm at p90 (3.28) and 16 mm from LeapC. cam_A sees the hand edge-on and had no
  LeapC tracking, so the start comes from cam_B (which calls the hand "right" in 67 of 841 frames).
- Fist: 9.3 mm from LeapC with the surface still at 2.51 mm: curled fingers are hidden, so the cloud constrains them less.
- Fast motion: surface 3.1 mm (target 5); the worst frames (4.2–4.9 mm) bunch around frames 39327–39348 and 38143–38197.
- MANO vs LeapC grows with speed (4 → 10–12 mm). Not a time lag: shifting LeapC's joints by ±1–10 frames only makes it
  worse (best at 0: 11.2 mm; ±2 frames 12.3–12.9). With no ground truth for moving joints, which one is off is unknown.
- Fingertip length: open (see N7).

## Rig recalibrated after the cameras were moved (2026-11-11, `calib_rig.py`) — current rig.json from 184155

The cameras were remounted, so the old rig gave a 10 mm gap and 1 % overlap on a hand (180927). `stereo.find` now
falls back to `cv2.findChessboardCornersSB` when `findChessboardCorners` fails: on board 183206, cam_B found the
board in 56 of 76 sampled frames instead of 11, and both cameras together in 23 instead of 3.

| board | still views | rotation / distance | corner residual median / p95 | hand check, same-surface gap |
|---|---|---|---|---|
| 183206 | 72 | 94.7° / 346 mm | 2.69 / 6.45 mm (box check 181644: 5.07 mm) | 183309: 2.48 mm, 100 % overlap |
| 184155 (camera moved again; **current**) | 28 | 96.7° / 380 mm | 3.12 / 9.72 mm | 184301: 2.46 mm, 98 % overlap |

Both pass N6's 3 mm on a hand; the corner residuals stay above N1's 2 mm, as with the old rig. cam_B has an object
close to its lens (blurred blob at the top and bottom of its view); cam_A is further than 25–30 cm from the scene.

## N9 hand sessions at the closer rig, fitted on Kaggle (2026-11-11/12, `fit_batch.py`, `fit_kaggle.ipynb`)

Cameras moved closer (rig from board 230458: 92.2°, 254 mm apart, corner residual 2.74 mm). Six 30 s left-hand
sessions, fused every 3rd frame (free fast every 2nd): same-surface gap 1.72–2.67 mm, ~38k points per frame (3× the
N2 sessions). FPS was 52–60 in five of them (Control Panel probably open).

MANO fitted with `fit_batch.py` on a Kaggle T4: all frames of a session at once, every frame starting from LeapC,
neighbours tied by a smoothness term, plus propagation passes (each frame tries the result 1, 2, 4 … 16 frames away).
Check against fit_mano.py on 2026-11-06_035901 frames 300–380: 2.71 vs 2.79 mm, 5.5 vs 9.6 mm from LeapC (offset
removed); without propagation 2.93 mm (a stretch stuck near LeapC's start). On the laptop CPU it is ~7 s per frame.

Bug found and fixed: fits that start each frame afresh can switch between the two axis-angle forms of the same
orientation (e.g. 132° about u and 228° about −u); smoothing then averaged them into a hand turned ~180° (worst
frames 50–63 mm). `fit_mano.unwrap_rotvec` now makes orient / pose continuous before `--post` smoothing (rotations
unchanged, 0.0000°). That left 10–18 mm frames in wrist rotation: a flat hand seen edge-on fits almost equally well
with palm and back swapped (mesh moves 50–80 mm, joints stay within LeapC's 14–22 mm, fingers point the same way),
and the batched fit switched between the two at 8 steps (wrist rotation) and 1 (free fast). `--post` now smooths each
stretch between switches (orientation step > 90°) separately: worst frame 7.8 mm (free fast), wrist rotation
2.58 / 3.23 mm. Which of the two meshes is right at a switch is not decided; the joint labels are nearly the same.

Training data for the teammate's `train_kaggle.ipynb`: `tools/pack_drive.py` → `drive_upload/` (13 zips, 1.7 GB, only
labelled frame pairs; tested with the notebook's own loader, pixels identical).

| session | pose | frames | surface mm (median / p90) | speed mm/s | jitter | vs LeapC raw / offset removed |
|---|---|---|---|---|---|---|
| 232548 | open close slow | 240 | 2.39 / 2.58 | 49 | 0.84 | 16.4 / 7.6 |
| 232631 | finger counting | 254 | 2.47 / 2.91 | 77 | 1.13 | 16.1 / 9.6 |
| 232713 | pinches each finger | 285 | 2.38 / 2.67 | 112 | 1.27 | 16.7 / 9.0 |
| 232756 | wrist rotation | 262 | 2.58 / 3.23 | 120 | 1.61 | 16.3 / 12.4 |
| 232838 | free slow | 339 | 2.43 / 2.80 | 194 | 1.26 | 16.7 / 11.4 |
| 232944 | free fast | 412 | 2.47 / 3.02 | 315 | 1.86 | 15.8 / 13.4 |

By speed: 0–50 mm/s 2.35 mm, 50–150 2.40, 150–400 2.50, 400+ 2.68 (p90 3.47). Before smoothing every session is
2.27–2.43 mm. MANO vs LeapC (offset removed) is 9–13 mm, higher than N8's still poses (2–3 mm) and like N8's motion
(7–11 mm); tips 17.9 mm. Numbers in `evaluation_n9.json` (`evaluation.json` stays N8).

## Capture requirement: keep the Ultraleap Control Panel closed (2026-10-27)

With the Control Panel open, FPS fell to ~50 together (45–58 per camera; which camera was slowest changed
between runs) even with no disk writes. Closed: 89.9 / 90.0 alone, 88.0 / 88.4 together. Restarting the
service did not help on its own. `capture_multi.py` now prints a warning if the panel is running.

## Open issues
- A bad USB connection made cam_B silent. Control Panel warning: "USB hub or low powered port".
  Service log: "Broken device detected, missing frame data". Fixed by reseating the cable and restarting.
  capture_multi.py now aborts if a camera sends nothing in 2 s.
- Disk use (~3 GB/min for 2 cams, raw) limits long sessions. Fine for bursts.
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
- 2026-10-26: M3b passed. Rigid object (phone, back to camera, `sessions/2026-10-26_235243`, --size 800):
  depth 443 mm (IQR ±5 mm), height 160.7 vs 166.64 mm (−3.6 %), width 77.6 vs 78.43 mm (−1.1 %);
  extent from the phone's image box × median depth / f. A 67 mm CloudCompare pick was top-to-middle,
  not a scale error.
- 2026-10-27: M4 passed after switching capture from per-image PNGs to one raw file per camera
  (60 s together: 89.1 / 89.3 FPS, 0.03 % drops).
- 2026-10-27: M5 passed: 99.3 % of frames grouped, median gap −0.05 ms, p95 3.3 ms (60 s session).
- 2026-10-27: First three M6 recordings invalid: no hand inside either view, and ~52–58 FPS because the
  Ultraleap Control Panel was open. `tools/check_sync.py` written; detection uses the most-changed cell
  of a 12×12 grid so a local hand motion stands out.
- 2026-10-27: M6 passed (5/5 events within one group). Added `tools/live_view.py` for aiming the
  cameras without the Control Panel.
