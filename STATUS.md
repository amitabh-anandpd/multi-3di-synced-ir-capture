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
