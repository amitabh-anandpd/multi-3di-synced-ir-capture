# Build Plan — Phase 1 (capture + sync)

Two capture modes, built in this order:
- **Simultaneous**: all units stream at once and frames are grouped by timestamp. Built first.
- **Sequential**: one unit at a time, others off. Static scenes only. Built later (M8).

Each milestone has an exit check. Don't start the next one until it passes.
Record results in `STATUS.md` as you go.

---

## M0 — Setup (manual)
- Install the Ultraleap tracking software on Windows. Record the exact version.
- Plug both 3Dis into separate USB 3 ports, ideally on separate controllers.
- Create a Python venv and `requirements.txt` (numpy, opencv-python, LeapC bindings).

**Exit:** both units show up in the Ultraleap control panel at the same time.

## M1 — Bindings + two-camera spike (the risk gate)
- Install the LeapC Python bindings (`ultraleap/leapc-python-bindings`).
- Write a throwaway script `spike_two_cams.py`:
  - open a multi-device-aware connection
  - list devices and their serials
  - subscribe to each device and enable the images policy for each one
  - for 10 s, print `device, frame_id, device_ts, LeapGetNow()` and per-device FPS
- If the bindings are missing the per-device calls, add the cffi declarations ourselves.
- Check every API name against the installed SDK headers. Don't guess.

**Exit:** both devices deliver image frames concurrently, at close to the FPS each gets alone.
If not, stop and fix it (USB ports, SDK version) before going on.

## M2 — `list_devices.py` + `config.json`
- Map serial → `cam_A`, `cam_B`, … Keep existing entries and add new serials at the end.

**Exit:** after unplugging and replugging, the same serial still gets the same name.

## M3 — `capture_multi.py` (simultaneous)
- The callback only copies the buffer and puts it on a queue. A writer thread saves it.
- Per frame, save the L/R images, frame_id, device_ts, host_ts (`LeapGetNow()`), and the queue depth.
- Save the distortion map once per device per session, and again whenever its version changes.
- Session layout as in CLAUDE.md. Write `session.json` (cameras, serials, SDK version, notes).
- Flags: `--duration`, `--cams`.

**Exit:** a 60 s two-camera session produces a clean folder, and queue depth stays bounded.

## M3b — Depth preview (early check of phase 2)
- From one captured frame of one camera: undistort with the saved distortion map into one shared
  virtual pinhole camera (the imagers are parallel, so the pair comes out rectified), run OpenCV
  StereoSGBM, convert disparity to depth with the baseline from the LeapC extrinsics, and write a `.ply`.
  Implemented as `depth_preview.py`.
- View it in CloudCompare or MeshLab.
- Throwaway script. It answers whether 3Di stereo depth is usable before phase 2 relies on it.

**Exit:** a `.ply` of a known object at 20–50 cm that looks right, with its size roughly correct.

## M4 — Stream health (`tools/stream_health.py`)
- Per camera: FPS, frame_id gaps (drops), timestamp jitter. Read from `frames.csv`.
- Run each unit alone, then both together. Compare.

**Exit:** both together lose under ~5 % FPS vs. alone, with no systematic drops. Numbers go in STATUS.md.

## M5 — `pair_frames.py`
- Take a reference camera and find the nearest-timestamp frame on every other camera.
- Write `groups.json`. Print a histogram of the time gaps. Drop groups with a gap over half a frame period.

**Exit:** the gap distribution is measured and recorded.

## M6 — Physical sync test (`tools/check_sync.py`)
- Make a sharp event (clap, or a hand slapping the table) and detect the brightness spike per camera.

**Exit:** the event lands in the same group, or in adjacent groups within the expected error.

## M7 — Interference (`tools/interference.py`)
- At 180° and at 90°: A alone vs. A with B on. Use multi-second runs.
- Report mean brightness, % saturated pixels, and flicker over time. Save example frames.

- Once the mounting angle is chosen, mount the rig rigidly and record one ordinary session of a
  printed checkerboard moved through the shared view. Phase 2 calibrates the extrinsics from it,
  so no code is needed now. Recalibrate if the rig moves.

**Exit:** the result per angle is recorded, which decides whether simultaneous is usable for each
geometry. A calibration-board session is recorded on the final rig.

## M7b — Merge check (`tools/merge_check.py`)
- Find the board corners in 3D from each camera's stereo pair, then fit the rigid transform between
  the cameras from the M7 calibration session.
- Apply it to a separate session of a static rigid object. Write a merged `.ply` for MeshLab and
  measure the gap between the two clouds where they overlap.

**Exit:** median gap of about 5 mm or less at 30–45 cm (starting target, revisit with real data).

## M8 — Sequential mode (later)
- `capture_multi.py --sequential`: capture each camera in turn while the others are paused or off.
- First find out whether LeapC can pause a single device's LEDs. If it can't, fall back to
  unplugging with a prompt.
- Same session layout, so pairing and later phases don't change.

**Exit:** a sequential session of a static scene, with no interference visible.

---

## Phase 1 done
M0–M7b pass, M8 works, and STATUS.md holds: FPS alone vs. together, the sync gap distribution,
the physical sync result, interference per angle, and the merge gap.

---

# Build Plan — Phase 2 (hand reconstruction)

Goal: a merged hand point cloud per frame from both cameras while the hand moves (simultaneous
capture, 90 FPS), then a MANO hand model fitted to it (pose + shape per frame).

Starting point (STATUS.md): 90° rig, merge gap 4.4–5 mm on static boxes with `--depth-fit`; each
3Di's depth has an effective baseline of ~71 mm and a ~3 px disparity offset; the cameras run up to
half a frame (5.6 ms) apart; the 64×64 distortion map leaves ~2 px row error between L and R.

Same rules as phase 1: each milestone has an exit check, results go in STATUS.md.

---

## N0 — Per-unit stereo calibration (`calib_unit.py`)
- From the board sessions, calibrate each 3Di's left–right pair properly: the rotation between the
  imagers (the toe-in behind the disparity offset) and any focal/centre correction, on top of LeapC's
  distortion map.
- Replace the 64×64 map with a dense one from `LeapPixelToRectilinearEx` (or refine it from the board)
  so rows line up.
- Save per serial in `units/<serial>.json`; it belongs to the device, not the rig.

**Exit:** on a board session not used for fitting, squares measure within ±1 % (p10–p90) across the
view at 20–50 cm, and the L/R row error is under 0.5 px.

## N1 — Rig calibration (`calib_rig.py` → `rig.json`)
- Move the board-corner and transform code out of `tools/merge_check.py` into a module.
- A board session gives `rig.json`: camera poses in cam_A's frame, the board session id, the unit
  calibrations used, and the residuals. Recalibrate whenever the rig moves.
- `merge_check.py` and later tools read `rig.json`.

**Exit:** board corner residual between cameras median ≤ 2 mm; merge gap on a static matte object
≤ 3 mm.

## N2 — Record hand sessions + Ultraleap tracking
- Rig at 90°, both cameras ~25 cm from the hand space (fingers are thin; depth error grows with
  distance squared), matte cloth on the desk, nobody else in view, Control Panel closed.
- `capture_multi.py` also saves LeapC's own hand tracking per device (joints, palm, confidence) next
  to the images. It costs nothing and gives a hand box and a pose to start from.
- Protocol: a still open hand, a set of still poses (fist, pinch, spread, point), then slow and fast
  motion. 10–20 s each.

**Exit:** sessions recorded at ≥ 88 FPS on both cameras, tracking saved, hand inside both views.

## N3 — Per-frame depth (`depth.py`)
- One frame per camera (no multi-frame median: the hand moves). SGBM tuned for hands; run offline.
- Restrict matching to the hand region (N4) to save time and cut background noise.

**Exit:** on still-hand frames, ≥ 70 % of hand pixels get depth, and the single-frame cloud is within
2 mm (median) of the 10-frame median cloud. Time per frame recorded.

## N4 — Hand segmentation
- Start from the Ultraleap hand box, then keep pixels that are bright in IR (skin is) and within a
  depth band around the palm. Remove the arm past the wrist using the tracked wrist joint.

**Exit:** mask IoU ≥ 0.9 against 20 hand-labelled frames per camera.

## N5 — Motion between the cameras' exposures
- At 1 m/s a 5 ms gap moves the hand 5 mm. Measure the gap's effect first (still vs moving hand).
- If it matters: move cam_B's cloud to cam_A's time by interpolating between its two neighbouring
  frames (optical flow on the IR images, or per-point motion from the tracked joints).

**Exit:** merge gap on moving hands ≤ the still-hand gap + 1 mm.

## N6 — Fused hand cloud (`fuse.py`)
- Per frame: both clouds into cam_A's frame (`rig.json`), outlier removal, optional ICP refinement
  limited to a few mm / degrees, written to `<session>/fused/<frame>.ply` (or one `.npz`).

**Exit:** on a still hand, cloud-to-cloud gap median ≤ 3 mm and both cameras cover the back and the
side of the hand; a moving sequence plays back without visible doubling.

## N7 — MANO fit (`fit_mano.py`)
- Manual first: register at mano.is.tue.mpg.de and download the model (licence); add PyTorch +
  `smplx` (or `manotorch`) to requirements.
- Shape: fit once per person from the still poses. Pose: per frame, start from the previous frame or
  the Ultraleap joints; minimise point-to-surface distance + joint-angle prior + temporal smoothness.

**Exit:** median point-to-mesh distance ≤ 3 mm on still poses and ≤ 5 mm on motion, with no visible
jitter. MANO finger lengths within 5 mm of ruler measurements of the person's hand.

## N8 — Evaluation
- Compare MANO joints with Ultraleap's tracking and with the ruler measurements; check the result
  per pose type and per speed.

**Exit:** the numbers are in STATUS.md, with the poses and speeds where it fails.

---

## Phase 2 done
N0–N8 pass: per-frame fused hand clouds and MANO fits for moving hands, with accuracy recorded.
