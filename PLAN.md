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
- From one captured frame of one camera: undistort with the saved distortion map, crop to
  a virtual pinhole stereo pair (adapt `perspective_crop.py`), run OpenCV StereoSGBM, convert
  disparity to depth with the device baseline, and write a `.ply`.
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

## M8 — Sequential mode (later)
- `capture_multi.py --sequential`: capture each camera in turn while the others are paused or off.
- First find out whether LeapC can pause a single device's LEDs. If it can't, fall back to
  unplugging with a prompt.
- Same session layout, so pairing and later phases don't change.

**Exit:** a sequential session of a static scene, with no interference visible.

---

## Phase 1 done
M0–M7 pass, M8 works, and STATUS.md holds: FPS alone vs. together, the sync gap distribution,
the physical sync result, and interference per angle.
