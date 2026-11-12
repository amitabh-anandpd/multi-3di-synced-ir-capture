# Handoff (2026-11-07)

Two Ultraleap 3Di cameras record synced IR stereo. From that, the pipeline builds a fused 3D hand point cloud per
frame and fits MANO to it. Phase 1 (capture and sync, M0–M8) and phase 2 (hand reconstruction, N0–N8) are both done.
PLAN.md has the plan and exit criteria. STATUS.md has every measured result; read the N0–N8 sections first.

## Not in git: get these from Sangameshwar

| Folder / file | What it is | If you can't get it |
|---|---|---|
| `sessions/` | All recordings (~3 GB per minute). Needed: boards `2026-11-06_011406`, `2026-11-11_184155`; hands `2026-11-06_035647` … `035942` (7 sessions, old rig `2026-11-06_033648`), `2026-11-11_184301` (current rig) | Record new ones (below) |
| `units/*_dense_800_1.5_256.npz` | Dense rectification maps for the two cameras | `python tools/dense_map.py` with both cameras plugged in |
| `mano/` | MANO model. **Licence: register at mano.is.tue.mpg.de yourself; never commit or share these files** | Download `MANO_LEFT.pkl` / `MANO_RIGHT.pkl` into `mano/`, then `python tools/convert_mano.py` |
| `labels/` | 42 hand-painted masks for the N4 check (192 KB) | `python tools/label_hands.py` |

`units/*.json` (per-camera calibration), `rig.json` (cam_B pose relative to cam_A), `shapes/<person>_<hand>.json`
(MANO hand shape per person and hand; P01 = the first person, left hand), `dataset.json` and `evaluation*.json` are in
git. These files belong to these two physical cameras
(cam_A = serial …6195, cam_B = …6230, see `config.json`) and this rig placement. `rig.json` is from board `sessions/2026-11-11_230458` (cameras moved closer; N9 hand sessions
2026-11-11_232548 … 232944 fuse at 1.7–2.7 mm). **If either camera moves even slightly, redo the rig calibration**: record a 40 s board session
(25 mm squares, hold each pose still ~1 s, board facing between the cameras), `pair_frames.py` it, then
`python calib_rig.py <that session>`, and check by fusing a short hand recording (`fuse.py`, gap should be <= 3 mm).
`fuse.py` (and `segment.py`) always read `./rig.json`. To re-fuse the 2026-11-06 hand sessions, first copy the rig
they were recorded with back: `sessions/2026-11-06_033648/rig.json` (and restore `sessions/2026-11-11_184155/rig.json`
afterwards).

## Setup

- Windows, Python 3.11, Ultraleap SDK 6.2.0 (Gemini). Use Windows Python, not WSL (the WSL Python can't load the DLL).
- `pip install -r requirements.txt` (torch on CPU is enough).
- **Close the Ultraleap Control Panel before capturing.** With it open, the frame rate drops to ~50 FPS.
- Check that both cameras are seen: `python list_devices.py`. If one isn't, reseat its USB cable and restart the
  Ultraleap service.
- The desk is glossy and reflects the hand in IR, which puts a ghost hand under the desk. Use a matte surface.

## Quick path (PLAN.md P3): record -> Drive in three commands and one Kaggle run

```
python capture_multi.py --duration 30 --notes "N9 free slow"   # one recording at a time
python tools/rig_check.py          # after the first hand recording of a sitting: gap <= 3 mm, else recalibrate
#   a camera moved? record a board, then: python pair_frames.py <board>; python calib_rig.py <board>
#   python tools/set_rig.py <board> --dates <first hand session> <last hand session>   (those sessions get its rig)
python process.py --latest 6       # pair, fuse, rig check, pack -> fit_inputs/<session>.npz, rows in dataset.json
#   Kaggle: upload fit_inputs/<session>.npz as a dataset, run fit_kaggle.ipynb, download params.zip into fit_inputs/
python finish.py                   # unpack, smooth, evaluate, pack -> drive_upload/<session>.zip, dataset.json
python dataset.py                  # table of every session in the training set
```

- New person or hand: record 5 still poses ("... still" in the notes) + motion, then
  `python process.py --latest 11 --person P02 --hand right` fits `shapes/P02_right.json` first.
- A hand held away from where the board was held can fail the rig check (2026-11-12 right hand: 5–7 mm with a good
  rig). `process.py --refine` lets fuse.py correct the rig from the hand points, kept only if it helps on held-out
  frames (right hand: 1.6–2.5 mm afterwards). Better: move the board through the whole hand space when calibrating.
- Label accuracy: `python tools/click_joints.py` (click the 21 joints in ~50 frames), then
  `python tools/eval_clicks.py` (MANO's and LeapC's joints vs the clicks, mm).

## Pipeline, in order

```
python capture_multi.py --duration 10 --notes "N2 fist still"          # both cameras at once (hands: this mode)
python capture_multi.py --sequential --duration 10 --notes "static"     # one camera at a time, Administrator shell
python pair_frames.py                                                   # newest session -> groups.json

python tools/dense_map.py                                               # once per camera pair (cameras plugged in)
python calib_unit.py --fit sessions/2026-11-06_011406 --test sessions/2026-11-05_004009 --square 25
python calib_rig.py sessions/2026-11-06_033648 --check sessions/2026-11-06_033854

python fuse.py sessions/2026-11-06_035942                               # -> <session>/fused/ (hands.npz, PLYs, playback.mp4)
python fit_mano.py --shape sessions/2026-11-06_035726 sessions/2026-11-06_035752 sessions/2026-11-06_035809 --person P01
python fit_mano.py sessions/2026-11-06_035942 --every 2                 # -> <session>/mano/params.npz (CPU, ~1 s/frame)
python fit_batch.py --pack sessions/2026-11-11_232548                   # faster: pack -> fit on Kaggle GPU (fit_kaggle.ipynb)
python fit_batch.py --unpack fit_inputs                                 #   -> <session>/mano/params.npz from Kaggle's params.zip
python fit_mano.py sessions/2026-11-06_035942 --post 1                  # smoothing -> smoothed.npz, overlay.mp4 (sigma 2 for still poses)
python tools/evaluate.py sessions/2026-11-06_035647 sessions/2026-11-06_035726 sessions/2026-11-06_035752 sessions/2026-11-06_035809 sessions/2026-11-06_035830 sessions/2026-11-06_035901 sessions/2026-11-06_035942
```

Training data for `train_kaggle.ipynb` (two_cam mode): `python tools/pack_drive.py sessions/<fitted sessions>` writes
one zip per session to `drive_upload/` (only the frame pairs with a MANO label, ~70–250 MB each, plus the rig.json it
was fused with); drag the zips into the shared Drive's `sessions/` folder. The notebook unpacks `sessions/*.zip`.
Labels: `mano/smoothed.npz` joints (21, mm, cam_A frame). Known ambiguity: a flat hand seen edge-on can fit with palm
and back swapped (joints nearly unchanged); `--post` smooths each stretch between such switches separately.

Checks for individual stages: `python depth.py <sessions>` (N3), `python segment.py --eval` (N4),
`python tools/motion_gap.py <sessions>` (N5). Board corner detection takes 6–11 min the first time per board
session, then it is cached in the session folder. The MANO fit runs at about 1 s per frame on the CPU.

Main modules: `leapc.py` (LeapC ctypes bindings, images and tracking), `session_io.py` (reads frames),
`stereo.py` (rectification, `Unit` calibration, triangulation), `depth.py` (SGBM depth in the hand box),
`segment.py` (hand mask from LeapC joints and depth), `fuse.py`, `fit_mano.py`.

## Where it stands (details in STATUS.md)

- MANO surface vs fused cloud: 2.4–2.7 mm still, 2.7 mm slow, 3.1 mm fast motion (target 5 mm). Jitter after
  smoothing: 0.1–0.25 still, 1.7 mm/frame² fast.
- Exit criteria not fully met:
  - N0: per-camera size is off by about +1 %.
  - N1: rig residual 2.64 mm (target 2) and gap 4.8 mm (target 3); a 5.7° plane mismatch outside the board's volume.
  - N2: 79–87.5 FPS (target 88).
  - N4: mask IoU 0.86 strict (target 0.9).
- Weakest cases:
  - Open hand, still: cam_A sees it edge-on, p90 3.28 mm, 16 mm from LeapC.
  - Fist: 9 mm from LeapC.
  - Fast motion: worst frames 4–5 mm.
  - MANO vs LeapC grows with hand speed (4 → 10–12 mm). This is not a time lag.

## Suggested next steps

1. **Fingertip length question** (STATUS.md, "N7 open question"): the cloud reaches 4–9 mm past MANO's fingertips.
   Test it with white paper on a dark book cover (sharp edge, known A4 size) to see whether the stereo edges are
   inflated.
2. Re-record the open-hand pose with the palm facing between the two cameras.
3. Improve the rig calibration (N1): use more board poses spread through the hand's working volume, at ~25 cm.
4. Speed up fitting: batch frames, or use a GPU.
