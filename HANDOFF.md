# Handoff (updated 2026-11-18)

Two Ultraleap 3Di cameras record synced IR stereo. From that, the pipeline builds a fused 3D hand point cloud per
frame and fits MANO to it. Phase 1 (capture and sync, M0–M8) and phase 2 (hand reconstruction, N0–N8) are both done;
phase 3 (the training dataset, P0–P6) has its tooling and accuracy work done and needs more people recorded.
PLAN.md has the plan and exit criteria. STATUS.md has every measured result; read the N0–N8 sections first.
`dataset.json` lists every session in the dataset — use it for training splits instead of scanning folders.

## Not in git: get these from Sangameshwar

| Folder / file | What it is | If you can't get it |
|---|---|---|
| `sessions/` | All recordings (~3 GB per minute). The 22 dataset sessions are the keys of `dataset.json`; each keeps the rig it was fused with in `<session>/rig.json`. Boards: `2026-11-06_011406`, `2026-11-11_184155`, `2026-11-11_230458` (the current rig), `2026-11-12_134335` | Record new ones (below) |
| `drive_upload/*.zip` | The packed dataset, 22 zips / 3.6 GB — this is what `train_kaggle.ipynb` reads from the shared Drive's `sessions/` folder | `python finish.py` after a Kaggle fit |
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
  After clicking joints (`tools/click_joints.py`) and running `tools/eval_clicks.py` to get the 3D finger lengths,
  refit the shape with measurements: `python fit_mano.py --shape <still sessions> --person P02 --hand right`
  `--click-lengths I M R P --thumb-mm T --palm-mm W` then repack and refit on Kaggle.
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
python fit_mano.py --shape sessions/2026-11-06_035726 sessions/2026-11-06_035752 sessions/2026-11-06_035809 --person P01 \
    --click-lengths 90.9 93.4 88.9 73.4 --thumb-mm 115 --palm-mm 70
# --click-lengths INDEX MIDDLE RING PINKY (MCP to tip, mm, from tools/eval_clicks.py 3D triangulation)
# --thumb-mm: CMC base to tip (ruler); --palm-mm: index-MCP to little-MCP (ruler)
# Omit the measurement flags if you don't have ruler/click data yet; refit later when you do.
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

- Dataset: 22 sessions, 12,428 labelled frame pairs, one person (P01) both hands, 3.6 GB of zips in
  `drive_upload/`. Two wrist-rotation sessions (2026-11-11_232756, 2026-11-12_103904) are `exclude` in
  dataset.json and are not packed.
- MANO surface vs fused cloud: 2.45 mm median over the 22 sessions (2.1 mm still, 2.7 mm at 400+ mm/s;
  target 5 mm). Jitter after smoothing: 0.1–0.25 still, up to 2.2 mm/frame² fast.
- Label accuracy against 99 hand-clicked frames: MANO 10.9 mm median vs LeapC's 14.2 mm on the same joints.
  Finger lengths match the clicked lengths within 3.5 mm except the thumb (7.9 mm short).
- Exit criteria not fully met:
  - N0: per-camera size is off by about +1 %.
  - N1: rig residual 2.64 mm (target 2) and gap 4.8 mm (target 3); a 5.7° plane mismatch outside the board's volume.
  - N2: 79–87.5 FPS (target 88).
  - N4: mask IoU 0.86 strict (target 0.9).
  - P4: 1 person of the 5 wanted.
- Weakest cases:
  - Wrist rotation: the hand goes edge-on and the fit swaps palm and back (11 switches in 103904). Both wrist
    rotation sessions are excluded for this reason.
  - Open hand, still: cam_A sees it edge-on, p90 3.28 mm, 16 mm from LeapC.
  - Fast motion: worst frames 4–5 mm. MANO vs LeapC grows with hand speed (4 → 10–12 mm); this is not a time lag.
  - Clicked wrist position: 19.8 mm, because the wrist joint sits under the skin and cannot be clicked precisely.

## Suggested next steps

1. **Record more people** (PLAN.md P4). This is the one thing the dataset needs most: 1 person of the 5 wanted.
   Per person and hand, ~10 min: 5 still poses (10 s each), then 8 motion sessions (30 s each). Take ruler
   measurements at the start, then refit the shape with them once that person's joints have been clicked.
2. Wrist rotation: the fit starts each stretch from LeapC's orientation, and LeapC itself flips there, so longer
   propagation shifts do not help. Either anchor the orientation on hand-clicked frames, or re-record the pose
   more slowly with the palm facing between the two cameras.
3. The thumb is 7.9 mm shorter than the clicked length, and the middle finger 3.4 mm. Both are at the limit of
   what MANO's shape space reaches with the beta prior in place.
4. Improve the rig calibration (N1): use more board poses spread through the hand's working volume, at ~25 cm.
   A rigid camera base (P1) is still open — the cameras have been bumped twice.
