# multi-3di-synced-ir-capture

Two Ultraleap 3Di cameras record synchronised IR stereo of a hand from two sides. The pipeline pairs the
frames by timestamp, builds a fused 3D point cloud per frame, and fits the MANO hand model to it. The output
is a training dataset: IR image pairs with 3D hand labels that are more accurate than the Ultraleap device's
own tracking of the same frames.

Each 3Di is itself a stereo pair, so four images are captured per frame (2 cameras × left/right imager) at
~88 FPS.

## Where it stands

| | |
|---|---|
| Dataset | 22 sessions, 12,428 labelled frame pairs, 3.6 GB of zips |
| People | 1 (P01), both hands — the goal is ≥ 5 |
| MANO surface vs the fused cloud | 2.45 mm median (2.1 mm still, 2.7 mm fast motion) |
| MANO vs 99 hand-clicked frames | 10.9 mm median, against Ultraleap's 14.2 mm on the same joints |
| Finger lengths vs clicked lengths | within 3.5 mm, except the thumb (7.9 mm short) |

Phases 1 (capture and sync) and 2 (hand reconstruction) are complete. Phase 3 (the dataset and a model
trained on it) is in progress: the tooling and the accuracy work are done, more people still need recording.

## Read next

- **[HANDOFF.md](HANDOFF.md)** — start here to run anything. Setup, the files that are not in git, every
  command in order, and the known failure modes.
- **[PLAN.md](PLAN.md)** — the milestones (M, N, P) with their exit criteria, and what each phase still owes.
- **[STATUS.md](STATUS.md)** — every measured result, in the order it was measured.
- **`dataset.json`** — one row per session: person, hand, poses, rig, FPS, frame count, surface error,
  packed zip. Read this for training splits rather than scanning folders.

## The short path

```
python capture_multi.py --duration 30 --notes "free slow"   # one recording at a time
python tools/rig_check.py                                   # gap <= 3 mm, else recalibrate the rig
python process.py --latest 6                                # pair, fuse, pack -> fit_inputs/<session>.npz
#   then fit on a Kaggle GPU with fit_kaggle.ipynb, download params.zip into fit_inputs/
python finish.py                                            # smooth, evaluate, pack -> drive_upload/<session>.zip
python dataset.py                                           # the table of every session in the dataset
```

Fitting MANO costs about 1 s per frame on the CPU, so it runs batched on Kaggle (`fit_batch.py`,
`fit_kaggle.ipynb`). Everything else runs locally on Windows.

## Requirements

Windows, Python 3.11, Ultraleap SDK 6.2.0 (Gemini), and two 3Di cameras on separate USB 3 controllers.
`pip install -r requirements.txt` — PyTorch on the CPU is enough. **Close the Ultraleap Control Panel before
capturing**; with it open the frame rate drops to about 50 FPS.

## MANO licence

The MANO model is not in this repository and must not be committed to it. Register at
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/), download the model yourself, put it in `mano/`, and run
`python tools/convert_mano.py`. On Kaggle, each person supplies their own MANO login as notebook secrets;
`fit_kaggle.ipynb` removes `mano/` before the output is saved.

Recordings (`sessions/`, ~3 GB per minute) and the packed dataset zips are not in git either — see the table
at the top of [HANDOFF.md](HANDOFF.md) for how to get each one or regenerate it.
