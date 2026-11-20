# Metric Vehicle Speed Estimation from Monocular Video
**RAFT optical flow + YOLOv8-seg/ByteTrack + ground-plane homography -> per-vehicle km/h**

RAFT (Teed & Deng, ECCV 2020; https://github.com/princeton-vl/RAFT, torchvision port used here) outputs pixel motion only.
This project adds: (1) per-vehicle segmentation masks and tracking, (2) a 4-point homography that maps pixels to ground-plane metres,
(3) median flow over each vehicle's road-contact region (bottom 25% of its mask) converted to km/h, (4) validation against reference speeds.

![demo](assets/res_raftL.gif)

## Pipeline
video -> YOLOv8-seg + ByteTrack (vehicle masks, IDs) -> RAFT flow (t vs t-gap) -> homography H (calibrate.py) -> median ground displacement -> x fps / gap x 3.6 -> km/h

## Setup (Python 3.12 recommended)
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt

## Run
    python calibrate.py clip.mp4 --width-m <W> --length-m <L>         # click 4 road points: near-left, near-right, far-right, far-left
    python speed_estimator.py clip.mp4 --flow raft_large --out res_raftL
    python speed_estimator.py clip.mp4 --flow farneback  --out res_farneback   # classical baseline
    python speed_estimator.py clip.mp4 --flow track      --out res_track       # tracking-only reference (no flow)
    python list_tracks.py res_raftL.csv                               # map recorded passes to track ids -> write gt.csv (track_id,gt_kmh)
    python compare.py gt.csv res_raftL.csv res_farneback.csv res_track.csv     # MAE / bias / RMSE / r table + plot
    python benchmark.py clip.mp4 gt.csv [--quick]                     # ablations: flow model, gap, resolution, mask erosion, percentile
    python make_assets.py res_raftL.mp4 res_farneback.mp4 res_track.mp4        # screenshots, GIFs, side-by-side image

## Results
**A. Cross-method agreement (stock highway clip, no ground truth, 21 vehicles seen by all methods; reference = tracking-only):**

| Method | r vs tracking | Bias (km/h) | MAE (km/h) | RMSE (km/h) | MAPE (%) |
|---|---|---|---|---|---|
| RAFT-small | 0.54 | -5.6 | 5.7 | 7.3 | 13.9 |
| **RAFT-large** | **0.92** | +1.8 | **2.2** | 2.8 | 5.2 |
| Farneback | 0.83 | +9.5 | 9.7 | 10.5 | 23.7 |

Tracking-only is a reference method, not ground truth.

**B. Known-speed clip (ground truth):** _fill from `compare_results.md` after recording (see RECORD_GUIDE.md)._

## Limitations
Fixed camera; planar road assumption; manual 4-point calibration (absolute km/h depends on the real distances entered); blurred or tilt-shift footage degrades flow; CPU RAFT-large is slow.

## References
Teed & Deng, RAFT (ECCV 2020) | Jocher et al., Ultralytics YOLOv8 | Zhang et al., ByteTrack (ECCV 2022) | Sochor et al., BrnoCompSpeed (IEEE T-ITS 2019).