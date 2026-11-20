# Record a known-speed clip (this gives the real MAE)
1. **Spot:** footbridge or first-floor window over a straight road. Phone fixed (tripod / taped), landscape, 1080p, no zoom change, no shake. 20-40 s per clip.
2. **Measure with a tape:** two markers (cones, chalk lines) 10-20 m apart ALONG the road, and the lane width. Both must be visible in the first frame.
3. **Drive passes:** a friend drives past 6-10 times at constant speeds (20, 30, 40, 50 km/h; repeat some). Hold speed with cruise control or read a GPS speedometer app. Write down the speed of each pass IN ORDER. Keep other traffic minimal.
4. **Calibrate:** `python calibrate.py mine.mp4 --width-m <lane width> --length-m <marker distance>`; click near-left, near-right, far-right, far-left so the 4 points are the marker ends on both lane edges (a clean trapezoid).
5. **Run:** estimator with `--flow raft_large`, `farneback`, `track` (use the same `--out` prefixes as the README).
6. **Build gt.csv:** `python list_tracks.py res_raftL.csv`; match each pass (by frame order) to a track_id; write `track_id,gt_kmh` rows.
7. **Table:** `python compare.py gt.csv res_raftL.csv res_farneback.csv res_track.csv` -> `compare_results.md` (put on the result slide), then `python benchmark.py mine.mp4 gt.csv --quick`.
