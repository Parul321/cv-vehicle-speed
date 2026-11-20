"""Ablation grid -> runs/results.csv (resumable: finished runs are reused).
Usage: python benchmark.py clip.mp4 REF.csv [--quick]
REF = gt.csv (track_id,gt_kmh) or res_track.csv (agreement with tracking reference)."""
import sys, os, json, subprocess, pandas as pd
from compare import load_ref, per_track, metrics

video, ref_path, quick = sys.argv[1], sys.argv[2], "--quick" in sys.argv
ref, is_gt = load_ref(ref_path); os.makedirs("runs", exist_ok=True)
grid = [("track", 2, 640, []), ("farneback", 2, 640, []), ("raft_small", 2, 640, []), ("raft_large", 2, 640, [])]
if not quick:
    grid += [("raft_large", 1, 640, []), ("raft_large", 3, 640, []), ("raft_large", 2, 480, []),
             ("raft_large", 2, 640, ["--erode", "5"]), ("raft_large", 2, 640, ["--pct", "75"])]
rows = []
for flow, gap, w, extra in grid:
    tag = "".join(e.strip("-") for e in extra)
    out = f"runs/{flow}_g{gap}_w{w}{tag}"
    if not os.path.exists(out + ".json"):
        subprocess.run([sys.executable, "speed_estimator.py", video, "--flow", flow, "--gap", str(gap), "--width", str(w), "--out", out] + extra, check=True)
    s = json.load(open(out + ".json")); row = {"config": os.path.basename(out), "ms_per_flow": s["ms_per_flow"]}
    if flow == "track" and not is_gt: row.update(n="-", note="reference")
    else: row.update({k: round(v, 2) for k, v in metrics(per_track(out + ".csv"), ref).items()})
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv("runs/results.csv", index=False); print(df.to_string(index=False))
