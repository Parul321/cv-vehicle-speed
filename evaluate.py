"""MAE / MAPE of per-track median speed vs ground truth. gt csv: track_id,gt_kmh"""
import sys, pandas as pd
pred_files, gt = sys.argv[1:-1], pd.read_csv(sys.argv[-1])
rows = []
for p in pred_files:
    d = pd.read_csv(p).groupby("track_id")["speed_kmh"].median().rename("pred").reset_index()
    m = gt.merge(d, on="track_id"); e = (m.pred - m.gt_kmh).abs()
    rows.append({"method": p, "n": len(m), "MAE_kmh": round(e.mean(), 2), "MAPE_%": round((e / m.gt_kmh * 100).mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))
