"""List tracks so you can map each recorded pass to a track_id when writing gt.csv. Usage: python list_tracks.py res_raftL.csv"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
g = d.groupby("track_id").agg(first_frame=("frame", "min"), last_frame=("frame", "max"), rows=("frame", "count"), median_kmh=("speed_kmh", "median")).round(1)
print(g[g.rows >= 5].to_string())
