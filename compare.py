"""Compare speed estimates per vehicle.
Usage: python compare.py REF.csv A.csv B.csv ...
REF = gt.csv (track_id,gt_kmh) -> ground-truth accuracy,
      or a run csv (e.g. res_track.csv) -> agreement with a reference method (NOT ground truth)."""
import sys, os, pandas as pd, numpy as np


def per_track(path, min_rows=3):
    g = pd.read_csv(path).groupby("track_id")["speed_kmh"]
    return g.median()[g.count() >= min_rows]


def load_ref(path):
    d = pd.read_csv(path)
    if "gt_kmh" in d.columns:
        return d.set_index("track_id").gt_kmh, True
    return per_track(path), False


def metrics(pred, ref):
    j = pd.concat([pred, ref], axis=1, keys=["p", "r"]).dropna()
    e = j.p - j.r
    return {"n": len(j), "MAE_kmh": e.abs().mean(), "bias_kmh": e.mean(), "RMSE_kmh": float(np.sqrt((e ** 2).mean())),
            "MAPE_%": (e.abs() / j.r.replace(0, np.nan) * 100).mean(), "r": j.p.corr(j.r) if len(j) > 2 else np.nan}


def main():
    ref, is_gt = load_ref(sys.argv[1])
    names = [p for p in sys.argv[2:] if os.path.abspath(p) != os.path.abspath(sys.argv[1])]
    preds = {os.path.splitext(os.path.basename(p))[0]: per_track(p) for p in names}
    common = ref.index
    for p in preds.values(): common = common.intersection(p.index)
    rows = [{"method": n, **metrics(p.loc[common], ref.loc[common])} for n, p in preds.items()]
    df = pd.DataFrame(rows).round(2)
    kind = "ground truth" if is_gt else f"reference method {os.path.basename(sys.argv[1])} (NOT ground truth)"
    print(f"reference = {kind}; {len(common)} common vehicles\n"); print(df.to_string(index=False))
    df.to_csv("compare_results.csv", index=False)
    with open("compare_results.md", "w") as f:
        f.write(f"Reference: {kind}; {len(common)} common vehicles\n\n| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows(): f.write("| " + " | ".join(str(v) for v in r) + " |\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5)); lim = max(ref.loc[common].max(), *(p.loc[common].max() for p in preds.values())) * 1.1
        for n, p in preds.items(): plt.scatter(ref.loc[common], p.loc[common], label=n, s=25)
        plt.plot([0, lim], [0, lim], "k--", lw=1); plt.xlabel("reference km/h" + ("" if is_gt else " (tracking)")); plt.ylabel("estimate km/h")
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("compare_plot.png", dpi=150); print("saved compare_plot.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
