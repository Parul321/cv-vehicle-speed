"""Frames, GIFs and side-by-side image from result videos.
Usage: python make_assets.py res_raftL.mp4 res_farneback.mp4 res_track.mp4"""
import sys, os, cv2, numpy as np
from PIL import Image

os.makedirs("assets", exist_ok=True); allf = {}
for p in sys.argv[1:]:
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release(); name = os.path.splitext(os.path.basename(p))[0]; allf[name] = fr
    for k, i in enumerate(np.linspace(len(fr) * 0.2, len(fr) * 0.8, 3).astype(int)): cv2.imwrite(f"assets/{name}_f{k + 1}.png", fr[i])
    g = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (480, int(f.shape[0] * 480 / f.shape[1]))), cv2.COLOR_BGR2RGB)) for f in fr[::3]]
    g[0].save(f"assets/{name}.gif", save_all=True, append_images=g[1:], duration=120, loop=0)
i = min(len(v) for v in allf.values()) // 2; h = next(iter(allf.values()))[0].shape[0]; tiles = []
for name, fr in allf.items():
    t = cv2.resize(fr[i], (int(fr[i].shape[1] * h / fr[i].shape[0]), h)); cv2.putText(t, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2); tiles.append(t)
cv2.imwrite("assets/side_by_side.png", np.hstack(tiles)); print("saved to assets/")
