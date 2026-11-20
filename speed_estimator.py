"""Metric vehicle speed from monocular video: YOLOv8-seg + RAFT flow + ground-plane homography."""
import argparse, json, csv, time
from collections import defaultdict, deque
import cv2, numpy as np, torch
from ultralytics import YOLO
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

VEHICLES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
r8 = lambda x: max(8, int(round(x / 8)) * 8)


class FlowNet:
    def __init__(self, kind, device):
        self.kind, self.dev = kind, device
        if kind == "raft_large":
            w = Raft_Large_Weights.DEFAULT; self.m = raft_large(weights=w)
        elif kind == "raft_small":
            w = Raft_Small_Weights.DEFAULT; self.m = raft_small(weights=w)
        if kind not in ("farneback", "track"):
            self.m = self.m.to(device).eval(); self.tf = w.transforms()

    @torch.no_grad()
    def __call__(self, a, b):
        """Flow from frame a to frame b (BGR uint8) -> HxWx2 float32."""
        if self.kind == "farneback":
            ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
            return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].to(self.dev)
        ta, tb = self.tf(t(a), t(b))
        return self.m(ta, tb)[-1][0].permute(1, 2, 0).cpu().numpy()


def vehicle_speed(flow, mask, Hm, fps, gap, bottom_frac=0.25, max_pts=400, pct=50, erode=0):
    """Median ground-plane displacement of mask pixels near the road contact (bottom of vehicle)."""
    if erode: mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))  # shrink mask to avoid static road pixels
    ys, xs = np.nonzero(mask)
    if len(ys) < 50: return None
    cut = ys.min() + (1 - bottom_frac) * (ys.max() - ys.min())
    sel = ys >= cut; ys, xs = ys[sel], xs[sel]
    if len(ys) > max_pts:
        i = np.random.choice(len(ys), max_pts, replace=False); ys, xs = ys[i], xs[i]
    p0 = np.stack([xs, ys], 1).astype(np.float32)
    p1 = p0 + flow[ys, xs]  # flow is cur -> prev, magnitude of displacement is identical
    w0 = cv2.perspectiveTransform(p0[None], Hm)[0]
    w1 = cv2.perspectiveTransform(p1[None], Hm)[0]
    d = np.linalg.norm(w1 - w0, axis=1)
    return float(np.percentile(d, pct) * fps / gap * 3.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--flow", default="raft_large", choices=["raft_large", "raft_small", "farneback", "track"])
    ap.add_argument("--width", type=int, default=960, help="processing width (RAFT cost scales with this)")
    ap.add_argument("--gap", type=int, default=2, help="frame gap for flow")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--yolo", default="yolov8n-seg.pt")
    ap.add_argument("--max-speed", type=float, default=160, help="reject speeds above this (km/h)")
    ap.add_argument("--pct", type=float, default=50, help="percentile of per-pixel speed (50 = median)")
    ap.add_argument("--erode", type=int, default=0, help="mask erosion kernel (px), 0 = off")
    ap.add_argument("--limit", type=float, default=0, help="speed limit km/h; flags violators (0=off)")
    ap.add_argument("--out", default="out", help="output prefix")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    W = r8(a.width); H = r8(H0 * W / W0)
    c = json.load(open(a.calib))
    src = (np.float32(c["image_points"]) * np.float32([W / W0, H / H0])).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, np.float32(c["world_points"]))

    det, flownet = YOLO(a.yolo), FlowNet(a.flow, dev)
    vw = cv2.VideoWriter(f"{a.out}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f = open(f"{a.out}.csv", "w", newline=""); wr = csv.writer(f)
    wr.writerow(["frame", "track_id", "speed_raw_kmh", "speed_kmh"])
    buf, hist, n = deque(maxlen=a.gap + 1), defaultdict(lambda: deque(maxlen=a.smooth)), 0
    allspd, nflow, t0 = defaultdict(list), 0, time.time()
    trk = defaultdict(lambda: deque(maxlen=10))  # foot-point history for --flow track

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (W, H)); buf.append(frame); n += 1
        r = det.track(frame, persist=True, classes=VEHICLES, conf=0.4, verbose=False, tracker="bytetrack.yaml")[0]
        if len(buf) <= a.gap or r.masks is None or r.boxes.id is None:
            vw.write(frame); continue
        flow = None if a.flow == "track" else flownet(frame, buf[0]); nflow += 1
        for poly, box, tid in zip(r.masks.xy, r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().tolist()):
            if cv2.pointPolygonTest(src, (float((box[0] + box[2]) / 2), float(box[3])), False) < 0: continue  # foot point must be inside calibrated quad
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            if a.flow == "track":  # tracking-only reference: box foot point through homography
                p = cv2.perspectiveTransform(np.float32([[[(box[0] + box[2]) / 2, box[3]]]]), Hm)[0, 0]
                trk[tid].append((n, p)); v = None
                if len(trk[tid]) == 10:
                    n0, p0 = trk[tid][0]; v = float(np.linalg.norm(p - p0) / ((n - n0) / fps) * 3.6)
            else:
                v = vehicle_speed(flow, m, Hm, fps, a.gap, pct=a.pct, erode=a.erode)
            if v is None or v > a.max_speed: continue
            hist[tid].append(v); s = float(np.median(hist[tid])); allspd[tid].append(s)
            bad = a.limit > 0 and s > a.limit; col = (0, 0, 255) if bad else (0, 255, 0)
            wr.writerow([n, tid, round(v, 2), round(s, 2)])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"#{tid} {s:.0f} km/h", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.polylines(frame, [src.astype(np.int32)], True, (255, 0, 0), 2)
        vw.write(frame)
    cap.release(); vw.release(); f.close()
    el = time.time() - t0
    med = {t: float(np.median(v)) for t, v in allspd.items() if len(v) >= 3}
    summ = {"flow": a.flow, "gap": a.gap, "width": W, "pct": a.pct, "erode": a.erode, "vehicles": len(med),
            "mean_kmh": round(float(np.mean(list(med.values()))), 1) if med else None,
            "violators": [t for t, v in med.items() if a.limit > 0 and v > a.limit],
            "ms_per_flow": round(1000 * el / max(nflow, 1), 1), "device": dev}
    json.dump(summ, open(f"{a.out}.json", "w"), indent=2); print(summ)


if __name__ == "__main__":
    main()