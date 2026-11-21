"""Synthetic ground-truth check (no detector or RAFT needed). A flat textured 'vehicle' moves along a planar road at a KNOWN
speed; the scene is rendered through a known camera homography and km/h is recovered with the same vehicle_speed() used by
speed_estimator.py. Test A uses the exact flow field (checks geometry + formula). Test B uses Farneback flow (adds flow error).
Usage: python synthetic_test.py"""
import numpy as np, cv2
from speed_estimator import vehicle_speed

rng = np.random.default_rng(0)
W_M, L_M, PPM = 14.0, 60.0, 40                  # road patch (m) and texture px per metre
FPS, GAP, W, H = 25, 2, 640, 360
IMG = np.float32([[120, 340], [520, 340], [400, 120], [240, 120]])   # near-left, near-right, far-right, far-left
WORLD = np.float32([[0, 0], [W_M, 0], [W_M, L_M], [0, L_M]])
H_w2i = cv2.getPerspectiveTransform(WORLD, IMG)                      # ground metres -> image (the "camera")
H_t2i = H_w2i @ np.diag([1 / PPM, 1 / PPM, 1])                       # texture pixels -> image
H_i2w = cv2.getPerspectiveTransform(IMG, WORLD)                      # image -> metres (what calibrate.py builds)

def blobs(h, w, cell):
    a = cv2.resize(rng.random((h // cell + 2, w // cell + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(a, 0, 1)

ROAD = (blobs(int(L_M * PPM), int(W_M * PPM), 20) * 255).astype(np.uint8) // 2 + 60
CAR = (blobs(int(4 * PPM), int(1.8 * PPM), 12) * 255).astype(np.uint8)

def render(y_m, x_m=6.0):
    t, m = ROAD.copy(), np.zeros_like(ROAD); y0, x0 = int(round(y_m * PPM)), int(round(x_m * PPM)); h, w = CAR.shape
    t[y0:y0 + h, x0:x0 + w] = CAR; m[y0:y0 + h, x0:x0 + w] = 1
    return cv2.warpPerspective(t, H_t2i, (W, H)), cv2.warpPerspective(m, H_t2i, (W, H), flags=cv2.INTER_NEAREST)

Y0 = 6.0   # vehicle start (m), fairly close to the camera so it is large in the image
G = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).astype(np.float32).reshape(-1, 1, 2)

def exact_flow(kmh):
    P = cv2.perspectiveTransform(G, H_i2w); P[:, 0, 1] -= kmh / 3.6 * GAP / FPS      # where each point was GAP frames earlier
    return (cv2.perspectiveTransform(P, H_w2i) - G).reshape(H, W, 2)

def test_a(kmh, len_scale=1.0):
    _, m = render(Y0 + 2); Hm = cv2.getPerspectiveTransform(IMG, WORLD * np.float32([1, len_scale]))
    return vehicle_speed(exact_flow(kmh), m, Hm, FPS, GAP)

def test_b(kmh, erode=0):
    v = kmh / 3.6; out = []
    for k in range(GAP, GAP + 5):
        a, _ = render(Y0 + v * (k - GAP) / FPS); b, m = render(Y0 + v * k / FPS)          # a = earlier frame, b = current
        flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)          # current -> earlier, as in speed_estimator
        s = vehicle_speed(flow, m, H_i2w, FPS, GAP, erode=erode)
        if s is not None: out.append(s)
    return float(np.median(out))

if __name__ == "__main__":
    speeds = (10, 20, 30, 40, 50, 60, 80, 100)
    rows = [(k, test_a(k), test_b(k), test_b(k, erode=3)) for k in speeds]
    L = ["| True (km/h) | A: exact flow | B: Farneback flow | B: Farneback + mask erosion 3 |", "|---|---|---|---|"]
    L += [f"| {k} | {a:.1f} | {b:.1f} | {c:.1f} |" for k, a, b, c in rows]
    mae = lambda i: np.mean([abs(r[i] - r[0]) for r in rows])
    L += ["", f"MAE (km/h): A exact flow {mae(1):.2f} | B Farneback {mae(2):.2f} | B Farneback + erosion {mae(3):.2f}",
          f"Calibration length entered 10% too long: 50 km/h is read as {test_a(50, 1.1):.1f} km/h (constant scale error)"]
    print("\n".join(L)); open("synthetic_results.md", "w").write("\n".join(L) + "\n")
