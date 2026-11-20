"""Click 4 road points forming a rectangle of known size -> calib.json.
Order: 1 near-left, 2 near-right, 3 far-right, 4 far-left. Press q to abort."""
import argparse, json, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--width-m", type=float, required=True, help="real width of clicked rectangle (m)")
ap.add_argument("--length-m", type=float, required=True, help="real length along road (m)")
ap.add_argument("--out", default="calib.json")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video); ok, frame = cap.read(); cap.release()
assert ok, "cannot read video"
pts = []

def click(e, x, y, *_):
    if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y]); cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(len(pts)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("calib", frame)

cv2.imshow("calib", frame); cv2.setMouseCallback("calib", click)
while len(pts) < 4:
    if cv2.waitKey(20) & 0xFF == ord("q"): raise SystemExit
poly = np.int32(pts)
cv2.polylines(frame, [poly], True, (255, 0, 0), 2); cv2.imshow("calib", frame); cv2.waitKey(2000)
cv2.destroyAllWindows()
if not cv2.isContourConvex(poly) or cv2.contourArea(poly) < 0.02 * frame.shape[0] * frame.shape[1]:
    raise SystemExit("BAD POINTS: need a convex trapezoid on the road, order near-left, near-right, far-right, far-left. Run again.")
w, l = a.width_m, a.length_m
json.dump({"image_points": pts, "world_points": [[0, 0], [w, 0], [w, l], [0, l]]}, open(a.out, "w"), indent=2)
print("saved", a.out)
