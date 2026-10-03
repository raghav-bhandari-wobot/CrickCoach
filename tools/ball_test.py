"""Quick check of a ball detector on a clip: draws every detection, green = moving, grey = static.

    python tools/ball_test.py videos/clip.mp4 --weights weights/ball_best.pt [--imgsz 1280] [--conf 0.15]
    -> out/_ball_test/<clip>_ball.mp4  +  per-frame detections in <clip>_ball.npz
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.15)
ap.add_argument("--out", default="out/_ball_test")
ap.add_argument("--tag", default="", help="suffix for output names, e.g. the model name")
a = ap.parse_args()

model = YOLO(a.weights)
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stem = Path(a.video).stem + (f"_{a.tag}" if a.tag else "")
vw = cv2.VideoWriter(str(out / f"{stem}_ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

hist = []                 # per frame: (k,5) [x, y, w, h, conf]
STATIC_PX = 0.004 * max(W, H)   # moved less than this over the last 5 frames -> static (lying on the ground)
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    r = model.predict(fr, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
    d = np.c_[r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()] if len(r.boxes) else np.zeros((0, 5))
    hist.append(d)
    prev = hist[-6] if len(hist) > 5 else np.zeros((0, 5))
    n_mov = 0
    for x, y, w, h, c in d:
        moving = True
        if len(prev):
            moving = np.min(np.hypot(prev[:, 0] - x, prev[:, 1] - y)) > STATIC_PX
        n_mov += moving
        col = (0, 255, 0) if moving else (150, 150, 150)
        cv2.rectangle(fr, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), col, 2)
        cv2.putText(fr, f"{c:.2f}", (int(x + w / 2) + 2, int(y)), 0, 0.5, col, 1, cv2.LINE_AA)
    cv2.rectangle(fr, (0, 0), (430, 34), (0, 0, 0), -1)
    cv2.putText(fr, f"f{i} t={i / fps:.2f}s  balls {len(d)}  moving {n_mov}", (8, 24), 0, 0.7, (0, 255, 255), 2)
    vw.write(fr)
    if i % 100 == 0:
        print(f"\r[ball] {i}/{n}", end="", flush=True)
    i += 1
vw.release()
np.savez_compressed(out / f"{stem}_ball.npz", **{f"f{k}": v for k, v in enumerate(hist)})
cnt = np.array([len(h) for h in hist])
print(f"\n{stem}: frames {i}, frames with >=1 ball {np.mean(cnt > 0):.0%}, mean balls/frame {cnt.mean():.2f}, "
      f"max {cnt.max()}  -> {out / (stem + '_ball.mp4')}")
