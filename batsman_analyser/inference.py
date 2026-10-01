"""Model passes over the video. Results are cached as .npz so rules can be re-tuned quickly."""
import os
from contextlib import contextmanager

import numpy as np

from .video_io import iter_frames

MAX_POLY = 48  # bat mask polygon points kept for drawing


def _device(cfg_device):
    if cfg_device:
        return cfg_device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@contextmanager
def _cwd(path):
    """YOLOE downloads its text encoder into the CWD; keep it next to the weights."""
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_pose(video, meta, mcfg, progress=True):
    from ultralytics import YOLO
    dev = _device(mcfg.device)
    model = YOLO(mcfg.pose_weights)
    half = dev.startswith("cuda")
    fidx, boxes, confs, kpts = [], [], [], []
    n = meta["n_frames"]
    for i, frame in iter_frames(video):
        r = model.predict(frame, imgsz=mcfg.pose_imgsz, conf=mcfg.pose_conf, device=dev,
                          half=half, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            k = r.keypoints.data.cpu().numpy()
            fidx.append(np.full(len(b), i))
            boxes.append(b)
            confs.append(c)
            kpts.append(k)
        if progress and i % 50 == 0:
            print(f"\r[pose] frame {i}/{n}", end="", flush=True)
    if progress:
        print()
    if not fidx:
        raise RuntimeError("No people detected in the video.")
    return {
        "p_frame": np.concatenate(fidx).astype(np.int32),
        "p_box": np.concatenate(boxes).astype(np.float32),
        "p_conf": np.concatenate(confs).astype(np.float32),
        "p_kpt": np.concatenate(kpts).astype(np.float32),
    }


def _load_bat_model(mcfg):
    name = os.path.basename(mcfg.bat_weights).lower()
    wdir = os.path.dirname(os.path.abspath(mcfg.bat_weights)) or "."
    if name.startswith("yoloe"):
        from ultralytics import YOLOE
        with _cwd(wdir):
            model = YOLOE(os.path.basename(mcfg.bat_weights))
            model.set_classes(list(mcfg.bat_prompts))
        bat_ids = {0}
    else:  # custom / fine-tuned detector: map class names
        from ultralytics import YOLO
        model = YOLO(mcfg.bat_weights)
        names = {int(k): v.lower() for k, v in model.names.items()}
        bat_ids = {k for k, v in names.items() if "bat" in v and "batsman" not in v and "batter" not in v}
    return model, bat_ids


def _bat_axis(points):
    """PCA of mask/box points -> two extreme points along the principal axis."""
    pts = np.asarray(points, np.float32)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    proj = (pts - c) @ ax
    return c + ax * proj.min(), c + ax * proj.max(), float(s[1] / (s[0] + 1e-6))


def run_bat(video, meta, bat_box, wrist_xy, mcfg, progress=True):
    """bat_box: (T,4) batsman box per frame (NaN = unknown). wrist_xy: (T,2) mean wrist."""
    dev = _device(mcfg.device)
    model, bat_ids = _load_bat_model(mcfg)
    half = dev.startswith("cuda")
    T = meta["n_frames"]
    H, W = meta["height"], meta["width"]
    out = {
        "bat_conf": np.zeros(T, np.float32),
        "bat_a": np.full((T, 2), np.nan, np.float32),       # axis end 1
        "bat_b": np.full((T, 2), np.nan, np.float32),       # axis end 2
        "bat_thin": np.full(T, np.nan, np.float32),         # minor/major spread (low = stick-like)
        "bat_poly": np.full((T, MAX_POLY, 2), np.nan, np.float32),
    }
    for i, frame in iter_frames(video):
        if i >= T:
            break
        b = bat_box[i]
        if not np.all(np.isfinite(b)):
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bw, bh = (b[2] - b[0]) * mcfg.crop_expand_w, (b[3] - b[1]) * mcfg.crop_expand_h
        side = max(bw, bh * 0.75)
        x0, x1 = int(max(0, cx - side / 2)), int(min(W, cx + side / 2))
        y0, y1 = int(max(0, cy - bh / 2)), int(min(H, cy + bh / 2))
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        crop = frame[y0:y1, x0:x1]
        r = model.predict(crop, imgsz=mcfg.bat_imgsz, conf=mcfg.bat_conf,
                          device=dev, half=half, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None
        wr = wrist_xy[i]
        ref = wr if np.all(np.isfinite(wr)) else np.array([cx, cy])
        # --- bat: highest conf, penalised by distance of its nearest point to the hands
        best, best_s = None, -1e9
        for j in range(len(cls)):
            if cls[j] not in bat_ids or conf[j] < mcfg.bat_conf:
                continue
            if polys is not None and len(polys[j]) >= 5:
                pts = polys[j] + [x0, y0]
            else:
                x_0, y_0, x_1, y_1 = xyxy[j]
                pts = np.array([[x_0, y_0], [x_1, y_1], [x_0, y_1], [x_1, y_0]], np.float32)
            d = np.min(np.linalg.norm(pts - ref, axis=1)) / max(1.0, (b[3] - b[1]))
            s = conf[j] - 1.5 * d
            if s > best_s:
                best, best_s = (j, pts), s
        if best is not None:
            j, pts = best
            a, c, thin = _bat_axis(pts)
            out["bat_conf"][i] = conf[j]
            out["bat_a"][i], out["bat_b"][i], out["bat_thin"][i] = a, c, thin
            sel = np.linspace(0, len(pts) - 1, min(MAX_POLY, len(pts))).astype(int)
            out["bat_poly"][i, :len(sel)] = pts[sel]
        if progress and i % 50 == 0:
            print(f"\r[bat] frame {i}/{T}", end="", flush=True)
    if progress:
        print()
    return out
