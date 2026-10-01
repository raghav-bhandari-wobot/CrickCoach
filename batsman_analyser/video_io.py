import subprocess

import cv2
import numpy as np


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 1000:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "n_frames": n, "width": w, "height": h}


def iter_frames(path):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_audio(path, sr=22050):
    """Mono float32 audio, or None if the video has no audio track."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except Exception:
        return None, sr
    if not out or len(out) < sr // 2:
        return None, sr
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0, sr


def scene_cuts(path, cell_thr=14.0, frac=0.5, min_gap=15):
    """Frame indices where a hard cut starts: most of a coarse 12x20 grid changes at once.
    (A person walking past the camera changes only part of the grid, so it is not a cut.)"""
    cap = cv2.VideoCapture(str(path))
    prev, cuts, i = None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (12, 20), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and (np.abs(g - prev) > cell_thr).mean() > frac and (not cuts or i - cuts[-1] >= min_gap):
            cuts.append(i)
        prev = g
        i += 1
    cap.release()
    return cuts
