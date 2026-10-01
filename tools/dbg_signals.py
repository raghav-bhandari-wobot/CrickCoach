"""Rebuild signals from the inference cache (no models) - used for tuning the phase rules.

    python tools/dbg_signals.py "videos/Cricket Sample 29 Clip.mp4"
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batsman_analyser.config import Config  # noqa: E402
from batsman_analyser.signals import build_signals  # noqa: E402
from batsman_analyser.tracking import select_batsman, track  # noqa: E402
from batsman_analyser.video_io import video_meta  # noqa: E402


def load(video, out_dir="out", cfg=None):
    cfg = cfg or Config()
    video = Path(video)
    out = Path(out_dir) / video.stem
    meta = video_meta(video)
    det = dict(np.load(out / "pose.npz"))
    T = int(max(meta["n_frames"], det["p_frame"].max() + 1))
    meta["n_frames"] = T
    cuts = json.loads((out / "cuts.json").read_text())
    tid = track(det, T, cfg.track, cuts)
    per_frame, best, tinfo = select_batsman(det, tid, T, meta["fps"], cfg.track, (meta["height"], meta["width"]))
    seg_id = np.where(per_frame >= 0, tid[np.maximum(per_frame, 0)], -1)
    bat = dict(np.load(out / "bat.npz")) if (out / "bat.npz").exists() else None
    S = build_signals(det, per_frame, bat, meta, cfg.smooth, seg_id)
    S["seg_id"] = seg_id
    S["video_fps"] = meta["fps"]
    return S, det, per_frame, meta, cfg


if __name__ == "__main__":
    S, det, pf, meta, cfg = load(sys.argv[1])
    np.savez(sys.argv[2] if len(sys.argv) > 2 else "signals_dbg.npz",
             **{k: v for k, v in S.items() if isinstance(v, np.ndarray)})
    print("saved; keys:", sorted(k for k, v in S.items() if isinstance(v, np.ndarray)))
