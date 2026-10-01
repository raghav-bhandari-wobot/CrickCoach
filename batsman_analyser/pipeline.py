"""End-to-end: inference (cached) -> batsman -> signals -> phases -> KPIs -> feedback."""
import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .feedback import coaching_feedback
from .inference import run_bat, run_pose
from .kpis import compute_kpis
from .phases import PHASES, analyse_phases
from .signals import build_signals
from .tracking import select_batsman, track
from .video_io import read_audio, scene_cuts, video_meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def analyse(video, out_dir, cfg: Config = None, hand="auto", use_bat=True, recompute=False, log=print):
    cfg = cfg or Config()
    video = Path(video)
    out = Path(out_dir) / video.stem
    out.mkdir(parents=True, exist_ok=True)
    meta = video_meta(video)
    t0 = time.time()

    # ---- pose (cached)
    pose_cache = out / "pose.npz"
    if pose_cache.exists() and not recompute:
        det = dict(np.load(pose_cache))
        log(f"[cache] pose <- {pose_cache}")
    else:
        det = run_pose(video, meta, cfg.model)
        np.savez_compressed(pose_cache, **det)
    T = int(max(meta["n_frames"], det["p_frame"].max() + 1))
    meta["n_frames"] = T

    # ---- scene cuts (cached)
    cut_cache = out / "cuts.json"
    if cut_cache.exists() and not recompute:
        cuts = json.loads(cut_cache.read_text())
    else:
        cuts = scene_cuts(video)
        cut_cache.write_text(json.dumps(cuts))
    log(f"[cuts] {len(cuts)} scene cut(s)")

    # ---- batsman(s)
    tid = track(det, T, cfg.track, cuts)
    per_frame, best, tinfo = select_batsman(det, tid, T, meta["fps"], cfg.track, (meta["height"], meta["width"]))
    seg_id = np.where(per_frame >= 0, tid[np.maximum(per_frame, 0)], -1)
    log(f"[track] {len(tinfo['chosen'])} batsman track(s), present in {np.mean(per_frame >= 0) * 100:.0f}% of frames")

    # ---- slow-motion factor (all KPIs are in real time)
    y, sr = read_audio(video)
    slowmo, sm_info = estimate_slowmo(det, per_frame, seg_id, meta, cfg)
    em = dict(meta, fps=meta["fps"] * slowmo)          # effective (real-time) fps
    log(f"[time] slow-mo factor {slowmo}x ({sm_info}) -> {em['fps']:.0f} real fps")

    # provisional signals (no bat) -> crop boxes for the bat pass
    S0 = build_signals(det, per_frame, None, em, cfg.smooth, seg_id)
    bat = None
    if use_bat:
        bat_cache = out / "bat.npz"
        if bat_cache.exists() and not recompute:
            bat = dict(np.load(bat_cache))
            log(f"[cache] bat <- {bat_cache}")
        else:
            bat = run_bat(video, meta, S0["box"], S0["hands"], cfg.model)
            np.savez_compressed(bat_cache, **bat)
    S = build_signals(det, per_frame, bat, em, cfg.smooth, seg_id)
    S["seg_id"] = seg_id
    S["video_fps"] = meta["fps"]
    log(f"[signals] TU = {S['TU']:.1f}px, bat tracked in {S['bat_valid'].mean() * 100:.0f}% of frames")

    # ---- phases
    off_fn = lambda sh: layout_off_sign(det, per_frame, S, sh)
    shots, labels, onsets = analyse_phases(S, cfg.phase, y, sr, meta["fps"], hand, off_fn)
    log(f"[phases] {len(shots)} shot(s); audio {S['audio_state']}")

    # ---- KPIs + feedback
    for n, sh in enumerate(shots):
        K = compute_kpis(S, sh["events"], sh["info"], cfg.kpi)
        if sh["info"]["kind"] == "leave":   # no bat-on-ball: keep body KPIs only
            K = {k: v for k, v in K.items() if k.startswith(("head_", "stance", "balance", "finish", "eye"))}
            K["shot_type"] = {"name": "Shot type (heuristic)", "value": "leave", "unit": "hands up, no downswing",
                              "reliability": "medium", "status": "info", "note": ""}
        sh["kpis"] = K
        sh["feedback"] = coaching_feedback(sh["kpis"])
        sh["events_time_s"] = {k: (None if v is None else round(v / meta["fps"], 3)) for k, v in sh["events"].items()}
        sh["batsman_track"] = int(seg_id[sh["events"]["impact"]])

    warnings = []
    if np.mean(per_frame >= 0) < 0.7:
        warnings.append("Batsman missing in >30% of frames (occlusion / out of frame) - KPIs less reliable.")
    if use_bat and S["bat_valid"].mean() < 0.3:
        warnings.append("Bat tracked in <30% of frames - bat KPIs and bat-based impact are unreliable.")
    if S["audio_state"] != "used":
        warnings.append(f"Audio not used for impact ({S['audio_state']}) - impact from hand/head kinematics only.")
    if not shots:
        warnings.append("No swing detected (hand speed never exceeded threshold).")

    report = {
        "video": str(video), "meta": meta, "view": "front-on (bowler's end)",
        "scene_cuts": cuts, "slowmo_factor": slowmo, "slowmo_estimate": sm_info, "real_fps": em["fps"],
        "event_times_note": "events_time_s are video-clock seconds; KPI durations are real time",
        "torso_unit_px": round(S["TU"], 1), "processing_s": round(time.time() - t0, 1),
        "bat_tracked_fraction": round(float(S["bat_valid"].mean()), 3),
        "warnings": warnings, "phase_names": PHASES, "shots": shots,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    np.save(out / "phase_labels.npy", labels)
    return report, S, labels, onsets, meta, out



def estimate_slowmo(det, per_frame, seg_id, meta, cfg):
    """Slow-motion clips play e.g. 120 fps footage at 30 fps. Pick the factor that makes the median
    downswing (top of backlift -> impact) closest to a typical real downswing."""
    if cfg.time.slowmo and cfg.time.slowmo > 0:
        return float(cfg.time.slowmo), "user"
    if meta["fps"] >= 90:
        return 1.0, "high-fps container"
    best, best_err, detail = 1.0, 1e9, {}
    for c in cfg.time.candidates:
        em = dict(meta, fps=meta["fps"] * c)
        S = build_signals(det, per_frame, None, em, cfg.smooth, seg_id)
        S["seg_id"] = seg_id
        from .phases import detect_phases, detect_shots
        ds = []
        for sh in detect_shots(S, cfg.phase):
            if sh[3] != "shot":
                continue
            ev, _, _ = detect_phases(S, sh, cfg.phase, [], "auto", None)
            ds.append((ev["impact"] - ev["top_of_backlift"]) / em["fps"])
        if len(ds) == 0:
            continue
        med = float(np.median(ds))
        err = abs(np.log(max(med, 1e-3) / cfg.time.ref_downswing_s))
        detail[c] = (len(ds), round(med, 3))
        if err < best_err:
            best, best_err = float(c), err
    return best, f"auto: factor -> (n_shots, median downswing s) {detail}"


def layout_off_sign(det, per_frame, S, shot):
    """+1 if the batsman's off side is image-right. Votes: keeper position (behind, off side) and where
    the bat/hands rest in the stance (grounded beside the back foot, on the off side)."""
    lo, p, hi = shot[:3]
    votes = 0.0
    frames = range(lo, min(p, lo + int(1.0 * S["fps"])) + 1)
    kx = []
    for f in frames:
        d = per_frame[f]
        if d < 0:
            continue
        b = det["p_box"][d]
        bh, cx = b[3] - b[1], (b[0] + b[2]) / 2
        m = np.where(det["p_frame"] == f)[0]
        for o in m:
            if o == d:
                continue
            ob = det["p_box"][o]
            oh, ocx = ob[3] - ob[1], (ob[0] + ob[2]) / 2
            # behind the batsman: smaller, feet higher in image, close horizontally
            if 0.35 < oh / bh < 1.0 and ob[3] < b[3] and abs(ocx - cx) < 2.5 * (b[2] - b[0]):
                kx.append(np.sign(ocx - cx))
    if len(kx) >= 3:
        votes += 2.0 * np.mean(kx)
    sl = slice(lo, min(p, lo + int(1.0 * S["fps"])) + 1)
    toe_dx = np.nanmedian(S["bat_toe"][sl, 0] - S["hip_mid"][sl, 0]) if np.isfinite(S["bat_toe"][sl, 0]).any() else np.nan
    if np.isfinite(toe_dx):
        votes += np.sign(toe_dx)
    hd = np.nanmedian(S["hands"][sl, 0] - S["hip_mid"][sl, 0]) if np.isfinite(S["hands"][sl, 0]).any() else np.nan
    if np.isfinite(hd):
        votes += 0.5 * np.sign(hd)
    return None if votes == 0 else float(np.sign(votes))
