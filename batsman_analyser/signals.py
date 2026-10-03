"""Per-frame kinematic signals for the selected batsman (cleaned, gap-filled, smoothed)."""
import numpy as np
from scipy.signal import medfilt, savgol_filter

NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16


def fill_gaps(x, max_gap):
    """Linear interpolation over NaN gaps <= max_gap samples (per column)."""
    x = np.array(x, np.float64)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        v = flat[:, c]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        interp = np.interp(t, t[ok], v[ok])
        # only fill short gaps
        bad = ~ok
        i = 0
        while i < len(v):
            if bad[i]:
                j = i
                while j < len(v) and bad[j]:
                    j += 1
                if i > 0 and j < len(v) and (j - i) <= max_gap:
                    v[i:j] = interp[i:j]
                i = j
            else:
                i += 1
        flat[:, c] = v
    return flat.reshape(x.shape)


def smooth(x, win, poly):
    """Savitzky-Golay on finite runs only (NaNs kept)."""
    x = np.array(x, np.float64)
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        v = flat[:, c]
        ok = np.isfinite(v)
        i = 0
        while i < len(v):
            if ok[i]:
                j = i
                while j < len(v) and ok[j]:
                    j += 1
                w = min(win, (j - i) if (j - i) % 2 == 1 else (j - i) - 1)
                if w > poly + 1:
                    v[i:j] = savgol_filter(v[i:j], w, poly)
                i = j
            else:
                i += 1
    return flat.reshape(x.shape)


def despike(x, k):
    """Median filter on finite runs: removes single-frame keypoint flips (wrist swaps, occlusion)."""
    x = np.array(x, np.float64)
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        v = flat[:, c]
        ok = np.isfinite(v)
        i = 0
        while i < len(v):
            if ok[i]:
                j = i
                while j < len(v) and ok[j]:
                    j += 1
                if j - i > k:
                    v[i:j] = medfilt(v[i:j], k)
                i = j
            else:
                i += 1
    return flat.reshape(x.shape)


def speed(xy, fps):
    d = np.gradient(xy, axis=0) * fps
    return np.linalg.norm(d, axis=-1)


def _mean_pts(kp, kc, ids, thr):
    pts = kp[:, ids, :]
    w = ((kc[:, ids] >= thr) & np.isfinite(pts[..., 0])).astype(np.float64)
    pts = np.nan_to_num(pts)
    s = w.sum(1)
    out = (pts * w[..., None]).sum(1) / np.maximum(s, 1e-9)[:, None]
    out[s == 0] = np.nan
    return out


def build_signals(det, per_frame, bat, meta, scfg, seg_id=None):
    """seg_id: per-frame id of the batsman track (different batsmen / zoom -> own torso unit)."""
    fps = meta["fps"]
    T = len(per_frame)
    kp = np.full((T, 17, 2), np.nan)
    kc = np.zeros((T, 17))
    box = np.full((T, 4), np.nan)
    ok = per_frame >= 0
    kp[ok] = det["p_kpt"][per_frame[ok], :, :2]
    kc[ok] = det["p_kpt"][per_frame[ok], :, 2]
    box[ok] = det["p_box"][per_frame[ok]]
    kp[kc < scfg.kp_conf] = np.nan

    max_gap = int(round(scfg.max_gap_s * fps))
    win = max(5, int(round(scfg.sg_window_s * fps)) | 1)
    if seg_id is None:
        ranges = [(0, T)]
    else:  # never interpolate / smooth across two different batsman tracks
        ranges = []
        for sid in np.unique(seg_id[seg_id >= 0]):
            idx = np.where(seg_id == sid)[0]
            ranges.append((idx[0], idx[-1] + 1))
    mk = max(3, int(round(scfg.median_s * fps)) | 1)
    for a, b in ranges:
        kp[a:b] = smooth(despike(fill_gaps(kp[a:b], max_gap), mk), win, scfg.sg_poly)
        box[a:b] = fill_gaps(box[a:b], max_gap)
    kc_f = np.where(np.isfinite(kp[..., 0]), np.maximum(kc, scfg.kp_conf), 0)

    S = {"fps": fps, "T": T, "kp": kp, "kc": kc, "box": box, "present": ok}
    S["head"] = _mean_pts(kp, kc_f, [NOSE, LEYE, REYE, LEAR, REAR], scfg.kp_conf)
    S["sh_mid"] = (kp[:, LS] + kp[:, RS]) / 2
    S["hip_mid"] = (kp[:, LH] + kp[:, RH]) / 2
    S["hands"] = _mean_pts(kp, kc_f, [LW, RW], scfg.kp_conf)
    torso = np.linalg.norm(S["sh_mid"] - S["hip_mid"], axis=1)
    if np.isfinite(torso).any():
        TU = float(np.nanmedian(torso))
    else:  # hips/shoulders never visible: ~1/3 of person height
        bh = box[:, 3] - box[:, 1]
        TU = float(np.nanmedian(bh) / 3.0) if np.isfinite(bh).any() else 100.0
    S["TU"] = TU
    # per-frame torso unit: rolling median (+-1 s) inside each batsman track -> follows camera zoom
    TUf = np.full(T, TU)
    half = max(1, int(fps))
    for a, b in ranges:
        seg_t = torso[a:b]
        if np.isfinite(seg_t).sum() < 5:
            continue
        pad = np.pad(seg_t, half, mode="edge")
        wv = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                roll = np.nanmedian(wv, axis=1)
        roll = np.where(np.isfinite(roll), roll, np.nanmedian(seg_t))
        TUf[a:b] = roll
    S["TU_f"] = TUf
    S["hand_h"] = (S["hip_mid"][:, 1] - S["hands"][:, 1]) / TUf
    S["head_h"] = (S["hip_mid"][:, 1] - S["head"][:, 1]) / TUf
    sw = max(5, int(round(scfg.slow_window_s * fps)) | 1)          # heavier smoothing for event finding
    S["hand_h_s"] = smooth(S["hand_h"], sw, 2)
    S["head_h_s"] = smooth(S["head_h"], sw, 2)
    S["hand_speed"] = speed(S["hands"], fps) / TUf
    S["head_speed"] = speed(S["head"], fps) / TUf
    S["ank_l"], S["ank_r"] = kp[:, LA], kp[:, RA]
    S["ank_l_speed"] = speed(kp[:, LA], fps) / TUf
    S["ank_r_speed"] = speed(kp[:, RA], fps) / TUf
    S["shoulder_w"] = np.abs(kp[:, LS, 0] - kp[:, RS, 0]) / TUf
    S["hip_w"] = np.abs(kp[:, LH, 0] - kp[:, RH, 0]) / TUf
    S["shoulder_tilt"] = np.degrees(np.arctan2(kp[:, LS, 1] - kp[:, RS, 1],
                                               np.abs(kp[:, LS, 0] - kp[:, RS, 0]) + 1e-6))
    S["eye_tilt"] = np.degrees(np.arctan2(kp[:, LEYE, 1] - kp[:, REYE, 1],
                                          np.abs(kp[:, LEYE, 0] - kp[:, REYE, 0]) + 1e-6))

    # ---------------- bat: orient ends (handle = near hands), validate, fill, smooth
    handle = np.full((T, 2), np.nan)
    toe = np.full((T, 2), np.nan)
    if bat is not None:
        a, b = bat["bat_a"].astype(np.float64), bat["bat_b"].astype(np.float64)
        da = np.linalg.norm(a - S["hands"], axis=1)
        db = np.linalg.norm(b - S["hands"], axis=1)
        swap = db < da
        handle = np.where(swap[:, None], b, a)
        toe = np.where(swap[:, None], a, b)
        length = np.linalg.norm(toe - handle, axis=1) / TUf
        near = np.minimum(da, db) / TUf
        valid = (bat["bat_conf"] > 0) & (length > 0.35) & (length < 3.2) & (near < 0.9)
        handle[~valid] = np.nan
        toe[~valid] = np.nan
        gap = int(round(0.15 * fps))
        handle = smooth(fill_gaps(handle, gap), max(5, int(round(0.1 * fps)) | 1), 2)
        toe = smooth(fill_gaps(toe, gap), max(5, int(round(0.1 * fps)) | 1), 2)
        S["bat_poly"] = bat["bat_poly"]
        S["bat_conf"] = bat["bat_conf"]
    S["bat_handle"], S["bat_toe"] = handle, toe
    S["bat_valid"] = np.isfinite(toe[:, 0])
    v = toe - handle
    # angle from vertical-down, + = toe towards image right
    S["bat_angle"] = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
    S["bat_toe_h"] = (S["hip_mid"][:, 1] - toe[:, 1]) / TUf
    S["bat_toe_speed"] = speed(toe, fps) / TUf
    return S


def ball_evidence(ball, S, bcfg, frame_wh):
    """Per frame: how many ball detections look like the LIVE ball - moving (not lying on the ground
    or carried with the bat), near the hands / bat, below the chest, and not the helmet, gloves or lifted bat handle (all get detected as balls).
    Returns (count per frame, xy of the best one per frame for drawing)."""
    xyc = ball["ball_xyc"]
    T = min(len(xyc), S["T"])
    ev = np.zeros(S["T"], np.int16)
    best = np.full((S["T"], 2), np.nan)
    static_px = bcfg.static_frac * max(frame_wh)
    TU, hands, head, wr = S["TU_f"], S["hands"], S["head"], S["kp"][:, [LW, RW]]
    for i in range(T):
        d = xyc[i]
        d = d[np.isfinite(d[:, 0]) & (d[:, 2] >= bcfg.conf)]
        if not len(d) or not np.isfinite(TU[i]) or not np.all(np.isfinite(hands[i])):
            continue
        p = xyc[i - 5] if i >= 5 else xyc[i][:0]
        p = p[np.isfinite(p[:, 0])]
        top = 0.0
        for x, y, c in d:
            q = np.array([x, y])
            if np.all(np.isfinite(head[i])) and np.linalg.norm(q - head[i]) / TU[i] < bcfg.ignore_head_tu:
                continue
            if any(np.all(np.isfinite(w)) and np.linalg.norm(q - w) / TU[i] < bcfg.ignore_wrist_tu for w in wr[i]):
                continue
            if np.linalg.norm(q - hands[i]) / TU[i] > bcfg.near_hands_tu:
                continue
            if np.isfinite(S["hip_mid"][i, 1]) and (S["hip_mid"][i, 1] - y) / TU[i] > bcfg.max_height_tu:
                continue              # up at the shoulders/head: the lifted bat handle, not a delivery
            if len(p):
                j = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
                move = q - p[j, :2]
                if np.hypot(*move) <= static_px:
                    continue          # lying on the ground
                dh = hands[i] - hands[i - 5] if i >= 5 and np.all(np.isfinite(hands[i - 5])) else np.zeros(2)
                if np.linalg.norm(move - dh) / TU[i] < bcfg.attached_tu:
                    continue          # moves with the hands: the bat handle / glove, not the ball
            ev[i] += 1
            if c > top:
                top, best[i] = c, q
    return ev, best
