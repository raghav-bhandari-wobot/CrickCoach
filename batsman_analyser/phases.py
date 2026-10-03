"""Shot segmentation and phase detection (rule-based, event-anchored)."""
import numpy as np
from scipy.signal import find_peaks

from . import audio as audio_mod

PHASES = ["-", "STANCE", "BACKLIFT", "DOWNSWING", "IMPACT", "FOLLOW-THROUGH", "FINISH", "LEAVE"]
PHASE_COLORS = {  # BGR
    "-": (80, 80, 80), "STANCE": (200, 160, 60), "BACKLIFT": (60, 200, 230),
    "DOWNSWING": (40, 140, 255), "IMPACT": (40, 40, 240), "FOLLOW-THROUGH": (80, 200, 80),
    "FINISH": (180, 120, 200), "LEAVE": (200, 200, 200),
}


def _nan0(x):
    return np.nan_to_num(np.asarray(x, np.float64), nan=0.0)


def _nanargmax(x, lo, hi, fallback):
    seg = np.asarray(x[lo:hi + 1], np.float64)
    if hi < lo or not np.isfinite(seg).any():
        return fallback
    return lo + int(np.nanargmax(seg))


def _nanargmin(x, lo, hi, fallback):
    seg = np.asarray(x[lo:hi + 1], np.float64)
    if hi < lo or not np.isfinite(seg).any():
        return fallback
    return lo + int(np.nanargmin(seg))


def _still_run(sig, start, stop, thr, n):
    run = 0
    for i in range(start, stop + 1):
        run = run + 1 if (np.isfinite(sig[i]) and sig[i] < thr) else 0
        if run >= n:
            return i - n + 1
    return None


def _rng(x, lo, hi, fn):
    seg = np.asarray(x[max(0, lo):hi + 1], np.float64)
    return fn(seg) if np.isfinite(seg).any() else np.nan




def _stance_height_ref(hs, hd, hh, seg, pcfg, f):
    """Per batsman track: the hand height this batter usually holds when still (their stance -
    some stand with hands low, some with the bat raised). Falls back to the whole video."""
    from scipy.ndimage import uniform_filter1d
    still = (uniform_filter1d(hs + hd, max(3, f(pcfg.stance_still_s))) < pcfg.stance_still_max) & np.isfinite(hh)
    glob = float(np.median(hh[still])) if still.sum() else 0.0
    ref = np.full(len(hh), glob)
    if seg is not None:
        for t in np.unique(seg[seg >= 0]):
            m = seg == t
            if (m & still).sum() >= f(1.0):
                ref[m] = np.median(hh[m & still])
    return ref


def _stance_before(hs, hd, hh, p, pcfg, f, ref=0.0):
    """Stillest window in [p - stance_search_s, p - stance_gap_s]. A delivery is always faced from a
    still, set stance with the hands at this batter's usual stance height; bending to tap the bat,
    walking back with the bat held up, or taking guard are not."""
    w = max(3, f(pcfg.stance_still_s))
    a, b = max(0, p - f(pcfg.stance_search_s)), p - f(pcfg.stance_gap_s) - w
    best, bi = np.inf, None
    for s in range(a, b + 1):
        m = hs[s:s + w].mean() + hd[s:s + w].mean()
        if m < best:
            best, bi = m, s
    if bi is None:
        return None
    hand = np.nanmedian(hh[bi:bi + w])
    if best > pcfg.stance_still_max or not np.isfinite(hand) or abs(hand - ref) > pcfg.stance_hand_tol_tu:
        return None
    # extend the still window forward to the moment the batter starts moving (trigger / backlift)
    e = bi + w - 1
    while e + 1 < p and hs[e + 1] < pcfg.stance_move_speed and hd[e + 1] < pcfg.stance_move_speed:
        e += 1
    return int(bi), int(e)


def detect_shots(S, pcfg, bcfg=None):
    """Returns list of (lo, downswing_peak, hi, kind, (stance_lo, stance_hi), source), kind in
    {'shot', 'leave'}, source in {'body', 'ball'}.

    A shot = still stance -> hands up (backlift) -> fast hands-down downswing. Candidate = hand-speed
    peak while the hands are moving DOWN (the follow-through rise is often faster than the
    downswing, so an unsigned speed peak lands on the wrong part of the swing)."""
    fps, T = S["fps"], S["T"]
    f = lambda x: int(round(x * fps))
    from scipy.ndimage import median_filter
    hs = median_filter(_nan0(S["hand_speed"]), 3)
    hd = median_filter(_nan0(S["head_speed"]), 3)
    hh = S["hand_h_s"]
    dh = np.gradient(_nan0(hh))
    down = np.where(dh < 0, hs, 0.0)
    rise_spd = np.where(dh > 0, hs, 0.0)
    seg = S.get("seg_id")
    ref = _stance_height_ref(hs, hd, hh, seg, pcfg, f)

    cands = []
    pk, _ = find_peaks(down, height=pcfg.swing_min_speed, distance=max(1, f(0.15)))
    for p in pk:
        st = _stance_before(hs, hd, hh, p, pcfg, f, ref[p])
        if st is None:
            continue
        base = np.nanmedian(hh[st[0]:st[1] + 1])
        top = _nanargmax(hh, st[1], p, fallback=p)
        drop = hh[top] - _rng(hh, p, p + f(0.6), np.nanmin)
        hip = S["hip_mid"][max(0, p - f(1.0)):p + f(1.0), 0]
        walk = (np.nanmax(hip) - np.nanmin(hip)) / S["TU_f"][p] if np.isfinite(hip).any() else 9
        if walk > pcfg.walk_max_tu or not np.isfinite(drop):
            continue
        if hh[top] - base >= pcfg.backlift_min_rise and drop >= pcfg.shot_min_drop:
            cands.append((int(p), "shot", st, float(hs[p])))
    # leaves: from a set stance the hands go up fast and never come back down through the ball
    lk, _ = find_peaks(rise_spd, height=pcfg.swing_min_speed, distance=max(1, f(0.15)))
    for p in lk:
        st = _stance_before(hs, hd, hh, p, pcfg, f, ref[p])
        if st is None:
            continue
        base = np.nanmedian(hh[st[0]:st[1] + 1])
        up = _rng(hh, p, p + f(1.0), np.nanmax) - base
        after = _rng(hh, p, p + f(1.2), np.nanmin) - base
        if up >= pcfg.leave_min_rise and after > -0.1 and not any(abs(c[0] - p) < f(1.5) for c in cands):
            cands.append((int(p), "leave", st, float(hs[p])))
    for c in range(len(cands)):
        cands[c] = cands[c] + ("body",)

    # ball-confirmed path (optional, confirmation only): a softer downswing with no clear set stance
    # counts only when a moving ball is seen near the bat around it. Never replaces a body-rule shot.
    ball_ev = S.get("ball_ev")
    if bcfg is not None and bcfg.enabled and ball_ev is not None:
        relaxed = []
        pk2, _ = find_peaks(down, height=bcfg.relaxed_swing_min_speed, distance=max(1, f(0.15)))
        for p in pk2:
            if any(abs(c[0] - p) < f(pcfg.shot_min_separation_s) for c in cands):
                continue
            n_ball = int(ball_ev[max(0, p - f(bcfg.window_pre_s)):p + f(bcfg.window_post_s) + 1].sum())
            if n_ball < bcfg.confirm_min_frames:
                continue
            drop = _rng(hh, max(0, p - f(0.6)), p, np.nanmax) - _rng(hh, p, p + f(0.6), np.nanmin)
            hip = S["hip_mid"][max(0, p - f(1.0)):p + f(1.0), 0]
            walk = (np.nanmax(hip) - np.nanmin(hip)) / S["TU_f"][p] if np.isfinite(hip).any() else 9
            if walk > pcfg.walk_max_tu or not np.isfinite(drop) or drop < pcfg.shot_min_drop:
                continue
            # stance reference for the KPIs: stillest window before, without the stillness requirement
            w = max(3, f(pcfg.stance_still_s))
            a0, b0 = max(0, p - f(pcfg.stance_search_s)), max(0, p - f(pcfg.stance_gap_s) - w)
            m = [hs[s:s + w].mean() + hd[s:s + w].mean() for s in range(a0, b0 + 1)]
            s0 = a0 + int(np.argmin(m)) if m else a0
            relaxed.append((int(p), "shot", (int(s0), int(min(p - 1, s0 + w - 1))), float(hs[p]), "ball", n_ball))
        # one per delivery: the earliest qualifying downswing wins (same rule as the body path)
        relaxed.sort(key=lambda c: c[0])
        for c in relaxed:
            if all(abs(c[0] - k[0]) >= f(pcfg.shot_min_separation_s) for k in cands):
                cands.append(c[:5])

    # one delivery per stance: candidates sharing a stance, or too close together, are the same ball
    # (a shot beats a leave; then the earliest downswing after the stance wins)
    cands.sort(key=lambda c: (c[1] != "shot", c[0]))
    kept = []
    for c in cands:
        if all(abs(c[0] - k[0]) >= f(pcfg.shot_min_separation_s) and
               (abs(c[2][1] - k[2][1]) > f(0.3) or "ball" in (c[4], k[4])) for k in kept):
            kept.append(c)
    kept.sort()
    shots = []
    for n, (p, kind, st, _, src) in enumerate(kept):
        lo, hi = max(0, st[0]), min(T - 1, p + f(pcfg.shot_post_s))
        if n < len(kept) - 1:
            hi = min(hi, kept[n + 1][2][0] - 1)
        if seg is not None and seg[p] >= 0:   # stay inside this batsman's track
            same = np.where(seg == seg[p])[0]
            lo, hi = max(lo, same[0]), min(hi, same[-1])
        shots.append((int(lo), int(p), int(hi), kind, st, src))
    return shots




def detect_phases(S, shot, pcfg, audio_onsets, hand_override="auto", off_sign=None):
    """audio_onsets: list of (frame_index, z). off_sign: +1 if the batsman's off side is image-right."""
    fps = S["fps"]
    lo, p, hi, kind = shot[:4]
    TU = float(S["TU_f"][p])
    f = lambda s: int(round(s * fps))
    ev, info = {}, {"kind": kind}
    hh = S["hand_h_s"]

    # ---- stance: the still, set position the ball is faced from (found by detect_shots)
    stance = shot[4] if len(shot) > 4 and shot[4] is not None else (lo, min(p - 1, lo + f(0.4)))
    info["stance_ref"] = stance
    base = np.nanmedian(hh[stance[0]:stance[1] + 1])

    # ---- top of backlift: highest hands between the end of the stance and the downswing
    top = _nanargmax(hh, stance[1], p, fallback=max(lo, p - f(0.4)))
    if kind == "leave":
        top = _nanargmax(hh, p, min(hi, p + f(1.2)), fallback=p)   # hands up = 'shoulder arms'
    ev["top_of_backlift"] = top
    peak_h = hh[top]
    if not np.isfinite(base):
        base = np.nanmin(hh[lo:top + 1]) if top > lo else peak_h
    thr = base + pcfg.backlift_start_frac * (peak_h - base)
    bl = top
    while bl > stance[1] and np.isfinite(hh[bl - 1]) and hh[bl - 1] > thr:
        bl -= 1
    ev["backlift_start"] = int(max(stance[1], min(bl, top - 1)))
    info["backlift_rise_tu"] = float(peak_h - base) if np.isfinite(peak_h - base) else None

    # ---- impact: the head is lowest when the batter is over the ball at contact. This works for
    #      vertical AND horizontal bat shots (hand height does not: on a cut/pull the hands stay high).
    #      Checked frame-by-frame on labelled clips (labels/, tools/evaluate.py). Refined by the
    #      bat-ball crack (audio) when it agrees.
    if kind == "shot":
        w1 = min(hi, p + f(pcfg.impact_search_s))
        k = _nanargmin(S["head_h_s"], p, w1, fallback=None)
        k_method = "head_lowest"
        seg_b = hh[p:w1 + 1]
        bottom = p + int(np.nanargmin(seg_b)) if len(seg_b) and np.isfinite(seg_b).any() else p
        if k is None:   # head keypoints missing -> hands' lowest point in the downswing
            k, k_method = bottom, "hand_lowest"
        drop = hh[top] - hh[bottom]
    else:
        bottom = k = p
        k_method = "leave_moment"
    ev["swing_bottom"] = int(bottom)
    cands = {"kinematic": int(k)}
    i0, i1 = max(lo, k - f(pcfg.audio_agree_s)), min(hi, k + f(pcfg.audio_agree_s))
    a_best = None
    for fr, z in audio_onsets:
        if i0 <= fr <= i1:
            w = z * np.exp(-abs(fr - k) / fps / 0.15)
            if a_best is None or w > a_best[2]:
                a_best = (fr, z, w)
    if a_best:
        cands["audio"] = int(a_best[0])
    if kind == "leave":
        imp, method, conf = k, k_method, "low"
    elif "audio" in cands:
        imp, method, conf = cands["audio"], "audio+kinematic", "high"
    else:
        imp, method, conf = k, k_method, "medium" if k_method == "head_lowest" else "low"
    imp = int(max(imp, top + 1)) if kind == "shot" else int(imp)
    ev["impact"] = imp
    # ---- ball confirmation (never moves the impact frame): a live ball near the bat around impact
    info["source"] = shot[5] if len(shot) > 5 else "body"
    ball_ev = S.get("ball_ev")
    if ball_ev is not None and kind == "shot":
        w = f(0.4)
        nb = int(ball_ev[max(0, imp - w):imp + w + 1].sum())
        info["ball_frames"] = nb
        info["ball_confirmed"] = bool(nb >= S.get("ball_min_frames", 3) or info["source"] == "ball")
        if info["ball_confirmed"] and conf != "high":
            method, conf = method + "+ball", "high"
    info["impact_method"], info["impact_confidence"], info["impact_candidates"] = method, conf, cands

    # ---- follow-through end: hands settle (still) or start the recovery back up
    j1 = min(hi, imp + f(pcfg.ft_max_s))
    post = _nanargmax(hh, imp + 1, j1, fallback=min(j1, imp + f(0.5))) if kind == "shot" else top
    ev["follow_through_peak"] = post
    spd = np.abs(np.gradient(_nan0(hh))) * fps
    ref_spd = max(0.5, float(np.nanmax(spd[top:imp + 2]))) if imp >= top else 1.0
    # the follow-through ends once the hands have (nearly) reached their finishing height and settle
    s0 = imp
    if kind == "shot" and np.isfinite(hh[post]) and np.isfinite(hh[imp]) and hh[post] > hh[imp]:
        near = hh[imp] + 0.85 * (hh[post] - hh[imp])
        s0 = next((i for i in range(imp, post + 1) if np.isfinite(hh[i]) and hh[i] >= near), post)
    st = _still_run(spd, s0, j1, pcfg.ft_still_frac * ref_spd, max(2, f(pcfg.ft_still_s)))
    ft_end = st if st is not None else j1
    ev["follow_through_end"] = int(max(ft_end, imp + 1))
    ev["shot_start"], ev["shot_end"] = lo, hi

    # ---- footwork: which foot is the front foot, lift and plant
    s_ref = slice(stance[0], stance[1] + 1)
    disp = {}
    for side in ("l", "r"):
        a = S[f"ank_{side}"]
        st = np.nanmedian(a[s_ref], axis=0)
        disp[side] = (a[imp] - st) / TU if np.all(np.isfinite(st)) and np.all(np.isfinite(a[imp])) else np.zeros(2)
    # front-on: stepping towards the bowler moves the ankle DOWN in the image (y+)
    score = {s: np.linalg.norm(disp[s]) + max(0.0, disp[s][1]) for s in disp}
    front = max(score, key=score.get)
    back = "r" if front == "l" else "l"
    info["front_foot"] = front
    info["foot_disp_tu"] = {s: [float(v) for v in disp[s]] for s in disp}
    if hand_override in ("R", "L"):
        hand, src = hand_override, "override"
    elif off_sign:
        # bowler's-end view: a right-hander's off side (keeper, stumps, grounded bat) is image-LEFT
        hand, src = ("L" if off_sign > 0 else "R"), "layout (keeper / bat side)"
    else:
        hand, src = ("R" if front == "l" else "L"), "front-foot heuristic"
    if src != "front-foot heuristic":
        front, back = ("l", "r") if hand == "R" else ("r", "l")
    info["front_foot"] = front
    info["handedness"] = hand
    info["handedness_source"] = src

    sp = S[f"ank_{front}_speed"]
    a0 = max(lo, bl - f(0.8))
    seg = _nan0(sp[a0:imp + 1])
    ev["front_foot_lift"] = ev["front_foot_plant"] = None
    if len(seg) and seg.max() >= pcfg.foot_move_speed:
        pk = a0 + int(np.argmax(seg))
        lift = pk
        while lift > a0 and np.isfinite(sp[lift - 1]) and sp[lift - 1] > pcfg.foot_plant_speed:
            lift -= 1
        plant = _still_run(sp, pk, min(hi, imp + f(0.5)), pcfg.foot_plant_speed, 3)
        ev["front_foot_lift"] = int(lift)
        ev["front_foot_plant"] = int(plant) if plant is not None else None

    # ---- per-frame labels
    labels = np.zeros(S["T"], np.int8)
    if kind == "leave":
        labels[lo:bl] = 1
        labels[bl:ev["follow_through_end"] + 1] = 7
        labels[ev["follow_through_end"] + 1:hi + 1] = 6
        return ev, info, labels
    labels[lo:bl] = 1
    labels[bl:top] = 2
    labels[top:imp] = 3
    labels[imp:ev["follow_through_end"] + 1] = 5
    labels[max(lo, imp - 1):imp + 2] = 4
    labels[ev["follow_through_end"] + 1:hi + 1] = 6
    return ev, info, labels


def analyse_phases(S, pcfg, audio_y, sr, video_fps, hand_override="auto", off_sign_fn=None, bcfg=None):
    ons = audio_mod.onsets(audio_y, sr, z_min=pcfg.audio_z_min) if audio_y is not None else []
    dur = S["T"] / video_fps
    audio_state = "none" if audio_y is None else "used"
    if ons and len(ons) / max(dur, 1e-6) > pcfg.audio_max_rate:
        audio_state = f"ignored: {len(ons) / dur:.1f} onsets/s (noisy - other nets / chatter)"
        ons = []
    S["audio_state"] = audio_state
    ons_f = [(int(round(t * video_fps)), z) for t, z in ons]       # audio runs on the video clock
    shots = detect_shots(S, pcfg, bcfg)
    labels = np.zeros(S["T"], np.int8)
    results = []
    for sh in shots:
        off = off_sign_fn(sh) if off_sign_fn else None
        ev, info, lab = detect_phases(S, sh, pcfg, ons_f, hand_override, off)
        m = lab > 0
        labels[m] = lab[m]
        results.append({"events": ev, "info": info})
    return results, labels, [(f / video_fps, z) for f, z in ons_f]
