"""KPIs for one shot (front-on camera). Every KPI states how much to trust it from this view."""
import numpy as np


def _f(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _status_max(v, good_warn):
    """Smaller is better."""
    if v is None:
        return "n/a"
    g, w = good_warn
    return "good" if abs(v) <= g else ("warn" if abs(v) <= w else "bad")


def _k(name, value, unit, reliability, status="info", note=""):
    return {"name": name, "value": value, "unit": unit, "reliability": reliability,
            "status": status, "note": note}


def _med(x, sl):
    seg = x[sl]
    return np.nanmedian(seg, axis=0) if np.isfinite(seg).any() else np.full(x.shape[1:], np.nan)


def compute_kpis(S, ev, info, kcfg):
    fps = S["fps"]
    TU = float(S["TU_f"][ev["impact"]])
    ms = lambda a, b: None if a is None or b is None else round((b - a) / fps * 1000.0)
    st0, st1 = info["stance_ref"]
    s_ref = slice(st0, st1 + 1)
    top, imp, ft_end, post = ev["top_of_backlift"], ev["impact"], ev["follow_through_end"], ev["follow_through_peak"]
    bl = ev["backlift_start"]
    hand = info["handedness"]
    off = -1.0 if hand == "R" else 1.0            # bowler's-end view: RHB off side is image-left
    front, back = info["front_foot"], ("r" if info["front_foot"] == "l" else "l")
    fa, ba = S[f"ank_{front}"], S[f"ank_{back}"]
    head, hip, hands = S["head"], S["hip_mid"], S["hands"]
    K = {}

    # ---------------- HEAD
    h_ref = _med(head, s_ref)
    dx_imp = off * (head[imp, 0] - h_ref[0]) / TU
    K["head_drift_at_impact"] = _k("Head lateral drift (stance->impact)", _f(dx_imp), "TU (+ = towards off side)",
                                   "high", _status_max(_f(dx_imp), kcfg.head_drift_tu))
    seg = head[top:imp + 1, 0]
    mx = np.nanmax(np.abs(seg - h_ref[0])) / TU if np.isfinite(seg).any() else np.nan
    K["head_max_drift_downswing"] = _k("Max head drift in downswing", _f(mx), "TU", "high",
                                       _status_max(_f(mx), kcfg.head_drift_tu))
    drop = (head[imp, 1] - h_ref[1]) / TU
    K["head_drop_at_impact"] = _k("Head drop towards ball", _f(drop), "TU (+ = lower)", "medium", "info",
                                  "Getting the head down to the line helps on front-foot shots.")
    hof = off * (head[imp, 0] - fa[imp, 0]) / TU
    K["head_over_front_foot"] = _k("Head vs front foot at impact", _f(hof), "TU (+ = head outside front foot, off side)",
                                   "high", _status_max(_f(hof), kcfg.head_over_front_foot_tu))
    et = S["eye_tilt"][imp]
    K["eye_line_tilt"] = _k("Eye-line tilt at impact", _f(abs(et), 1), "deg", "medium",
                            _status_max(_f(abs(et), 1), kcfg.eye_tilt_deg))
    path = np.nansum(np.linalg.norm(np.diff(head[top:imp + 1], axis=0), axis=1)) / TU
    K["head_path_downswing"] = _k("Head path length in downswing", _f(path), "TU", "medium", "info")

    # ---------------- FOOTWORK
    sw = np.linalg.norm(_med(S["ank_l"], s_ref) - _med(S["ank_r"], s_ref)) / TU
    g0, g1 = kcfg.stance_width_tu
    K["stance_width"] = _k("Stance width", _f(sw), "TU", "medium",
                           "n/a" if not np.isfinite(sw) else ("good" if g0 <= sw <= g1 else "warn"))
    plant = ev.get("front_foot_plant")
    d_ms = ms(plant, imp)
    if d_ms is None:
        pst = "n/a"
    else:
        pst = "good" if d_ms >= kcfg.plant_before_impact_ms[0] else (
            "warn" if d_ms >= kcfg.plant_before_impact_ms[1] else "bad")
    K["front_foot_plant_before_impact"] = _k("Front foot planted before impact by", d_ms, "ms (+ = before)",
                                             "medium", pst, "A late front foot means the bat arrives before the base is set.")
    fdx, fdy = info["foot_disp_tu"][front]
    K["front_foot_lateral"] = _k("Front foot movement across", _f(off * fdx), "TU (+ = towards off side)", "high", "info")
    K["front_foot_stride"] = _k("Front foot stride towards bowler", _f(fdy), "TU (foreshortened)", "low", "info")
    bdy = info["foot_disp_tu"][back][1]
    if fdy > 0.25 or np.hypot(fdx, fdy) > 0.35:
        foot = "front foot"
    elif bdy < -0.12:
        foot = "back foot"
    else:
        foot = "stayed in crease"
    K["footwork_type"] = _k("Footwork", foot, "", "medium", "info")

    # ---------------- TEMPO
    K["backlift_duration"] = _k("Backlift duration", ms(bl, top), "ms", "high", "info")
    K["downswing_duration"] = _k("Downswing duration", ms(top, imp), "ms", "high", "info")
    K["follow_through_duration"] = _k("Follow-through duration", ms(imp, ft_end), "ms", "high", "info")
    r = (top - bl) / max(1, imp - top)
    K["tempo_ratio"] = _k("Backlift : downswing ratio", _f(r), "x", "high", "info",
                          "Consistency across balls matters more than the absolute value.")

    # ---------------- BACKLIFT / BAT
    K["backlift_height"] = _k("Hands height at top of backlift", _f(S["hand_h"][top]), "TU above hips", "high", "info")
    ang = S["bat_angle"]
    bat_imp = ang[imp] if S["bat_valid"][imp] else np.nan
    if not np.isfinite(bat_imp):  # nearest valid within 2 frames
        for d in (1, -1, 2, -2):
            j = imp + d
            if 0 <= j < S["T"] and S["bat_valid"][j]:
                bat_imp = ang[j]
                break
    a_abs = abs(bat_imp) if np.isfinite(bat_imp) else np.nan
    bat_kind = None if not np.isfinite(a_abs) else ("vertical" if a_abs < 35 else ("horizontal" if a_abs > 55 else "angled"))
    K["bat_angle_at_impact"] = _k("Bat angle from vertical at impact", _f(a_abs, 1), "deg", "medium",
                                  _status_max(_f(a_abs, 1), kcfg.bat_vertical_deg) if bat_kind == "vertical" else "info",
                                  "Only judged as good/bad for straight-bat shots.")
    K["bat_face_type"] = _k("Bat plane", bat_kind or "bat not tracked", "", "medium", "info")
    toe = S["bat_toe"][top:imp + 1]
    if np.isfinite(toe).all(axis=1).sum() >= 3:
        dev = (np.nanmax(toe[:, 0]) - np.nanmin(toe[:, 0])) / TU
        K["bat_path_lateral"] = _k("Bat-toe lateral travel in downswing", _f(dev), "TU", "medium",
                                   _status_max(_f(dev), kcfg.bat_path_dev_tu) if bat_kind == "vertical" else "info",
                                   "Large sideways travel on a straight shot = bat coming across the line.")
        K["bat_toe_peak_speed"] = _k("Bat-toe peak speed (relative)", _f(np.nanmax(S["bat_toe_speed"][top:imp + 2]), 1),
                                     "TU/s", "low", "info")
    ta = ang[top] if S["bat_valid"][top] else np.nan
    K["backlift_bat_angle"] = _k("Bat angle at top of backlift", _f(off * ta, 1) if np.isfinite(ta) else None,
                                 "deg (+ = toe towards off side / gully)", "low", "info")

    # ---------------- ROTATION / SEQUENCING (2D proxies)
    shw, hpw = S["shoulder_w"], S["hip_w"]
    K["shoulder_openness_stance"] = _k("Shoulder openness (stance)", _f(np.nanmedian(shw[s_ref])), "TU projected width", "medium", "info")
    K["shoulder_openness_impact"] = _k("Shoulder openness (impact)", _f(shw[imp]), "TU projected width", "medium", "info",
                                       "Side-on stance looks narrow from the bowler's end; wide = chest open.")
    K["shoulder_tilt_impact"] = _k("Shoulder tilt at impact", _f(S["shoulder_tilt"][imp], 1), "deg", "high", "info")
    lo2, hi2 = top, max(top + 2, ft_end)
    hp_seg, sh_seg = np.nan_to_num(hpw[lo2:hi2 + 1]), np.nan_to_num(shw[lo2:hi2 + 1])
    if len(hp_seg) > 3 and len(sh_seg) > 3:   # window can run off the end of the clip
        dh, ds = np.gradient(hp_seg), np.gradient(sh_seg)
        lead = ms(lo2 + int(np.argmax(dh)), lo2 + int(np.argmax(ds)))
        K["hip_leads_shoulder"] = _k("Hips open before shoulders by", lead, "ms (+ = hips first)", "low",
                                     "info", "Kinematic-sequence proxy from projected widths.")

    # ---------------- BALANCE
    def balance(i):
        ax, bx = fa[i, 0], ba[i, 0]
        if not (np.isfinite(ax) and np.isfinite(bx) and np.isfinite(hip[i, 0])):
            return None
        width = max(abs(ax - bx), 0.3 * TU)
        return _f(off * (hip[i, 0] - (ax + bx) / 2) / width)
    bi, bf = balance(imp), balance(ft_end)
    K["balance_at_impact"] = _k("Hip centre vs base at impact", bi, "(0 = centred, ±0.5 = over a foot)", "high",
                                _status_max(bi, kcfg.balance_offset))
    K["balance_at_finish"] = _k("Hip centre vs base at finish", bf, "(0 = centred, ±0.5 = over a foot)", "high",
                                _status_max(bf, kcfg.balance_offset))
    e = min(S["T"] - 1, ft_end + int(0.5 * fps))
    fin = np.nansum(np.linalg.norm(np.diff(head[ft_end:e + 1], axis=0), axis=1)) / TU if e > ft_end else np.nan
    K["finish_stillness"] = _k("Head movement in 0.5 s after finish", _f(fin), "TU", "high",
                               _status_max(_f(fin), kcfg.finish_motion_tu), "Holding the finish = balance.")

    # ---------------- SHOT TYPE (heuristic)
    dirx = off * (hands[post, 0] - hip[imp, 0]) / TU if np.isfinite(hands[post, 0]) else np.nan
    side = "n/a" if not np.isfinite(dirx) else ("off side" if dirx > 0.4 else ("leg side" if dirx < -0.4 else "straight"))
    fh = S["hand_h"][post]
    if bat_kind == "horizontal":
        shot = "pull / hook / sweep" if side == "leg side" else ("cut" if side == "off side" else "cross-bat shot")
    elif foot == "front foot":
        if np.isfinite(fh) and fh < 0.6:
            shot = "forward defence"
        else:
            shot = {"off side": "cover / off drive", "leg side": "on drive / flick"}.get(side, "straight drive")
    elif foot == "back foot":
        shot = "back-foot defence / punch" if (np.isfinite(fh) and fh < 0.8) else "back-foot drive"
    else:
        shot = "defensive / leave-like" if (np.isfinite(fh) and fh < 0.6) else "shot from crease"
    K["shot_type"] = _k("Shot type (heuristic)", shot, f"{foot}, {bat_kind or 'bat ?'} bat, {side}", "low", "info")
    return K
