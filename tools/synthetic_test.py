"""Self-test of tracking -> phases -> KPIs -> viz on a synthetic front-on drive with known events.

    python tools/synthetic_test.py      (no models / video needed)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batsman_analyser.config import Config  # noqa: E402
from batsman_analyser.feedback import coaching_feedback  # noqa: E402
from batsman_analyser.kpis import compute_kpis  # noqa: E402
from batsman_analyser.phases import analyse_phases  # noqa: E402
from batsman_analyser.signals import build_signals  # noqa: E402
from batsman_analyser.tracking import select_batsman, track  # noqa: E402

FPS, T = 30.0, 150
GT = {"backlift_start": 45, "top_of_backlift": 70, "front_foot_plant": 74, "impact": 78, "follow_through_end": 96}


def person(cx, cy, s, t):
    """Right-hander, front-on. Returns (17,3) keypoints for frame t."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    g = GT
    # hand height profile (px above hip): stance 5 (hands resting on the bat at the hips), top 110,
    # impact -20 (fast downswing, ~0.25 s like the labelled clips), FT 180, then back to the stance
    def lerp(a, b, u):
        return a + (b - a) * np.clip(u, 0, 1)
    if t < g["backlift_start"]:
        hh = 5
    elif t < g["top_of_backlift"]:
        hh = lerp(5, 110, (t - g["backlift_start"]) / (g["top_of_backlift"] - g["backlift_start"]))
    elif t < g["impact"]:
        hh = lerp(110, -20, ((t - g["top_of_backlift"]) / (g["impact"] - g["top_of_backlift"])) ** 1.5)
    elif t < g["follow_through_end"] - 4:
        hh = lerp(-20, 180, (t - g["impact"]) / (g["follow_through_end"] - 4 - g["impact"]))
    elif t < g["follow_through_end"] + 16:
        hh = 180
    else:                                                              # walks back to the stance
        hh = lerp(180, 5, (t - g["follow_through_end"] - 16) / 12)
    stride = lerp(0, 40, (t - 62) / (g["front_foot_plant"] - 62))     # front (left) foot comes toward camera
    head_dx = lerp(0, -18, (t - 60) / 25)                               # head drifts towards off side (image left)
    head_dy = lerp(0, 0.5 * s, (t - 64) / (g["impact"] - 64)) - lerp(0, 0.5 * s, (t - 90) / 20)  # gets down to the ball, then up
    hip = np.array([cx, cy])
    sh = hip - [0, s] + [0, head_dy * 0.6]
    k[11, :2], k[12, :2] = hip + [-10, 0], hip + [10, 0]
    k[5, :2], k[6, :2] = sh + [-12, 0], sh + [12, 0]
    k[0, :2] = sh + [head_dx, -0.45 * s + head_dy]
    k[1, :2], k[2, :2] = k[0, :2] + [-5, -4], k[0, :2] + [5, -4]
    k[3, :2], k[4, :2] = k[0, :2] + [-10, 0], k[0, :2] + [10, 0]
    hand = hip + [15, -hh]
    k[9, :2], k[10, :2] = hand + [-3, 0], hand + [3, 3]
    k[7, :2], k[8, :2] = (k[5, :2] + hand) / 2, (k[6, :2] + hand) / 2
    k[13, :2], k[14, :2] = hip + [-15, 0.8 * s + stride / 2], hip + [15, 0.8 * s]
    k[15, :2], k[16, :2] = hip + [-18, 1.6 * s + stride], hip + [18, 1.6 * s]
    return k


def main():
    rng = np.random.default_rng(0)
    pf, pb, pc, pk = [], [], [], []
    for t in range(T):
        k = person(640, 400, 100, t)
        k[:, :2] += rng.normal(0, 1.0, (17, 2))
        pf.append(t); pk.append(k); pc.append(0.9)
        pb.append([k[:, 0].min() - 20, k[:, 1].min() - 20, k[:, 0].max() + 20, k[:, 1].max() + 20])
        # bowler running through frame (should NOT be selected)
        if 30 <= t < 75:
            bx = 100 + (t - 30) * 25
            kb = person(bx, 450, 140, 0)
            pf.append(t); pk.append(kb); pc.append(0.9)
            pb.append([kb[:, 0].min() - 20, kb[:, 1].min() - 20, kb[:, 0].max() + 20, kb[:, 1].max() + 20])
    det = {"p_frame": np.array(pf), "p_box": np.array(pb, np.float32), "p_conf": np.array(pc, np.float32),
           "p_kpt": np.array(pk, np.float32)}
    meta = {"fps": FPS, "n_frames": T, "width": 1280, "height": 720}
    cfg = Config()
    tid = track(det, T, cfg.track)
    per_frame, best, _ = select_batsman(det, tid, T, FPS, cfg.track)
    assert (det["p_box"][per_frame[per_frame >= 0], 0] > 400).all(), "picked the bowler!"
    # synthetic bat: from hands, toe 160px, rotating
    bat = {k: None for k in ()}
    hands = det["p_kpt"][per_frame, 9, :2]
    ang = np.radians(np.where(np.arange(T) < GT["impact"], 20, 5))
    toe = hands + 160 * np.c_[np.sin(ang), np.cos(ang)]
    bat = {"bat_conf": np.full(T, 0.5, np.float32), "bat_a": hands.astype(np.float32),
           "bat_b": toe.astype(np.float32), "bat_thin": np.full(T, 0.1), "bat_poly": np.full((T, 48, 2), np.nan)}
    S = build_signals(det, per_frame, bat, meta, cfg.smooth)
    # audio: silence + click at impact
    sr = 22050
    y = rng.normal(0, 0.002, int(T / FPS * sr)).astype(np.float32)
    c = int(GT["impact"] / FPS * sr)
    y[c:c + 200] += rng.normal(0, 0.6, 200).astype(np.float32)
    shots, labels, ons = analyse_phases(S, cfg.phase, y, sr, FPS)
    assert len(shots) == 1, f"expected 1 shot, got {len(shots)}"
    ev = shots[0]["events"]
    print("event                 GT   detected  err(frames)")
    ok = True
    for k, v in GT.items():
        d = ev.get(k)
        e = None if d is None else d - v
        print(f"{k:<20}{v:>5}{str(d):>10}{str(e):>10}")
        ok &= e is not None and abs(e) <= (6 if k == "backlift_start" else 4)
    print("impact method:", shots[0]["info"]["impact_method"], "| hand:", shots[0]["info"]["handedness"])
    K = compute_kpis(S, ev, shots[0]["info"], cfg.kpi)
    for key in ("head_drift_at_impact", "front_foot_plant_before_impact", "bat_angle_at_impact", "shot_type"):
        print(f"  {key}: {K[key]['value']} {K[key]['unit']} [{K[key]['status']}]")
    print(coaching_feedback(K))
    # render a few frames to make sure viz runs
    from batsman_analyser.viz import Renderer, signals_plot
    shots[0]["kpis"] = K
    r = Renderer(S, labels, shots, meta)
    for i in (0, 60, 82, 120):
        r.render(np.zeros((720, 1280, 3), np.uint8), i)
    Path("out").mkdir(exist_ok=True)
    signals_plot(S, labels, shots, ons, "out/synthetic_signals.png")
    print("PASS" if ok else "FAIL (event error > 4 frames)")


if __name__ == "__main__":
    main()
