"""Simple IoU tracker + batsman selection (who is swinging, not running, and always there)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

LW, RW, LS, RS, LH, RH = 9, 10, 5, 6, 11, 12


def iou_matrix(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def track(det, n_frames, tcfg, cuts=()):
    """Returns det_track: (M,) track id per detection. Tracks never continue across a scene cut."""
    cuts = set(int(c) for c in cuts)
    pf, pb = det["p_frame"], det["p_box"]
    tid = np.full(len(pf), -1, np.int32)
    active = {}  # id -> (last_box, last_frame)
    next_id = 0
    order = np.argsort(pf, kind="stable")
    starts = np.searchsorted(pf[order], np.arange(n_frames + 1))
    for f in range(n_frames):
        idx = order[starts[f]:starts[f + 1]]
        if f in cuts:
            active = {}
        active = {k: v for k, v in active.items() if f - v[1] <= tcfg.max_lost}
        ids = list(active.keys())
        if len(idx) and ids:
            prev = np.array([active[k][0] for k in ids])
            iou = iou_matrix(pb[idx], prev)
            r, c = linear_sum_assignment(-iou)
            for ri, ci in zip(r, c):
                if iou[ri, ci] >= tcfg.iou_match:
                    tid[idx[ri]] = ids[ci]
        for d in idx:
            if tid[d] < 0:
                tid[d] = next_id
                next_id += 1
            active[int(tid[d])] = (pb[d], f)
    return tid


def track_features(det, tid, fps, frame_hw=None):
    pf, pb, pk = det["p_frame"], det["p_box"], det["p_kpt"]
    feats = {}
    for t in np.unique(tid):
        m = np.where(tid == t)[0]
        if len(m) < 5:
            continue
        h = float(np.median(pb[m, 3] - pb[m, 1]))
        ctr = np.c_[(pb[m, 0] + pb[m, 2]) / 2, (pb[m, 1] + pb[m, 3]) / 2]
        travel = float(np.ptp(ctr[:, 0]) / h)                         # bowlers / runners travel
        hip = (pk[m, LH, :2] + pk[m, RH, :2]) / 2
        wr = (pk[m, LW, :2] + pk[m, RW, :2]) / 2 - hip
        activity = float((np.std(wr[:, 1]) + np.std(wr[:, 0])) / h)   # swinging hands
        edge = size = 1.0
        if frame_hw is not None:
            H, W = frame_hw
            touch = (pb[m, 0] < 3) | (pb[m, 2] > W - 3) | (pb[m, 3] > H - 3)
            edge = 1.0 - 0.8 * float(touch.mean())                    # cut-off bodies near the camera
            size = 1.0 if h < 0.4 * H else 0.3                        # huge = bowler/umpire at camera end
        dur = min(1.0, len(m) / (1.5 * fps))
        score = (activity + 0.02) * np.sqrt(h) / (1.0 + 2.0 * max(0.0, travel - 0.5)) * edge * size * dur
        feats[int(t)] = {"n": len(m), "span": (int(pf[m].min()), int(pf[m].max())), "h": h,
                         "travel": travel, "activity": activity, "score": float(score)}
    return feats


def select_batsman(det, tid, n_frames, fps, tcfg, frame_hw=None):
    """Batsman per frame. Several batsmen may appear one after another (net sessions), so tracks are
    accepted greedily by score as long as they don't overlap in time with an already accepted track
    (the keeper / umpire / bowler are concurrent with the batsman and lose)."""
    pf = det["p_frame"]
    feats = track_features(det, tid, fps, frame_hw)
    if not feats:
        raise RuntimeError("Could not find a consistent person to analyse.")
    order = sorted(feats, key=lambda t: feats[t]["score"] * np.sqrt(feats[t]["n"]), reverse=True)
    best = order[0]
    occupied = np.zeros(n_frames, bool)
    chosen = []
    for t in order:
        f = feats[t]
        if f["score"] < tcfg.min_score_frac * feats[best]["score"]:
            continue
        a, z = f["span"]
        frames = pf[tid == t]
        if occupied[frames].mean() > 0.1:
            continue
        chosen.append(t)
        occupied[a:z + 1] = True
    per_frame = np.full(n_frames, -1, np.int64)
    for d in np.where(np.isin(tid, chosen))[0]:
        f = pf[d]
        if per_frame[f] < 0 or det["p_conf"][d] > det["p_conf"][per_frame[f]]:
            per_frame[f] = d
    segs = sorted((feats[t]["span"], t) for t in chosen)
    return per_frame, best, {"tracks": feats, "chosen": [t for _, t in segs]}
