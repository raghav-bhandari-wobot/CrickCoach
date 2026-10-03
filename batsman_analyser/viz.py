"""OpenCV overlay, interactive player, and debug figures."""
import cv2
import numpy as np

from .phases import PHASES, PHASE_COLORS

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
EVENT_SHORT = {"backlift_start": "BL", "top_of_backlift": "TOP", "front_foot_plant": "FFP",
               "impact": "IMP", "follow_through_end": "FTE"}
STATUS_COL = {"good": (90, 200, 90), "warn": (40, 190, 240), "bad": (60, 60, 235),
              "info": (220, 220, 220), "n/a": (140, 140, 140)}


def _p(x):
    return None if x is None or not np.all(np.isfinite(x)) else (int(x[0]), int(x[1]))


def _text(img, s, org, scale=0.55, col=(255, 255, 255), th=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1, alpha=0.55):
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (sub * (1 - alpha)).astype(np.uint8)


class Renderer:
    def __init__(self, S, labels, shots, meta):
        self.S, self.labels, self.shots, self.meta = S, labels, shots, meta
        self.shot_of = np.full(S["T"], -1)
        for n, sh in enumerate(shots):
            e = sh["events"]
            self.shot_of[e["shot_start"]:e["shot_end"] + 1] = n

    def render(self, frame, i):
        S = self.S
        self.k = k = max(1.0, 900.0 / frame.shape[1])      # upscale small/portrait videos for legibility
        img = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR) if k > 1 else frame.copy()
        H, W = img.shape[:2]
        s = max(1.0, min(H, W) / 720.0)
        lab = PHASES[self.labels[i]] if i < S["T"] else "-"
        col = PHASE_COLORS[lab]
        n = self.shot_of[i] if i < S["T"] else -1
        if i < S["T"]:
            self._draw_body(img, i, col, s)
        # --- top banner
        _panel(img, 0, 0, W, int(72 * s))
        _text(img, lab, (int(12 * s), int(31 * s)), 0.9 * s, col, max(2, int(2 * s)))
        info = f"shot {n + 1}/{len(self.shots)}" if n >= 0 else "no shot"
        vf = S.get("video_fps", S["fps"])
        _text(img, f"{info}  f{i}  {i / vf:.2f}s", (int(12 * s), int(62 * s)), 0.5 * s)
        # --- live panel
        if i < S["T"]:
            self._live_panel(img, i, n, s)
        # --- timeline
        self._timeline(img, i, s)
        return img

    def _P(self, x):
        return _p(np.asarray(x) * self.k)

    def _draw_body(self, img, i, col, s):
        S = self.S
        k = self.k
        kp = S["kp"][i]
        th = max(2, int(2 * s))
        # bat mask + axis
        poly = S.get("bat_poly")
        if poly is not None and S.get("bat_conf") is not None and S["bat_conf"][i] > 0:
            pts = (poly[i][np.isfinite(poly[i][:, 0])] * k).astype(np.int32)
            if len(pts) >= 3:
                ov = img.copy()
                cv2.fillPoly(ov, [pts], (0, 200, 255))
                cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        h, t = self._P(S["bat_handle"][i]), self._P(S["bat_toe"][i])
        if h and t:
            cv2.line(img, h, t, (0, 220, 255), th + 1, cv2.LINE_AA)
            cv2.circle(img, t, int(5 * s), (0, 0, 255), -1)
        # skeleton
        for a, b in EDGES:
            pa, pb = self._P(kp[a]), self._P(kp[b])
            if pa and pb:
                cv2.line(img, pa, pb, col, th, cv2.LINE_AA)
        for j in range(17):
            pj = self._P(kp[j])
            if pj:
                cv2.circle(img, pj, int(3 * s), (255, 255, 255), -1)
        # head trail (last 1 s) + stance head line
        lo = max(0, i - int(S["fps"]))
        tr = S["head"][lo:i + 1]
        tr = (tr[np.isfinite(tr).all(1)] * k).astype(np.int32)
        if len(tr) > 1:
            cv2.polylines(img, [tr], False, (255, 0, 255), max(1, int(2 * s)), cv2.LINE_AA)
        # live ball (only the detections used as confirmation: moving, near the bat, not helmet/glove)
        bxy = S.get("ball_xy")
        if bxy is not None:
            b = self._P(bxy[i])
            if b:
                cv2.circle(img, b, int(10 * s), (0, 255, 0), max(1, int(2 * s)), cv2.LINE_AA)
        n = self.shot_of[i]
        if n >= 0:
            st0, st1 = self.shots[n]["info"]["stance_ref"]
            ref = np.nanmedian(S["head"][st0:st1 + 1], axis=0) * k
            tu = S["TU_f"][i] * k
            if np.isfinite(ref).all():
                x = int(ref[0])
                for y in range(int(ref[1] - 1.5 * tu), int(ref[1] + 3 * tu), int(14 * s)):
                    cv2.line(img, (x, y), (x, y + int(7 * s)), (255, 0, 255), 1)

    def _live_panel(self, img, i, n, s):
        S = self.S
        H, W = img.shape[:2]
        pw = int(min(330 * s, 0.62 * W))
        x0 = W - pw
        rows = [("hand height", S["hand_h"][i], "TU"), ("hand speed", S["hand_speed"][i], "TU/s"),
                ("head speed", S["head_speed"][i], "TU/s"),
                ("bat angle", S["bat_angle"][i] if S["bat_valid"][i] else np.nan, "deg")]
        kp_rows = []
        if n >= 0 and i >= self.shots[n]["events"]["impact"]:
            K = self.shots[n]["kpis"]
            for key in ("shot_type", "head_max_drift_downswing", "head_over_front_foot",
                        "front_foot_plant_before_impact", "bat_angle_at_impact", "balance_at_impact",
                        "downswing_duration", "finish_stillness"):
                if key in K:
                    kp_rows.append(K[key])
        h = int((40 + 22 * len(rows) + 22 * len(kp_rows) + (30 if kp_rows else 0)) * s)
        y0 = int(80 * s)
        _panel(img, x0, y0, W, y0 + h)
        y = y0 + int(22 * s)
        _text(img, "LIVE", (x0 + int(10 * s), y), 0.5 * s, (200, 200, 200))
        for name, v, u in rows:
            y += int(22 * s)
            vs = "--" if not np.isfinite(v) else f"{v:6.2f}"
            _text(img, f"{name:<12}{vs} {u}", (x0 + int(10 * s), y), 0.5 * s)
        if kp_rows:
            y += int(30 * s)
            _text(img, "SHOT KPIs", (x0 + int(10 * s), y), 0.5 * s, (200, 200, 200))
            for k in kp_rows:
                y += int(22 * s)
                v = k["value"]
                vs = v if isinstance(v, str) else ("--" if v is None else f"{v}")
                name = k["name"][:24]
                _text(img, f"{name}: {vs}", (x0 + int(10 * s), y), 0.43 * s, STATUS_COL.get(k["status"], (255, 255, 255)))

    def _timeline(self, img, i, s):
        S = self.S
        H, W = img.shape[:2]
        th = int(18 * s)
        y0 = H - th - int(8 * s)
        T = S["T"]
        xs = (np.arange(T + 1) * W / T).astype(int)
        for f in range(T):
            cv2.rectangle(img, (xs[f], y0), (xs[f + 1], y0 + th), PHASE_COLORS[PHASES[self.labels[f]]], -1)
        for sh in self.shots:
            for k, short in EVENT_SHORT.items():
                f = sh["events"].get(k)
                if f is None:
                    continue
                x = int(f * W / T)
                cv2.line(img, (x, y0 - int(6 * s)), (x, y0 + th), (255, 255, 255), 1)
                if k == "impact":
                    _text(img, short, (x - int(12 * s), y0 - int(8 * s)), 0.4 * s)
        x = int(i * W / T)
        cv2.line(img, (x, y0 - int(10 * s)), (x, y0 + th + int(4 * s)), (0, 0, 0), max(3, int(3 * s)))
        cv2.line(img, (x, y0 - int(10 * s)), (x, y0 + th + int(4 * s)), (255, 255, 255), max(1, int(s)))


def write_video(path_in, path_out, renderer, meta, max_w=1280):
    cap = cv2.VideoCapture(str(path_in))
    vw, size, i = None, None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        img = renderer.render(fr, i)
        if vw is None:
            h, w = img.shape[:2]
            sc = min(1.0, max_w / w, 1600 / h)
            size = (int(w * sc) // 2 * 2, int(h * sc) // 2 * 2)
            vw = cv2.VideoWriter(str(path_out), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], size)
        vw.write(cv2.resize(img, size))
        i += 1
    if vw is None:
        cap.release()
        return
    cap.release()
    vw.release()


def play(path_in, renderer, meta, max_w=1280):
    """Keys: space pause | d / a step (paused) | s slow-mo | n next impact | r restart | q quit."""
    cap = cv2.VideoCapture(str(path_in))
    T = meta["n_frames"]
    impacts = sorted(sh["events"]["impact"] for sh in renderer.shots)
    i, paused, slow = 0, False, False
    win = "Batsman Analyser"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame = None
    need_read = True
    while True:
        if need_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                i, paused = max(0, T - 1), True
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    break
        out = renderer.render(frame, i)
        sc = min(1.0, max_w / out.shape[1], 900 / out.shape[0])
        out = cv2.resize(out, None, fx=sc, fy=sc)
        _text(out, "[space] pause [a/d] step [s] slow [n] next impact [r] restart [q] quit",
              (10, out.shape[0] - 40), 0.45)
        cv2.imshow(win, out)
        delay = int(1000 / meta["fps"] * (4 if slow else 1))
        k = cv2.waitKey(0 if paused else max(1, delay)) & 0xFF
        need_read = True
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
            need_read = False
        elif k == ord("s"):
            slow = not slow
            need_read = False
        elif k == ord("d"):
            i = min(T - 1, i + 1)
        elif k == ord("a"):
            i = max(0, i - 1)
        elif k == ord("r"):
            i = 0
        elif k == ord("n") and impacts:
            nxt = [f for f in impacts if f > i + 1]
            i, paused = (nxt[0] if nxt else impacts[0]), True
        elif not paused:
            i += 1
            if i >= T:
                i, paused = T - 1, True
        else:
            need_read = False
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def keyframes(path_in, renderer, shots, path_out, thumb_h=360):
    """One row per shot: annotated frames at each detected event (for eyeball verification)."""
    keys = ["backlift_start", "top_of_backlift", "front_foot_plant", "impact", "follow_through_end"]
    cap = cv2.VideoCapture(str(path_in))
    rows = []
    for n, sh in enumerate(shots):
        tiles = []
        for k in keys:
            f = sh["events"].get(k)
            if f is None:
                tile = np.zeros((thumb_h, int(thumb_h * 16 / 9), 3), np.uint8)
                _text(tile, f"{k}: n/a", (10, 30), 0.6)
                tiles.append(tile)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            img = renderer.render(fr, f)
            tile = cv2.resize(img, (int(img.shape[1] * thumb_h / img.shape[0]), thumb_h))
            _panel(tile, 0, thumb_h - 34, tile.shape[1], thumb_h, 0.7)
            _text(tile, f"{k}  (frame {f})", (8, thumb_h - 10), 0.6, (0, 255, 255), 1)
            tiles.append(tile)
        if tiles:
            w = min(t.shape[1] for t in tiles)
            rows.append(np.hstack([t[:, :w] for t in tiles]))
    cap.release()
    if rows:
        w = min(r.shape[1] for r in rows)
        rows = [r[:, :w] for r in rows]
        path_out = str(path_out)
        for k in range(0, len(rows), 8):   # 8 shots per image so long sessions stay viewable
            name = path_out if k == 0 else path_out.replace(".jpg", f"_{k // 8 + 1}.jpg")
            cv2.imwrite(name, np.vstack(rows[k:k + 8]))


def signals_plot(S, labels, shots, onsets, path_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(S["T"]) / S["fps"]
    fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    series = [
        ("Hand height above hips (TU)", [("hands", S["hand_h"]), ("bat toe", S["bat_toe_h"])]),
        ("Speed (TU/s)", [("hands", S["hand_speed"]), ("bat toe", S["bat_toe_speed"])]),
        ("Head (TU from start)", [("x", (S["head"][:, 0] - np.nanmedian(S["head"][:, 0])) / S["TU"]),
                                  ("y", (S["head"][:, 1] - np.nanmedian(S["head"][:, 1])) / S["TU"])]),
        ("Ankle speed (TU/s)", [("left", S["ank_l_speed"]), ("right", S["ank_r_speed"])]),
        ("Bat angle from vertical (deg)", [("bat", np.where(S["bat_valid"], S["bat_angle"], np.nan))]),
    ]
    for a, (title, ss) in zip(ax, series):
        for name, y in ss:
            a.plot(t, y, label=name, lw=1.2)
        a.set_title(title, fontsize=10, loc="left")
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
        # phase shading
        cur, start = labels[0], 0
        for f in range(1, S["T"] + 1):
            if f == S["T"] or labels[f] != cur:
                if cur > 0:
                    c = np.array(PHASE_COLORS[PHASES[cur]][::-1]) / 255.0
                    a.axvspan(start / S["fps"], f / S["fps"], color=c, alpha=0.15, lw=0)
                if f < S["T"]:
                    cur, start = labels[f], f
        for sh in shots:
            a.axvline(sh["events"]["impact"] / S["fps"], color="red", ls="--", lw=1)
        for tt, z in onsets:
            ax[1].axvline(tt, color="k", ls=":", lw=0.8)
    ax[-1].set_xlabel("time (s)   | red dashed = impact, dotted = audio onsets")
    fig.tight_layout()
    fig.savefig(path_out, dpi=90)
    plt.close(fig)
