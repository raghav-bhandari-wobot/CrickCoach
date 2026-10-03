# Batsman Analyser — Plan (v0.1 prototype)

Scope: offline analysis of a **front-on (bowler's end)** batting video → shot phases, KPIs,
coaching cues, and an OpenCV overlay. No frontend yet.

## 1. Research summary (what already exists)

| Source | Useful idea we reuse |
|---|---|
| vmahesh-mopidevi/ai-cricket-coach-portfolio | 5 rule-based phases (Stance → Backlift → Downswing → Impact → Follow-through); front-on KPIs: head over base, stance width, shoulder/hip tilt, balance. Weights private. |
| AlexTredge8/batting-backend (BattingIQ) | **Audio-first contact detection** (bat-ball "crack") + anchoring the other events relative to contact and to stillness. |
| paulamartya25/cover_drive | 4-phase stroke model on YOLOv8-pose, 6 joint angles (elbows, knees, hip rotation, shoulder alignment). |
| Ultralytics YOLOE-26 | Open-vocabulary detection+segmentation with a text prompt ("cricket bat", "cricket ball") → zero-shot bat mask, no training needed. |
| Roboflow Universe "cricket-bat-detection" datasets | Path to a fine-tuned bat detector later (drop-in via `--bat-weights`). |

Nobody publishes a validated, open front-on phase detector → we build a rule-based one, and measure it.

## 2. Models

* **Pose:** `yolo26m-pose.pt` (17 COCO keypoints), imgsz 960 (batsman is small in front-on footage).
* **Bat:** `yoloe-26m-seg.pt` with the text prompt "cricket bat", run on a crop around the batsman. Ball detection removed for now (net sessions have many balls lying around) - a trained ball/bat model comes later.
  Bat mask → PCA axis → *handle* end (nearest the wrists) and *toe* end. Fallback: COCO-style custom weights.
* **Audio:** high-frequency spectral-flux onsets (ffmpeg via `imageio-ffmpeg`, no system install).

## 3. Batsman selection

IoU tracker over all people → score each track by
`presence × wrist activity × size ÷ (1 + body travel)` (bowler travels, umpire/keeper don't swing),
then stitch fragments broken by occlusion (bowler running through frame).

All distances are normalised by **torso length (TU)** = median shoulder-mid → hip-mid distance in the stance,
so KPIs are camera-distance independent.

## 4. Phase model

```
 STANCE ──► BACKLIFT ──► DOWNSWING ──► [IMPACT] ──► FOLLOW-THROUGH ──► FINISH
            (trigger/stride overlaps)              ▲
                        front-foot plant (event) ──┘ should happen BEFORE impact
```

Per shot (a shot = a still, set stance followed by a fast hands-DOWN downswing; multiple balls per video supported):

| Event | Rule |
|---|---|
| **stance** | stillest 0.6 s in the 2.5 s before the downswing: hand + head speed < 0.45 TU/s, hands within 0.45 TU of *this batter's* usual stance height (median still hand height of the track). Rejects bending to tap the bat, walking back with the bat up, taking guard. |
| downswing peak `p` | hand-speed peak (≥ 3 TU/s) while the hands move **down** (the follow-through rise is often faster, so unsigned speed lands on the wrong part of the swing) |
| **top of backlift** | max hand height between stance end and `p` |
| **backlift start** | walking back from the top: last frame hands are within 15 % of stance height (never before stance end) |
| **impact** | **head lowest** in `[p, p+0.6 s]` - the batter is over the ball at contact. Works for vertical and horizontal bat shots (hand height fails on cuts/pulls). Audio crack overrides when clean. Fallback: hands' lowest point. |
| **follow-through end** | hands reach 85 % of their finishing height, then first 0.15 s of stillness |
| front-foot lift / plant | burst of front-ankle speed before impact; plant = speed < 0.6 TU/s for 3 frames |
| ball (optional) | confirmation only: a live ball near the bat in >= 3 frames around the swing -> `ball_confirmed`, impact confidence high; also admits softer swings without a set stance (downswing >= 1.5 TU/s). Never creates a shot alone, never moves impact. |
| leave | from a set stance the hands go up fast and never come back down (low confidence - no ball detection) |

Measured on `Cricket Sample 29 Clip` (4 deliveries, labels/): recall 4/4, precision 4/4, impact MAE 58 ms (all ≤ 2 frames).
All labelled clips (Sample 29 x4 + YouTube clip 1, 26 balls): recall 100 %, precision 96 % with ball confirmation (the 1 false + is a slow-mo replay).
The previous rule set (engagement peaks) gave 12 detections for the same 4 balls.

Handedness: front foot = ankle that moves more toward the camera; COCO-left front foot ⇒ right-hander (override `--hand`).

## 5. KPIs (front-on)  — reliability from this view in brackets

**Head** — lateral drift stance→impact [high], max drift in downswing [high], vertical drop [med],
head over front foot at impact [high], eye-line tilt at impact [med].
**Footwork** — stance width / shoulder width [high], front-foot plant − impact timing (ms) [med],
front-foot lateral movement [high], stride toward bowler (foreshortened) [low].
**Rotation / sequencing** — projected shoulder & hip width (open/closed proxy) [med],
hip-opens-before-shoulders timing [low-med], shoulder tilt at impact [high].
**Bat** — bat angle from vertical at impact (straight vs cross bat) [med], downswing lateral path deviation [med],
backlift height [high], bat-toe peak speed (relative) [low].
**Tempo** — backlift, downswing, follow-through durations; backlift : downswing ratio [high].
**Balance** — hip-centre position inside base of support at impact & finish [high], finish stillness [high].
**Shot type (heuristic)** — vertical vs horizontal bat, front vs back foot, off / straight / leg [low-med].

Every KPI carries `value, unit, reliability, status (good/warn/bad)`; thresholds live in `config.py`
and are **initial heuristics to be tuned** against labelled videos.

## 6. Accuracy plan (how we check it)

1. `tools/annotate.py` — step through frames, key-mark ground-truth events (backlift start, top, plant, impact, FT end).
2. `tools/evaluate.py` — per-event error in frames / ms across all labelled videos; target |impact error| ≤ 2 frames @ 30 fps.
3. Per-run debug outputs: `signals.png` (all signals + phase shading), `keyframes.jpg` (one frame per event) for eyeballing.
4. Known failure modes logged in the report (bat not detected, keypoint dropouts, audio missing/out of sync, occlusion).

## 7. Implementation steps

1. Env: venv (Windows, CUDA 12.8 wheels) + Dockerfile / compose with NVIDIA GPU. ✅
2. Inference pass (pose + bat/ball) with `.npz` cache → re-tuning rules costs seconds. ✅
3. Tracking + batsman selection + keypoint smoothing. ✅
4. Audio onsets. ✅
5. Shot segmentation + phase detection. ✅
6. KPIs + rule-based coaching feedback. ✅
7. OpenCV player (overlay, timeline, KPI panel) + saved mp4/json/pngs. ✅
8. Annotation + evaluation tools. ✅
9. **Next:** tune thresholds on the provided videos, fine-tune bat detector on Roboflow data, side-on support.
