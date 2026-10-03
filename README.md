# Batsman Analyser (prototype v0.1)

Front-on (bowler's end) cricket batting video → **shot phases**, **KPIs**, **coaching cues**,
and an OpenCV overlay player. See `PLAN.md` for the design and KPI definitions.

Models: `yolo26m-pose.pt` (pose) + `yoloe-26m-seg.pt` (zero-shot "cricket bat" masks) + optional `weights/ball_best.pt`
(YOLOv8 ball detector, **confirmation only** - see below),
plus audio onset detection for the bat-ball "crack". Weights auto-download into `weights/` on first run.

## Setup

### Option A — venv (Windows, RTX 4060)
```bat
setup_venv.bat
.venv\Scripts\activate
python analyse.py videos\clip.mp4
```

### Option B — Docker (GPU, headless)
Needs Docker Desktop + WSL2 GPU support (NVIDIA driver ≥ 570 for CUDA 12.8).
```bat
docker compose build
docker compose run --rm analyser videos/clip.mp4 --no-show
```
The container has no display → use `--no-show` and open `out/<clip>/annotated.mp4`.
For the interactive player, run from the venv.

## Usage
```
python analyse.py <video|folder> [--hand auto|R|L] [--no-bat] [--bat-weights my_bat.pt]
                                 [--device cuda:0] [--recompute] [--no-show] [--no-video]
```
Player keys: `space` pause · `a/d` step · `s` slow-mo · `n` jump to next impact · `r` restart · `q` quit.

Outputs in `out/<clip>/`:

| file | what |
|---|---|
| `report.json` | per shot: event frames/times, impact method + confidence, all KPIs (value, unit, reliability, status), coaching cues |
| `annotated.mp4` | overlay: skeleton coloured by phase, bat mask + axis, head trail + stance head line, live panel, phase timeline |
| `keyframes.jpg` | one row per shot: backlift start / top / front-foot plant / impact / follow-through end — check these by eye |
| `signals.png` | hand height, speeds, head, ankles, bat angle with phase shading and audio onsets — for tuning |
| `pose.npz`, `bat.npz` | inference cache (delete or `--recompute` to redo) |

## Configuration
Defaults live in `batsman_analyser/config.py`; override any of them in `config.yaml` (or `--config other.yaml`):
```yaml
ball:
  enabled: true                 # false -> body rules only (same as --no-ball)
  weights: weights/ball_best.pt
phase:
  swing_min_speed: 3.0
```

### Ball detection = confirmation only
Shots are always found from the body (pose) rules, and the impact frame is always "head lowest". The ball never
creates a shot by itself and never moves the impact. A *live* ball (moving, near the hands, below the chest, not the
helmet / a glove / the lifted bat handle) seen around the swing:
* marks the shot `ball_confirmed` and raises impact confidence to `high`;
* lets a softer swing without a clear set stance count as a shot (`source: "ball"` in report.json) - coaching drills,
  gentle defensive pushes.
Two ball models were compared (`tools/ball_test.py`): `best.pt` (class `ball`) separates real deliveries from demo swings
on 1080p footage; `best(1).pt` (`tennis-ball`) missed the live ball. On 720p indoor clips the live ball is too small for
either, so those clips run on the body rules unchanged.

## Phases
`STANCE → BACKLIFT → DOWNSWING → IMPACT → FOLLOW-THROUGH → FINISH`, plus front-foot lift/plant events.
A shot = still stance → backlift → fast hands-down downswing. Impact = head lowest after the downswing (audio crack overrides when clean).

## Measuring accuracy
```
python tools/annotate.py videos/clip.mp4     # key-mark ground truth -> labels/clip.json
python analyse.py videos --no-show
python tools/evaluate.py                     # per-event MAE (ms), bias, shot recall/precision
python tools/synthetic_test.py               # logic self-test, no models needed
```
All thresholds live in `batsman_analyser/config.py`.

## Known limitations (v0.1)
* Front-on only: stride length toward the bowler is foreshortened (flagged `low` reliability).
* Zero-shot bat detection is imperfect on motion-blurred frames; bat KPIs are skipped when the bat isn't tracked.
  Next step: fine-tune a bat detector on a Roboflow cricket-bat dataset and pass `--bat-weights`.
* Batsman tracking can switch to the wicket-keeper when the camera is close behind the stumps (Sample 7 clips):
  phases then follow the wrong person. Those clips are excluded from evaluation (`labels/excluded/`) until tracking is fixed.
* Edited footage (jump cuts, zooms, slow-motion replays) is not handled - a replay is counted as another shot.
* Handedness is guessed from which foot strides; override with `--hand`.
* Rotation / sequencing KPIs are 2D proxies (projected shoulder/hip width).
* Thresholds are initial heuristics until tuned on labelled footage.
