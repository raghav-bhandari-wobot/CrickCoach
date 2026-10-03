"""All tunable thresholds in one place.

Distances are in torso units (TU = median shoulder-mid -> hip-mid length during stance),
speeds in TU/s, times in seconds unless stated. Values are initial heuristics and are meant
to be tuned with tools/evaluate.py against hand-labelled videos.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class ModelCfg:
    pose_weights: str = "weights/yolo26m-pose.pt"
    pose_imgsz: int = 960
    pose_conf: float = 0.25
    bat_weights: str = "weights/yoloe-26m-seg.pt"   # YOLOE text-prompt model, or a custom bat detector
    bat_prompts: tuple = ("cricket bat",)   # ball detection removed (to be trained later)
    bat_imgsz: int = 640
    bat_conf: float = 0.08            # low: we pick the detection nearest the hands ourselves
    crop_expand_w: float = 2.2        # crop around batsman bbox for bat detection
    crop_expand_h: float = 1.5
    device: str = ""                  # "" -> auto (cuda:0 if available)


@dataclass
class BallCfg:
    """Ball detection is CONFIRMATION ONLY. It never creates a shot on its own and never moves the
    impact frame (impact stays = head lowest). A moving ball near the bat around the swing:
      (a) raises impact confidence to "high" (ball_confirmed in report.json),
      (b) lets a softer swing without a clear set stance count as a shot (coaching drills, gentle
          defensive pushes) - those would otherwise be rejected as waggles / shadow swings.
    Validated: works on 1080p footage (YouTube clip 1); on small/blurred 720p balls (Sample 29) it
    rarely fires, so those clips fall back to the body rules unchanged."""
    enabled: bool = True
    weights: str = "weights/ball_best.pt"   # YOLOv8 single-class 'ball' detector
    imgsz: int = 1280                 # full frame - the ball is a few pixels wide
    conf: float = 0.25
    near_hands_tu: float = 1.5        # live ball must be this close to the hands / bat
    ignore_head_tu: float = 0.5       # the helmet is often detected as a ball
    ignore_wrist_tu: float = 0.3      # ... and so are white gloves
    static_frac: float = 0.004        # moved < this x frame size in 5 frames -> a ball lying on the ground
    max_height_tu: float = 0.5        # live ball reaches the bat below the chest (TU above the hips)
    attached_tu: float = 0.3          # moves with the hands (within this, over 5 frames) -> bat handle / glove
    window_pre_s: float = 0.6         # evidence counted in [downswing - pre, downswing + post]
    window_post_s: float = 0.8
    confirm_min_frames: int = 3       # frames with a live ball needed to confirm
    relaxed_swing_min_speed: float = 1.5   # TU/s downswing for a ball-confirmed shot without a set stance


@dataclass
class TrackCfg:
    iou_match: float = 0.25
    max_lost: int = 20                # frames a track survives without a match
    stitch_gap_s: float = 1.5         # re-link batsman fragments broken by occlusion
    stitch_dist_h: float = 0.6        # max centre jump (in box heights) when stitching
    min_score_frac: float = 0.15      # batsman-track score relative to the best track


@dataclass
class SmoothCfg:
    kp_conf: float = 0.35             # keypoints below this are treated as missing
    max_gap_s: float = 0.35           # interpolate gaps up to this long
    sg_window_s: float = 0.13         # Savitzky-Golay window (kept short to not blur impact)
    sg_poly: int = 2
    median_s: float = 0.12            # median pre-filter (despike) window
    slow_window_s: float = 0.3        # smoothing for event-finding signals (hand/head height)


@dataclass
class PhaseCfg:
    # shot = still stance -> backlift -> fast hands-DOWN downswing (tuned on Cricket Sample 29 Clip)
    swing_min_speed: float = 3.0     # TU/s hand speed (moving down) for a downswing
    stance_still_s: float = 0.6       # a set stance lasts at least this long ...
    stance_still_max: float = 0.45    # ... with mean hand + head speed below this (TU/s)
    stance_hand_tol_tu: float = 0.45  # ... and hands within this of the batter's usual stance height
    stance_search_s: float = 2.5      # stance searched in [downswing - 2.5 s, downswing - 0.2 s]
    stance_gap_s: float = 0.2
    stance_move_speed: float = 1.2    # TU/s: stance ends when hands or head move faster than this
    backlift_min_rise: float = 0.25   # TU hands rise from stance to top of backlift
    impact_search_s: float = 0.6      # impact (head lowest) searched this long after the downswing peak
    shot_min_drop: float = 0.25       # TU hands must drop from top to bottom of the swing ...
    leave_min_rise: float = 0.4       # TU hands rise for a leave ('shoulder arms')
    walk_max_tu: float = 1.2          # hip travel in 2 s above this = walking, not batting
    audio_max_rate: float = 0.4       # onsets per second above this -> audio too noisy to trust
    shot_min_separation_s: float = 3.0
    shot_post_s: float = 2.0
    backlift_start_frac: float = 0.15
    audio_z_min: float = 6.0          # onset strength (robust z) to trust the bat-ball crack
    audio_agree_s: float = 0.25       # audio must be within this of the kinematic estimate
    ft_still_frac: float = 0.15
    ft_still_s: float = 0.15
    ft_max_s: float = 1.4
    foot_plant_speed: float = 0.6     # TU/s front ankle speed -> planted
    foot_move_speed: float = 1.0      # TU/s front ankle speed -> moving


@dataclass
class KpiCfg:
    # (good_max, warn_max) pairs -> status good / warn / bad
    head_drift_tu: tuple = (0.20, 0.40)
    head_over_front_foot_tu: tuple = (0.30, 0.55)
    eye_tilt_deg: tuple = (12.0, 22.0)
    bat_vertical_deg: tuple = (20.0, 35.0)        # only judged for vertical-bat shots
    bat_path_dev_tu: tuple = (0.35, 0.6)
    balance_offset: tuple = (0.25, 0.45)          # hip centre from base middle (0 = middle, 0.5 = over a foot)
    finish_motion_tu: tuple = (0.15, 0.30)
    plant_before_impact_ms: tuple = (0.0, -60.0)  # >=0 good, down to -60 ms warn, less -> bad
    stance_width_tu: tuple = (0.45, 1.3)          # ankle distance in TU, inside -> good


@dataclass
class TimeCfg:
    slowmo: float = 1.0               # 0 = auto-estimate (experimental); 4 = 120 fps footage played at 30 fps
    ref_downswing_s: float = 0.25     # typical real downswing (top of backlift -> impact)
    candidates: tuple = (1, 2, 4, 8)


@dataclass
class Config:
    time: TimeCfg = field(default_factory=TimeCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    track: TrackCfg = field(default_factory=TrackCfg)
    smooth: SmoothCfg = field(default_factory=SmoothCfg)
    phase: PhaseCfg = field(default_factory=PhaseCfg)
    kpi: KpiCfg = field(default_factory=KpiCfg)
    ball: BallCfg = field(default_factory=BallCfg)


def load_config(path=None):
    """Defaults from the dataclasses above, overridden by a YAML file (config.yaml by default):

        ball:
          enabled: false
        phase:
          swing_min_speed: 3.0
    """
    cfg = Config()
    path = Path(path) if path else Path("config.yaml")
    if not path.exists():
        return cfg
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for sec, vals in data.items():
        if not hasattr(cfg, sec) or not isinstance(vals, dict):
            raise ValueError(f"{path}: unknown section '{sec}'")
        obj = getattr(cfg, sec)
        names = {f.name: f for f in fields(obj)}
        for k, v in vals.items():
            if k not in names:
                raise ValueError(f"{path}: unknown key '{sec}.{k}'")
            setattr(obj, k, tuple(v) if isinstance(v, list) else v)
    return cfg
