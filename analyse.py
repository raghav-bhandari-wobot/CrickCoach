"""Batsman Analyser - CLI.

    python analyse.py videos/clip.mp4                 # analyse + interactive OpenCV player
    python analyse.py videos/clip.mp4 --no-show       # headless (Docker): writes out/<clip>/...
    python analyse.py videos/ --no-show               # every video in a folder
"""
import argparse
from pathlib import Path

from batsman_analyser.config import load_config
from batsman_analyser.pipeline import analyse
from batsman_analyser.viz import Renderer, keyframes, play, signals_plot, write_video

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def print_summary(report):
    print("\n" + "=" * 72)
    print(f"{report['video']}  |  {report['meta']['fps']:.0f} fps (slow-mo x{report['slowmo_factor']:g} -> "
          f"{report['real_fps']:.0f} real fps)  |  shots: {len(report['shots'])}")
    for w in report["warnings"]:
        print(f"  ! {w}")
    for n, sh in enumerate(report["shots"], 1):
        e, inf, K = sh["events_time_s"], sh["info"], sh["kpis"]
        print(f"\n-- Shot {n}: {K['shot_type']['value']}  ({K['shot_type']['unit']})  "
              f"hand={inf['handedness']}")
        print("   events (s): " + ", ".join(f"{k}={v}" for k, v in e.items() if v is not None))
        ball = ""
        if "ball_confirmed" in inf:
            ball = f"  | ball {'confirmed' if inf['ball_confirmed'] else 'not seen'} ({inf['ball_frames']} fr)"
        print(f"   impact via {inf['impact_method']} (confidence {inf['impact_confidence']}){ball}"
              + ("  [found via ball-confirmed path]" if inf.get("source") == "ball" else ""))
        for k in K.values():
            if k["name"].startswith("Shot type"):
                continue
            flag = {"good": "OK ", "warn": "!  ", "bad": "!! "}.get(k["status"], "   ")
            print(f"   {flag}{k['name']:<42} {str(k['value']):>10} {k['unit']}  [{k['reliability']}]")
        fb = sh["feedback"]
        if fb["work_on"]:
            print("   WORK ON:")
            for c in fb["work_on"]:
                print(f"     - ({c['severity']}) {c['cue']}")
        if fb["went_well"]:
            print("   WENT WELL: " + " ".join(fb["went_well"]))
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Front-on cricket batting analyser")
    ap.add_argument("input", help="video file or folder")
    ap.add_argument("--out", default="out")
    ap.add_argument("--hand", default="auto", choices=["auto", "R", "L"], help="batsman handedness")
    ap.add_argument("--config", default="config.yaml", help="YAML overrides of batsman_analyser/config.py")
    ap.add_argument("--no-bat", action="store_true", help="skip bat detection")
    ap.add_argument("--no-ball", action="store_true", help="skip ball confirmation (overrides config)")
    ap.add_argument("--bat-weights", default=None, help="custom bat detector or other YOLOE weights")
    ap.add_argument("--pose-weights", default=None)
    ap.add_argument("--device", default="", help="e.g. cuda:0 or cpu (default auto)")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="slow-motion factor (4 = 120 fps shot, played at 30). 0 = auto-estimate (experimental)")
    ap.add_argument("--imgsz", type=int, default=None, help="pose inference size (default 960)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached inference")
    ap.add_argument("--no-show", action="store_true", help="do not open the OpenCV player")
    ap.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_ball:
        cfg.ball.enabled = False
    cfg.model.device = args.device
    cfg.time.slowmo = args.slowmo
    if args.imgsz:
        cfg.model.pose_imgsz = args.imgsz
    if args.bat_weights:
        cfg.model.bat_weights = args.bat_weights
    if args.pose_weights:
        cfg.model.pose_weights = args.pose_weights

    p = Path(args.input)
    videos = sorted(v for v in p.iterdir() if v.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    for v in videos:
        report, S, labels, onsets, meta, out = analyse(v, args.out, cfg, args.hand, not args.no_bat, args.recompute)
        print_summary(report)
        r = Renderer(S, labels, report["shots"], meta)
        signals_plot(S, labels, report["shots"], onsets, out / "signals.png")
        keyframes(v, r, report["shots"], out / "keyframes.jpg")
        if not args.no_video:
            write_video(v, out / "annotated.mp4", r, meta)
        print(f"outputs -> {out}")
        if not args.no_show:
            play(v, r, meta)


if __name__ == "__main__":
    main()
