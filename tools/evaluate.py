"""Compare detected events (out/<clip>/report.json) against hand labels (labels/<clip>.json).

    python tools/evaluate.py [--labels labels] [--out out]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENTS = ["backlift_start", "top_of_backlift", "front_foot_plant", "impact", "follow_through_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="out")
    ap.add_argument("--match-s", type=float, default=0.5,
                    help="max impact distance (s) to match shots; widened for labels with precision_s")
    a = ap.parse_args()
    err = defaultdict(list)
    n_gt = n_pred = n_match = 0
    methods = defaultdict(list)
    for lf in sorted(Path(a.labels).glob("*.json")):
        gt = json.loads(lf.read_text())
        rp = Path(a.out) / lf.stem / "report.json"
        if not rp.exists():
            print(f"skip {lf.stem}: no report (run analyse.py first)")
            continue
        rep = json.loads(rp.read_text())
        fps = rep["meta"]["fps"]
        # labels given to the second ("00:42" = somewhere in 42.0-42.99 s) need a wider window
        win = max(a.match_s, gt.get("precision_s", 0) + 0.5)
        preds = rep["shots"]
        n_gt += len(gt["shots"])
        n_pred += len(preds)
        used = set()
        clip_err = []
        for g in gt["shots"]:
            if "impact" not in g:
                continue
            best = min((abs(p["events"]["impact"] - g["impact"]), j) for j, p in enumerate(preds)) if preds else None
            if best is None or best[0] > win * fps or best[1] in used:
                print(f"{lf.stem}: GT shot @ {g['impact']} not detected")
                continue
            used.add(best[1])
            n_match += 1
            clip_err.append(preds[best[1]]["events"]["impact"] - g["impact"])
            p = preds[best[1]]
            for e in EVENTS:
                if e in g and p["events"].get(e) is not None:
                    d = p["events"][e] - g[e]
                    err[e].append(d / fps * 1000.0)
                    if e == "impact":
                        methods[p["info"]["impact_method"]].append(abs(d))
        fps_ = [round(p["events"]["impact"] / fps, 1) for j, p in enumerate(preds) if j not in used]
        print(f"{lf.stem:<28} GT {len(gt['shots']):>2}  pred {len(preds):>2}  hit {len(clip_err):>2}  "
              f"miss {len(gt['shots']) - len(clip_err):>2}  false+ {len(fps_):>2}  "
              f"impact err (s) {[round(e / fps, 2) for e in clip_err]}" + (f"  false+ @ {fps_}" if fps_ else ""))
    print(f"\nshots: GT={n_gt} predicted={n_pred} matched={n_match} "
          f"(recall {n_match / max(1, n_gt):.0%}, precision {n_match / max(1, n_pred):.0%})")
    print(f"{'event':<22}{'n':>4}{'MAE ms':>9}{'median ms':>11}{'bias ms':>9}{'<=2fr @30':>11}")
    for e in EVENTS:
        v = np.array(err[e])
        if not len(v):
            continue
        print(f"{e:<22}{len(v):>4}{np.mean(np.abs(v)):>9.0f}{np.median(np.abs(v)):>11.0f}"
              f"{np.mean(v):>9.0f}{np.mean(np.abs(v) <= 67):>11.0%}")
    for m, v in methods.items():
        print(f"  impact via {m:<22} n={len(v):<3} mean |err| = {np.mean(v):.1f} frames")


if __name__ == "__main__":
    main()
