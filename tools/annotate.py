"""Hand-label ground-truth events so the detector can be measured.

    python tools/annotate.py videos/clip.mp4        -> labels/clip.json

Keys:  d / a  +-1 frame     l / j  +-10 frames     space play/pause
       1 backlift_start   2 top_of_backlift   3 front_foot_plant   4 impact   5 follow_through_end
       n  start next shot    u  undo last mark    w  save    q  save + quit
"""
import json
import sys
from pathlib import Path

import cv2

EVENTS = {ord("1"): "backlift_start", ord("2"): "top_of_backlift", ord("3"): "front_foot_plant",
          ord("4"): "impact", ord("5"): "follow_through_end"}


def main(path):
    path = Path(path)
    out = Path("labels") / f"{path.stem}.json"
    out.parent.mkdir(exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {"video": str(path), "shots": [{}]}
    cap = cv2.VideoCapture(str(path))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    data["fps"] = fps
    i, playing, history = 0, False, []
    sc = min(1.0, 1280 / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            i = max(0, i - 1)
            playing = False
            continue
        img = cv2.resize(fr, None, fx=sc, fy=sc)
        cur = data["shots"][-1]
        cv2.putText(img, f"frame {i}/{T}  t={i / fps:.2f}s  shot {len(data['shots'])}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for n, (k, v) in enumerate(cur.items()):
            cv2.putText(img, f"{k}: {v}{'  <==' if v == i else ''}", (10, 60 + 26 * n),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(img, "1 BL  2 TOP  3 FF-plant  4 IMPACT  5 FT-end | n next shot  u undo  w save  q quit",
                    (10, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("annotate", img)
        k = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            playing = not playing
        elif k == ord("d"):
            i = min(T - 1, i + 1)
        elif k == ord("a"):
            i = max(0, i - 1)
        elif k == ord("l"):
            i = min(T - 1, i + 10)
        elif k == ord("j"):
            i = max(0, i - 10)
        elif k in EVENTS:
            cur[EVENTS[k]] = i
            history.append(EVENTS[k])
        elif k == ord("u") and history:
            cur.pop(history.pop(), None)
        elif k == ord("n"):
            data["shots"].append({})
            history = []
        elif k == ord("w"):
            out.write_text(json.dumps(data, indent=2))
            print("saved", out)
        elif playing:
            i = min(T - 1, i + 1)
    data["shots"] = [s for s in data["shots"] if s]
    out.write_text(json.dumps(data, indent=2))
    print("saved", out)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])
