"""Cut a shorter / smaller copy of a video (keeps audio).

    python tools/trim.py "videos/Sample 29.mp4" --start 0 --dur 180     -> videos/Sample 29_0-180.mp4
    python tools/trim.py "videos/Sample 29.mp4" --start 60 --dur 90 --height 832
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--dur", type=float, default=180)
ap.add_argument("--height", type=int, default=0, help="resize to this height (0 = keep)")
a = ap.parse_args()
src = Path(a.video)
dst = src.with_name(f"{src.stem}_{int(a.start)}-{int(a.start + a.dur)}.mp4")
vf = ["-vf", f"scale=-2:{a.height}"] if a.height else []
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(a.start), "-i", str(src), "-t", str(a.dur), *vf,
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", str(dst)]
subprocess.run(cmd, check=True)
print("wrote", dst)
