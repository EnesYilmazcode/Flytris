"""Build the finished video: timeline, soundtrack, render in 10-second chunks, mux.

  python make_final.py            # all 7,172 flies -> ../media/flytris_arcade_3d_final.mp4
  python make_final.py --n 1500   # quicker draft
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7172)
    ap.add_argument("--out", type=Path, default=HERE.parent / "media" / "flytris_arcade_3d_final.mp4")
    a = ap.parse_args()
    py = [sys.executable, "-X", "utf8"]
    chunks = HERE / "chunks"
    chunks.mkdir(exist_ok=True)

    run(*py, "capture.py", "timeline", "--n", a.n)
    run(*py, "soundtrack.py", "timeline.json", "soundtrack.wav")
    import json
    end = json.loads((HERE / "timeline.json").read_text())["T"]["end"]
    bounds = [0, 10, 20, end] if end > 20 else [0, 10, end]
    parts = []
    for i, (s, e) in enumerate(zip(bounds, bounds[1:])):
        part = chunks / f"c{i}.mp4"
        run(*py, "capture.py", "video", "--n", a.n, "--start", s, "--end", e, "--out", part)
        parts.append(part)
    (chunks / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", chunks / "list.txt", "-c", "copy", chunks / "video.mp4")
    run("ffmpeg", "-v", "error", "-y", "-i", chunks / "video.mp4", "-i", "soundtrack.wav", "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
