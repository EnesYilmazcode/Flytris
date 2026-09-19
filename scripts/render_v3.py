"""Render one untouched V3 evaluation game as a single-board MP4."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.tetris import Batch
from flytris.v3_policy import LinearAfterstatePolicy, play_policy
from render_best import H, W, frame


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "v3_nobrain" / "model.npz"))
    ap.add_argument("--seed", type=int, default=700014,
                    help="700014 scored 119 lines in the final evaluation")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames-per-piece", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "media" / "v3_learned_119_lines.mp4"))
    args = ap.parse_args(argv)

    result = play_policy(LinearAfterstatePolicy.load(args.model), args.seed, args.cap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ], stdin=subprocess.PIPE)
    game = Batch(1, args.seed)
    subtitle = f"learned V3 | untouched evaluation seed {args.seed}"
    first = frame(game.boards[0], 0, 0, 0, 0,
                  title="V3 LEARNED CONTROLLER", subtitle=subtitle)
    for _ in range(args.fps):
        ff.stdin.write(first.tobytes())
    for move in result["moves"]:
        game.step([int(move)])
        image = frame(game.boards[0], 0, 0, int(game.placed[0]), int(game.lines[0]),
                      title="V3 LEARNED CONTROLLER", subtitle=subtitle)
        for _ in range(args.frames_per_piece):
            ff.stdin.write(image.tobytes())
    note = f"{result['lines']} LINES | {result['pieces']} PIECES"
    final = frame(game.boards[0], 0, 0, result["pieces"], result["lines"], note,
                  title="V3 LEARNED CONTROLLER", subtitle=subtitle)
    for _ in range(args.fps * 2):
        ff.stdin.write(final.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {out}: {result['lines']} lines, {result['pieces']} pieces")


if __name__ == "__main__":
    main()
