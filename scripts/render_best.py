"""Render the best recorded training game as a single-board MP4."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.tetris import HIDDEN, Batch  # noqa: E402

W, H = 720, 1080
CELL = 42
BOARD_W, BOARD_H = 10 * CELL, 20 * CELL
X0, Y0 = (W - BOARD_W) // 2, 150
BG = (10, 11, 16)
WELL = (20, 22, 30)
GRID = (38, 41, 53)
COLORS = [
    (63, 208, 224), (242, 210, 60), (176, 92, 230), (90, 214, 90),
    (232, 80, 80), (74, 120, 232), (240, 150, 60),
]


def font(name, size):
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


TITLE = font("segoeuib.ttf", 40)
LABEL = font("segoeui.ttf", 25)
MONO = font("consola.ttf", 27)


def best_game(run_dir):
    best = None
    for path in sorted(run_dir.glob("gen_*.npz")):
        with np.load(path) as data:
            lines = data["lines"]
            moves = data["moves"]
            pieces = (moves != 255).sum(0)
            i = int(np.lexsort((-pieces, -lines))[0])
            key = (int(lines[i]), int(pieces[i]))
            if best is None or key > best[0]:
                best = (key, path, i, int(data["seed"]), moves[:, i].copy())
    if best is None:
        raise RuntimeError(f"no generation files found in {run_dir}")
    return best


def frame(board, generation, fly, piece_no, lines, note=""):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 55), "BEST FLY SO FAR", font=TITLE, fill=(245, 245, 248), anchor="mm")
    d.text((W // 2, 103), f"generation {generation}  •  fly #{fly + 1}",
           font=LABEL, fill=(148, 154, 170), anchor="mm")
    d.rectangle((X0 - 5, Y0 - 5, X0 + BOARD_W + 5, Y0 + BOARD_H + 5),
                outline=(92, 98, 118), width=5)
    visible = board[HIDDEN:]
    for r in range(20):
        for c in range(10):
            x, y = X0 + c * CELL, Y0 + r * CELL
            value = int(visible[r, c])
            color = WELL if value == 0 else COLORS[value - 1]
            d.rectangle((x, y, x + CELL - 2, y + CELL - 2), fill=color)
            if value:
                d.line((x + 2, y + 2, x + CELL - 5, y + 2), fill=(255, 255, 255), width=2)
                d.line((x + 2, y + 2, x + 2, y + CELL - 5), fill=(255, 255, 255), width=2)
            d.line((x + CELL - 1, y, x + CELL - 1, y + CELL), fill=GRID)
            d.line((x, y + CELL - 1, x + CELL, y + CELL - 1), fill=GRID)
    d.text((X0, 1025), f"PIECE {piece_no}", font=MONO, fill=(174, 180, 196), anchor="ls")
    d.text((X0 + BOARD_W, 1025), f"LINES {lines}", font=MONO, fill=(255, 196, 0), anchor="rs")
    if note:
        d.text((W // 2, 1063), note, font=LABEL, fill=(232, 100, 104), anchor="ms")
    return np.asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs/train_after"))
    ap.add_argument("--out", default=str(ROOT / "runs/train_after/best_so_far.mp4"))
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args(argv)

    (score, path, fly, seed, moves) = best_game(Path(args.run_dir))
    generation = int(path.stem.split("_")[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ], stdin=subprocess.PIPE)

    game = Batch(1, seed)
    first = frame(game.boards[0], generation, fly, 0, 0)
    for _ in range(args.fps):
        ff.stdin.write(first.tobytes())
    for choice in moves:
        if not game.alive[0] or choice == 255:
            break
        game.step([int(choice)])
        image = frame(game.boards[0], generation, fly, int(game.placed[0]), int(game.lines[0]))
        for _ in range(5):
            ff.stdin.write(image.tobytes())
    final = frame(game.boards[0], generation, fly, int(game.placed[0]), int(game.lines[0]),
                  "GAME OVER" if not game.alive[0] else "CURRENT CHECKPOINT")
    for _ in range(args.fps * 2):
        ff.stdin.write(final.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"rendered {out}: generation {generation}, fly {fly + 1}, "
          f"{score[0]} lines, {score[1]} pieces")


if __name__ == "__main__":
    main()
