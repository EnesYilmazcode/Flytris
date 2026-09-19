"""Render exact independently-seeded V3 fly replays as a tournament video."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.render import (  # noqa: E402
    BG, CELLS, CLOSE, CY, GHOST, GOLD, H, HOLD, PULL, PX, PY, W, WIDTHS, XS, ZOOM,
    blend, draw_view, font, grid_layout, tile_box, zoom_path,
)
from flytris.tetris import COLS, HIDDEN, Batch  # noqa: E402


class TournamentCamera:
    def __init__(self, n, opening, winner):
        self.cols, self.rows, self.s_full = grid_layout(n)
        self.full = (self.cols * PX / 2, self.rows * PY / 2, self.s_full)
        scale = (H - 380) / PY

        def close(index):
            row, col = divmod(index, self.cols)
            return ((col + 0.5) * PX, (row + 0.5) * PY, scale)

        self.opening = close(opening)
        self.winner = close(winner)

    def at(self, frame, end_start=None):
        if end_start is not None:
            return zoom_path(min(1.0, (frame - end_start) / ZOOM), self.full, self.winner)
        if frame < CLOSE:
            return self.opening
        if frame < CLOSE + PULL:
            return zoom_path((frame - CLOSE) / PULL, self.opening, self.full)
        return self.full


def load_replays(paths):
    records = []
    for path in paths:
        with np.load(path) as data:
            moves = data["moves"]
            seeds = data["seeds"]
            lines = data["lines"]
            placed = data["placed"]
            for i in range(len(seeds)):
                records.append((int(seeds[i]), moves[:, i].copy(), int(lines[i]), int(placed[i])))
    return records


def verify(records):
    for seed, moves, expected_lines, expected_pieces in records:
        game = Batch(1, seed)
        for move in moves:
            if move == 255 or not game.alive[0]:
                break
            game.step([int(move)])
        actual = (int(game.lines[0]), int(game.placed[0]))
        if actual != (expected_lines, expected_pieces):
            raise RuntimeError(
                f"replay mismatch for seed {seed}: expected "
                f"{expected_lines}/{expected_pieces}, got {actual[0]}/{actual[1]}"
            )


def render(records, output, still, fps=30, visual_repeat=1, clean=False):
    verify(records)
    n = len(records)
    target = np.array([(record[2], record[3]) for record in records])
    winner = int(np.lexsort((-target[:, 1], -target[:, 0]))[0])
    opening = 0 if winner != 0 else 1
    visual_source = np.tile(np.arange(n), visual_repeat)
    visual_n = len(visual_source)
    cols, rows, _ = grid_layout(visual_n)
    center = (rows // 2) * cols + cols // 2

    def central_copy(source):
        choices = np.flatnonzero(visual_source == source)
        return int(choices[np.argmin(np.abs(choices - center))])

    opening_visual = central_copy(opening)
    winner_visual = central_copy(winner)
    camera = TournamentCamera(visual_n, opening_visual, winner_visual)
    games = [Batch(1, seed) for seed, _, _, _ in records]
    died = np.full(n, -10**6)
    frame_no = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", str(output)],
        stdin=subprocess.PIPE,
    )

    def snapshot():
        return (
            np.stack([game.boards[0] for game in games]),
            np.array([bool(game.alive[0]) for game in games]),
            np.array([int(game.lines[0]) for game in games]),
        )

    def emit(end_start=None, boards_override=None, alive_override=None):
        nonlocal frame_no
        boards, alive, lines = snapshot()
        if boards_override is not None:
            boards = boards_override
        if alive_override is not None:
            alive = alive_override
        state = np.where(alive, 0, np.where(frame_no - died < 15, 2, 1))
        boards = boards[visual_source]
        state = state[visual_source]
        if end_start is not None:
            state[:] = 1
            state[winner_visual] = 0
        cam = camera.at(frame_no, end_start)
        im = draw_view(boards, state, cam, camera, visual_n)
        pixels = np.array(im)
        if not clean:
            pixels[:150] = (pixels[:150] * 0.12 + BG * 0.88).astype(np.uint8)
            pixels[H - 100:] = (pixels[H - 100:] * 0.12 + BG * 0.88).astype(np.uint8)
        im = Image.fromarray(pixels)
        draw = ImageDraw.Draw(im)
        alive_n = int(alive.sum())

        if end_start is None and frame_no < CLOSE:
            x0, y0, x1, y1 = tile_box(opening_visual, cam, camera)
            draw.rectangle((0, 0, max(0, x0 - 18), H), fill=tuple(BG))
            draw.rectangle((min(W, x1 + 18), 0, W, H), fill=tuple(BG))
            draw.rectangle((max(0, x0 - 18), 0, min(W, x1 + 18), max(0, y0 - 18)),
                           fill=tuple(BG))
            draw.rectangle((max(0, x0 - 18), min(H, y1 + 18), min(W, x1 + 18), H),
                           fill=tuple(BG))

        if not clean and end_start is None:
            if frame_no < CLOSE:
                draw.text((W // 2, 65), "ONE SIMULATED FLY CONNECTOME",
                          font=font("segoeuib.ttf", 50), fill=(240, 240, 240), anchor="mm")
                draw.text((W // 2, H - 48), "Now pull back...",
                          font=font("segoeui.ttf", 29), fill=(170, 176, 190), anchor="mm")
            else:
                draw.text((W // 2, 65), f"{n} FLY CONNECTOMES PLAY TETRIS",
                          font=font("segoeuib.ttf", 52), fill=(240, 240, 240), anchor="mm")
                draw.text((60, H - 48), f"ALIVE {alive_n} / {n}",
                          font=font("consola.ttf", 32), fill=(210, 214, 226), anchor="lm")
                draw.text((W - 60, H - 48), f"BEST {int(lines.max())} LINES",
                          font=font("consola.ttf", 32), fill=GOLD, anchor="rm")
        elif not clean:
            fade = min(1.0, (frame_no - end_start) / 25)
            draw.text((W // 2, 65), "WINNER",
                      font=font("segoeuib.ttf", 72), fill=blend(GOLD, fade), anchor="mm")
            draw.text((W // 2, H - 50),
                      f"FLY #{winner + 1}  |  {int(lines[winner])} LINES  |  "
                      f"{int(games[winner].placed[0])} PIECES",
                      font=font("segoeuib.ttf", 34),
                      fill=blend((240, 240, 240), fade), anchor="mm")

        if not clean:
            draw.text((W // 2, 120), "166,700-neuron simulation | exact unseen-game replays",
                      font=font("segoeui.ttf", 27), fill=(140, 146, 160), anchor="mm")

        if not clean and cam[2] >= 30:
            focus = winner_visual if end_start is not None else opening_visual
            x0, _, x1, y1 = tile_box(focus, cam, camera)
            source = int(visual_source[focus])
            draw.text((x0, y1 + 0.55 * cam[2]), f"FLY #{source + 1}",
                      font=font("segoeuib.ttf", 30), fill=GOLD, anchor="lm")
            draw.text((x1, y1 + 0.55 * cam[2]), f"{int(lines[source])} LINES",
                      font=font("segoeuib.ttf", 30), fill=GOLD, anchor="rm")

        if clean and end_start is not None and cam[2] >= 28:
            x0, y0, x1, y1 = tile_box(winner_visual, cam, camera)
            width = max(3, int(cam[2] * 0.08))
            draw.rectangle((x0 - width, y0 - width, x1 + width, y1 + width),
                           outline=GOLD, width=width)

        ff.stdin.write(np.asarray(im).tobytes())
        frame_no += 1
        return im

    def falling_boards(before, pieces, choices, target_y, progress):
        """Draw ghost landing outlines and interpolate each live piece toward them."""
        ids = before.copy()
        for i in range(n):
            piece, choice, yt = int(pieces[i]), int(choices[i]), int(target_y[i])
            if choice == 255 or yt < 0:
                continue
            cells = CELLS[piece][choice]
            xt = int(XS[piece][choice])
            start_x = (COLS - int(WIDTHS[piece][choice])) // 2
            start_y = min(HIDDEN, yt)

            ghost_rows = yt + cells[:, 0]
            ghost_cols = xt + cells[:, 1]
            empty = ids[i, ghost_rows, ghost_cols] == 0
            ids[i, ghost_rows[empty], ghost_cols[empty]] = GHOST

            x = round(start_x + (xt - start_x) * min(1.0, progress / 0.3))
            y = round(start_y + (yt - start_y) * max(0.0, (progress - 0.3) / 0.7))
            rows = y + cells[:, 0]
            cols = x + cells[:, 1]
            visible = (rows >= 0) & (rows < ids.shape[1]) & (cols >= 0) & (cols < ids.shape[2])
            rows, cols = rows[visible], cols[visible]
            empty = (ids[i, rows, cols] == 0) | (ids[i, rows, cols] == GHOST)
            ids[i, rows[empty], cols[empty]] = piece + 1
        return ids

    step = 0
    max_steps = max(len(record[1]) for record in records)
    while step < max_steps and sum(bool(game.alive[0]) for game in games) > 1:
        alive_before = np.array([bool(game.alive[0]) for game in games])
        before = np.stack([game.boards[0].copy() for game in games])
        pieces_before = np.array([int(game.piece) for game in games])
        choices = np.full(n, 255, np.uint8)
        target_y = np.full(n, -1, np.int16)
        for i, game in enumerate(games):
            moves = records[i][1]
            if not alive_before[i] or step >= len(moves) or moves[step] == 255:
                continue
            choices[i] = moves[step]
            game.step([int(choices[i])])
            target_y[i] = int(game.last_y[0])
        alive_after = np.array([bool(game.alive[0]) for game in games])
        newly_dead = alive_before & ~alive_after
        repeats = 4 if frame_no < CLOSE else (3 if step < 110 else 2)
        for repeat in range(repeats):
            if repeat == repeats - 1:
                died[newly_dead] = frame_no
                emit()
            else:
                progress = (repeat + 1) / repeats
                emit(boards_override=falling_boards(
                    before, pieces_before, choices, target_y, progress),
                    alive_override=alive_before)
        step += 1

    end_start = frame_no
    # Keep playing the last survivor during the push-in so the final card shows the
    # complete recorded 86-line game rather than the score at the moment it won.
    while step < max_steps and games[winner].alive[0]:
        move = records[winner][1][step]
        if move == 255:
            break
        before = np.stack([game.boards[0].copy() for game in games])
        pieces_before = np.array([int(game.piece) for game in games])
        choices = np.full(n, 255, np.uint8)
        target_y = np.full(n, -1, np.int16)
        choices[winner] = move
        games[winner].step([int(move)])
        target_y[winner] = int(games[winner].last_y[0])
        emit(end_start, falling_boards(before, pieces_before, choices, target_y, 0.45))
        emit(end_start, falling_boards(before, pieces_before, choices, target_y, 0.80))
        emit(end_start)
        step += 1
    final = None
    while frame_no - end_start < ZOOM + HOLD:
        final = emit(end_start)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    final.save(still)
    print(f"rendered {output}: {n} authentic games, winner fly #{winner + 1}, "
          f"{records[winner][2]} lines")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "media" / "fly_connectome_tournament.mp4")
    parser.add_argument("--still", type=Path, default=ROOT / "media" / "fly_connectome_tournament_final.png")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--visual-repeat", type=int, default=1)
    parser.add_argument("--clean", action="store_true", help="render without any text")
    args = parser.parse_args(argv)
    render(load_replays(args.replays), args.out, args.still, args.fps,
           visual_repeat=args.visual_repeat, clean=args.clean)


if __name__ == "__main__":
    main()
