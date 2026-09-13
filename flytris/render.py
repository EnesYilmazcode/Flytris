"""Render a replay of many parallel games as one grid video."""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tetris import HIDDEN, Batch

W, H = 1080, 1350
TOP, BOTTOM = 190, 110
BG = (11, 13, 18)
PALETTE = np.array([
    (26, 30, 40),     # empty
    (63, 208, 224),   # I
    (242, 210, 60),   # O
    (176, 92, 230),   # T
    (90, 214, 90),    # S
    (232, 80, 80),    # Z
    (74, 120, 232),   # J
    (240, 150, 60),   # L
], np.uint8)
GOLD = (255, 196, 0)
DEAD_FILL = np.array([46, 24, 30], np.uint8)
DEAD_EMPTY = np.array([15, 17, 23], np.uint8)
FLASH = np.array([255, 70, 70], np.uint8)


def _font(name, size):
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def grid_layout(n, max_w=W - 60, max_h=H - TOP - BOTTOM):
    for cell in range(12, 0, -1):
        gap = max(2, cell)
        pw, ph = 10 * cell + gap, 20 * cell + gap
        for cols in range(1, n + 1):
            rows = -(-n // cols)
            if cols * pw <= max_w and rows * ph <= max_h:
                return cols, rows, cell, gap
    raise ValueError(f"{n} games do not fit")


def board_pixels(boards, alive, flash):
    vis = boards[:, HIDDEN:, :]
    img = PALETTE[vis]
    filled = (vis != 0)[..., None]
    img[~alive] = np.where(filled[~alive], DEAD_FILL, DEAD_EMPTY)
    img[flash] = np.where(filled[flash], FLASH, PALETTE[0])
    return img


def compose(boards, alive, flash, layout):
    cols, rows, cell, gap = layout
    pw, ph, o = 10 * cell + gap, 20 * cell + gap, gap // 2
    tiles = np.empty((cols * rows, ph, pw, 3), np.uint8)
    tiles[:] = BG
    img = board_pixels(boards, alive, flash)
    tiles[:len(boards), o:o + 20 * cell, o:o + 10 * cell] = img.repeat(cell, 1).repeat(cell, 2)
    grid = tiles.reshape(rows, cols, ph, pw, 3).transpose(0, 2, 1, 3, 4)
    return grid.reshape(rows * ph, cols * pw, 3)


def render(seed, choices, out_mp4, title, subtitle, still_png=None, fps=30, hold=75):
    """choices: int array [steps, games] of placement indices."""
    steps, n = choices.shape
    layout = grid_layout(n)
    cols, rows, cell, gap = layout
    gw, gh = cols * (10 * cell + gap), rows * (20 * cell + gap)
    gx, gy = (W - gw) // 2, TOP + (H - TOP - BOTTOM - gh) // 2
    big, mid, mono = _font("segoeuib.ttf", 58), _font("segoeui.ttf", 30), _font("consola.ttf", 30)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", out_mp4],
        stdin=subprocess.PIPE)
    batch = Batch(n, seed)
    died_at = np.full(n, -10**9)

    def draw(t, winner=None, zoom=0.0):
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = BG
        flash = t - died_at < 5
        grid = compose(batch.boards, batch.alive, flash, layout)
        if zoom:
            grid = (grid * (1 - 0.7 * zoom)).astype(np.uint8)
        canvas[gy:gy + gh, gx:gx + gw] = grid
        if zoom:
            big_cell = int(cell + (22 - cell) * zoom)
            img = board_pixels(batch.boards[winner:winner + 1], batch.alive[winner:winner + 1],
                               np.zeros(1, bool))[0].repeat(big_cell, 0).repeat(big_cell, 1)
            r, c = divmod(winner, cols)
            sx, sy = gx + c * (10 * cell + gap), gy + r * (20 * cell + gap)
            bw, bh = img.shape[1], img.shape[0]
            x0 = int(sx + (W // 2 - bw // 2 - sx) * zoom)
            y0 = int(sy + (gy + gh // 2 - bh // 2 - sy) * zoom)
            canvas[y0 - 4:y0 + bh + 4, x0 - 4:x0 + bw + 4] = GOLD
            canvas[y0:y0 + bh, x0:x0 + bw] = img
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        d.text((W // 2, 70), title, font=big, fill=(240, 240, 240), anchor="mm")
        d.text((W // 2, 130), subtitle, font=mid, fill=(150, 156, 170), anchor="mm")
        if winner is None:
            line = f"alive {batch.alive.sum():>4} / {n}    best {batch.lines.max():>3} lines"
            d.text((W // 2, H - 55), line, font=mono, fill=(200, 204, 214), anchor="mm")
        else:
            line = f"Lord of the Flies: #{winner + 1}, {batch.lines[winner]} lines"
            d.text((W // 2, H - 55), line, font=mono, fill=GOLD, anchor="mm")
        return np.asarray(im)

    t = 0
    for t in range(steps):
        if not batch.alive.any():
            break
        was_alive = batch.alive.copy()
        batch.step(choices[t])
        died_at[was_alive & ~batch.alive] = t
        ff.stdin.write(draw(t).tobytes())

    winner = int(np.lexsort((-batch.placed, -batch.lines))[0])
    for k in range(hold):
        zoom = min(1.0, k / 20) ** 0.5
        frame = draw(t + 5 + k, winner, max(zoom, 1e-3))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    if still_png:
        Image.fromarray(frame).save(still_png)
    return winner
