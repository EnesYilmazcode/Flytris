"""Tournament video: open on one game, pull back to reveal all of them, end on the winner."""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tetris import COLS, HIDDEN, PLACEMENTS, Batch

W, H = 1080, 1350
CY = 705                  # screen row the camera centers on
PX, PY = 11, 22           # tile pitch in cells: a 10x20 well plus gutters
BG = np.array([10, 11, 16])
GOLD = (255, 196, 0)
PIECE_RGB = np.array([
    (63, 208, 224), (242, 210, 60), (176, 92, 230), (90, 214, 90),
    (232, 80, 80), (74, 120, 232), (240, 150, 60)], float)
GHOST, FLASH, DEAD_FILL, DEAD_EMPTY = 8, 9, 10, 11
CLOSE, PULL, ZOOM, HOLD = 75, 105, 60, 90

CELLS = [np.stack([c for _, _, c in opts]) for opts in PLACEMENTS]
XS = [np.array([x for _, x, _ in opts]) for opts in PLACEMENTS]
WIDTHS = [np.array([c[:, 1].max() + 1 for _, _, c in opts]) for opts in PLACEMENTS]

_fonts, _sprites = {}, {}


def font(name, size):
    if (name, size) not in _fonts:
        try:
            _fonts[name, size] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            _fonts[name, size] = ImageFont.load_default()
    return _fonts[name, size]


def sprites(k):
    """[12, k, k, 3] cell images: empty, 7 pieces, ghost, line-clear flash, dead block, dead empty."""
    if k in _sprites:
        return _sprites[k]
    s = np.zeros((12, k, k, 3), np.uint8)
    s[0] = (20, 22, 30)
    if k >= 6:
        s[0, -1, :] = s[0, :, -1] = (32, 35, 46)
    for i, c in enumerate(PIECE_RGB, 1):
        s[i] = c
        if k >= 5:
            e = max(1, k // 6)
            s[i, :e, :] = c * 0.5 + 127
            s[i, e:, :e] = c * 0.75 + 60
            s[i, -e:, e:] = c * 0.5
            s[i, e:-e, -e:] = c * 0.62
    s[GHOST] = s[0]
    if k >= 6:
        e = max(1, k // 12)
        g = (100, 106, 124)
        s[GHOST, :e, :] = s[GHOST, -e:, :] = s[GHOST, :, :e] = s[GHOST, :, -e:] = g
    s[FLASH] = (240, 244, 255)
    s[DEAD_EMPTY] = (14, 15, 21)
    s[DEAD_FILL] = (31, 26, 32)
    if k >= 5:
        s[DEAD_FILL, -max(1, k // 6):, :] = s[DEAD_FILL, :, -max(1, k // 6):] = (24, 20, 25)
    _sprites[k] = s
    return s


def tile_pixels(vis, state, k):
    """vis [m, 20, 10] cell ids, state [m] (0 alive, 1 dead, 2 just died) -> tiles [m, PY*k, PX*k, 3]."""
    m = len(vis)
    dead, dying = state == 1, state == 2
    if dead.any():
        vis = vis.copy()
        vis[dead] = np.where(vis[dead] != 0, DEAD_FILL, DEAD_EMPTY)
    img = sprites(k)[vis].transpose(0, 1, 3, 2, 4, 5).reshape(m, 20 * k, 10 * k, 3)
    tiles = np.empty((m, PY * k, PX * k, 3), np.uint8)
    tiles[:] = BG
    o = k // 2
    bw = max(1, k // 12) if k >= 6 else 0
    if bw:
        tiles[:, o - bw:o + 20 * k + bw, o - bw:o + 10 * k + bw] = (78, 84, 102)
    tiles[:, o:o + 20 * k, o:o + 10 * k] = img
    if dead.any():
        tiles[dead, o - bw:o + 20 * k + bw, o - bw:o] = BG
        tiles[dead, o - bw:o + 20 * k + bw, o + 10 * k:o + 10 * k + bw] = BG
        tiles[dead, o - bw:o, o - bw:o + 10 * k + bw] = BG
        tiles[dead, o + 20 * k:o + 20 * k + bw, o - bw:o + 10 * k + bw] = BG
    if dying.any():
        tiles[dying] = np.minimum(tiles[dying] * 0.45 + (150, 18, 24), 255).astype(np.uint8)
    return tiles


def grid_layout(n):
    best = None
    for cols in range(1, n + 1):
        rows = -(-n // cols)
        s = min((W - 60) / (cols * PX), (H - 250) / (rows * PY))
        if best is None or s > best[2]:
            best = (cols, rows, s)
    cols, rows, s = best
    return cols, rows, float(np.floor(s)) if s >= 2 else s


def zoom_path(t, a, b):
    """Camera from a=(cx, cy, s) to b. The center moves with the view width so the zoom stays smooth."""
    e = t * t * (3 - 2 * t)
    s = float(np.exp(np.log(a[2]) + (np.log(b[2]) - np.log(a[2])) * e))
    span = 1 / b[2] - 1 / a[2]
    w = (1 / s - 1 / a[2]) / span if span else e
    return a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, s


def blend(color, a):
    return tuple(int(BG[i] + (color[i] - BG[i]) * a) for i in range(3))


class Camera:
    def __init__(self, n, winner):
        self.cols, self.rows, self.s_full = grid_layout(n)
        self.full = (self.cols * PX / 2, self.rows * PY / 2, self.s_full)
        r, c = divmod(winner, self.cols)
        self.close = ((c + 0.5) * PX, (r + 0.5) * PY, (H - 380) / PY)

    def at(self, frame, end_start=None):
        if end_start is not None:
            return zoom_path(min(1.0, (frame - end_start) / ZOOM), self.full, self.close)
        if frame < CLOSE:
            return self.close
        if frame < CLOSE + PULL:
            return zoom_path((frame - CLOSE) / PULL, self.close, self.full)
        return self.full


def draw_view(ids, state, cam, camera, n):
    cx, cy, s = cam
    cols, rows = camera.cols, camera.rows
    k = int(min(48, max(1, np.ceil(s - 1e-6))))
    x0, y0 = cx - W / 2 / s, cy - CY / s
    x1, y1 = cx + W / 2 / s, cy + (H - CY) / s
    c0, c1 = max(0, int(x0 // PX)), min(cols, int(np.ceil(x1 / PX)))
    r0, r1 = max(0, int(y0 // PY)), min(rows, int(np.ceil(y1 / PY)))
    if c0 >= c1 or r0 >= r1:
        return Image.new("RGB", (W, H), tuple(BG))
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    t = (rr * cols + cc).ravel()
    valid = t < n
    tiles = np.empty((len(t), PY * k, PX * k, 3), np.uint8)
    tiles[:] = BG
    tiles[valid] = tile_pixels(ids[t[valid], HIDDEN:], state[t[valid]], k)
    block = tiles.reshape(r1 - r0, c1 - c0, PY * k, PX * k, 3).transpose(0, 2, 1, 3, 4)
    block = block.reshape((r1 - r0) * PY * k, (c1 - c0) * PX * k, 3)
    a = k / s
    data = (a, 0, (x0 - c0 * PX) * k, 0, a, (y0 - r0 * PY) * k)
    resample = Image.NEAREST if abs(a - 1) < 1e-3 else Image.BILINEAR
    return Image.fromarray(block).transform((W, H), Image.AFFINE, data, resample, fillcolor=tuple(BG))


def tile_box(i, cam, camera):
    """Screen rectangle of board i's well."""
    cx, cy, s = cam
    r, c = divmod(i, camera.cols)
    x = (c * PX + 0.5 - cx) * s + W / 2
    y = (r * PY + 0.5 - cy) * s + CY
    return x, y, x + 10 * s, y + 20 * s


def render(seed, choices, out_mp4, title, subtitle, stills=None, fps=30):
    """choices: int array [steps, games] of placement indices. Ends when one game is left."""
    steps, n = choices.shape
    stills = stills or {}

    sim = Batch(n, seed)
    for t in range(steps):
        if sim.alive.sum() <= 1:
            break
        sim.step(choices[t])
    winner = int(np.lexsort((-sim.lines, -sim.placed))[0])
    camera = Camera(n, winner)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", out_mp4],
        stdin=subprocess.PIPE)

    batch = Batch(n, seed)
    died = np.full(n, -10**6)
    frame = 0
    caption = title

    def emit(ids, alive_view, cam, piece_no, end_start=None):
        nonlocal frame
        state = np.where(alive_view, 0, np.where(frame - died < 6, 2, 1))
        if end_start is not None:
            state[winner] = 0
        im = draw_view(ids, state, cam, camera, n)
        px = np.array(im)
        px[:150] = (px[:150] * 0.12 + BG * 0.88).astype(np.uint8)
        px[H - 100:] = (px[H - 100:] * 0.12 + BG * 0.88).astype(np.uint8)
        im = Image.fromarray(px)
        d = ImageDraw.Draw(im)
        s = cam[2]

        if s >= 14:
            fs = min(34, int(s * 0.6))
            for i in range(n):
                x0, y0, x1, y1 = tile_box(i, cam, camera)
                if x1 < 0 or x0 > W or y1 < -s or y0 > H:
                    continue
                col = GOLD if (end_start is not None and i == winner) else (
                    (210, 214, 226) if state[i] == 0 else (96, 100, 114))
                d.text((x0, y1 + 0.55 * s), f"FLY #{i + 1}", font=font("segoeuib.ttf", fs), fill=col, anchor="lm")
                d.text((x1, y1 + 0.55 * s), f"{batch.lines[i]} LINE{'' if batch.lines[i] == 1 else 'S'}", font=font("segoeuib.ttf", fs),
                       fill=col, anchor="rm")
        alive_n = int(alive_view.sum())
        if end_start is None and alive_n <= 16 and s < 8:
            for i in np.nonzero(alive_view)[0]:
                x0, y0, x1, y1 = tile_box(i, cam, camera)
                d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=GOLD, width=3)

        big, mid, mono = font("segoeuib.ttf", 56), font("segoeui.ttf", 28), font("consola.ttf", 32)
        if end_start is None:
            if frame < CLOSE + PULL:
                a1 = min(1, frame / 15) * max(0, 1 - max(0, frame - CLOSE) / 30)
                a2 = min(1, max(0, frame - CLOSE - PULL + 45) / 45)
                if a1 > 0:
                    d.text((W // 2, 70), "This is one fly brain playing Tetris.", font=big,
                           fill=blend((240, 240, 240), a1), anchor="mm")
                if a2 > 0:
                    d.text((W // 2, 70), caption, font=big, fill=blend((240, 240, 240), a2), anchor="mm")
            else:
                d.text((W // 2, 70), caption, font=big, fill=(240, 240, 240), anchor="mm")
                d.text((60, H - 48), f"ALIVE {alive_n:,} / {n:,}", font=mono, fill=(210, 214, 226), anchor="lm")
                d.text((W - 60, H - 48), f"PIECE {piece_no}", font=mono, fill=(150, 156, 170), anchor="rm")
        else:
            a = min(1, (frame - end_start) / 30)
            d.text((W // 2, 66), "LORD OF THE FLIES", font=font("segoeuib.ttf", 72), fill=blend(GOLD, a), anchor="mm")
            how = "last one standing" if alive_n <= 1 else "most lines of any fly"
            d.text((W // 2, H - 52), f"Fly #{winner + 1}, {how}: {batch.lines[winner]} lines",
                   font=font("segoeuib.ttf", 34), fill=blend((240, 240, 240), a), anchor="mm")
        d.text((W // 2, 124), subtitle, font=mid, fill=(140, 146, 160), anchor="mm")

        out = np.asarray(im)
        ff.stdin.write(out.tobytes())
        for key, when in (("close", CLOSE // 2), ("pull", CLOSE + PULL * 2 // 3), ("grid", CLOSE + PULL + 150)):
            if key in stills and frame == when:
                im.save(stills[key])
        frame += 1
        return im

    p = 0
    while p < steps and batch.alive.sum() > 1:
        alive_before = batch.alive.copy()
        before = batch.boards.copy()
        if frame < CLOSE + PULL:
            F = 5
        elif alive_before.sum() <= 8:
            F = 3
        elif p < 90:
            F = 2
        else:
            F = 1
        batch.step(choices[p])
        pc, ch, y = batch.last_piece, batch.last_choice, batch.last_y
        m = np.nonzero(alive_before & (y >= 0))[0]
        for f in range(F):
            cam = camera.at(frame)
            if f == F - 1:
                ids = batch.last_locked.copy()
                ids[batch.last_full] = FLASH
                newly = alive_before & ~batch.alive
                died[newly] = frame
                emit(ids, batch.alive | newly, cam, p + 1)
                continue
            ids = before.copy()
            if len(m):
                u = f / F
                cells, xt = CELLS[pc][ch[m]], XS[pc][ch[m]]
                xs = (COLS - WIDTHS[pc][ch[m]]) // 2
                yt = y[m]
                ys = np.minimum(HIDDEN, yt)
                xn = np.rint(xs + (xt - xs) * min(1.0, u / 0.3)).astype(int)
                yn = np.rint(ys + (yt - ys) * max(0.0, (u - 0.3) / 0.7)).astype(int)
                if cam[2] >= 10:
                    _paint(ids, m, yt, xt, cells, GHOST)
                _paint(ids, m, yn, xn, cells, pc + 1)
            emit(ids, alive_before, cam, p + 1)
        p += 1

    end_start = frame
    final = None
    for _ in range(ZOOM + HOLD):
        final = emit(batch.boards, batch.alive, camera.at(frame, end_start), p, end_start)
    ff.stdin.close()
    ff.wait()
    if "final" in stills:
        final.save(stills["final"])
    return winner


def _paint(ids, m, yy, xx, cells, val):
    rows = yy[:, None] + cells[:, :, 0]
    cols = xx[:, None] + cells[:, :, 1]
    bb = np.broadcast_to(m[:, None], rows.shape)
    cur = ids[bb, rows, cols]
    ok = (cur == 0) | (cur == GHOST)
    ids[bb[ok], rows[ok], cols[ok]] = val
