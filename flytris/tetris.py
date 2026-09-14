"""Batched placement-mode Tetris.

Every game in a batch gets the same piece sequence, so a batch is a fair
tournament. A move is a placement index: (rotation, column), then hard drop.
The RNG is mulberry32 so the browser version can replay the same pieces.
"""
import numpy as np

ROWS, COLS, HIDDEN = 20, 10, 2
H = ROWS + HIDDEN
M32 = 0xFFFFFFFF

NAMES = "IOTSZJL"
_BASE = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}


def _rotations(cells):
    out, cur = [], np.array(cells)
    for _ in range(4):
        norm = cur - cur.min(0)
        norm = norm[np.lexsort((norm[:, 1], norm[:, 0]))]
        if not any(np.array_equal(norm, o) for o in out):
            out.append(norm)
        cur = np.stack([cur[:, 1], -cur[:, 0]], 1)
    return out


# PLACEMENTS[piece] = list of (rotation, column, cells)
PLACEMENTS = []
for name in NAMES:
    opts = []
    for r, cells in enumerate(_rotations(_BASE[name])):
        width = cells[:, 1].max() + 1
        for x in range(COLS - width + 1):
            opts.append((r, x, cells))
    PLACEMENTS.append(opts)


def mulberry32(seed):
    a = seed & M32
    while True:
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        yield ((t ^ (t >> 14)) & M32) / 4294967296


def piece_stream(seed):
    rng = mulberry32(seed)
    while True:
        bag = list(range(7))
        for i in range(6, 0, -1):
            j = int(next(rng) * (i + 1))
            bag[i], bag[j] = bag[j], bag[i]
        yield from bag


def column_tops(boards):
    filled = boards != 0
    return np.where(filled.any(1), filled.argmax(1), H)


def lock(boards, piece, placement):
    """Hard-drop one placement onto every board. Returns (locked, landing row, fits)."""
    _, x, cells = PLACEMENTS[piece][placement]
    top = column_tops(boards)
    cols = cells[:, 1] + x
    y = (top[:, cols] - 1 - cells[:, 0]).min(1)
    fits = y >= 0
    out = boards.copy()
    b = np.nonzero(fits)[0]
    out[b[:, None], y[b, None] + cells[:, 0], cols] = piece + 1
    return out, y, fits


def clear(boards):
    """Remove full rows. Returns (boards, lines, full-row mask)."""
    full = (boards != 0).all(2)
    lines = full.sum(1)
    if lines.any():
        order = np.argsort(~full, axis=1, kind="stable")
        boards = np.take_along_axis(boards, order[:, :, None], 1)
        boards[np.arange(H)[None, :] < lines[:, None]] = 0
    return boards, lines, full


def drop(boards, piece, placement):
    """lock + clear. Returns (boards, lines, dead)."""
    locked, _, fits = lock(boards, piece, placement)
    out, lines, _ = clear(locked)
    return out, lines, ~fits | (out[:, :HIDDEN] != 0).any((1, 2))


LINE_SCORE = np.array([0, 40, 100, 300, 1200])


class Batch:
    def __init__(self, n, seed):
        self.n = n
        self.boards = np.zeros((n, H, COLS), np.uint8)
        self.alive = np.ones(n, bool)
        self.lines = np.zeros(n, np.int32)
        self.score = np.zeros(n, np.int64)
        self.placed = np.zeros(n, np.int32)
        self._pieces = piece_stream(seed)
        self.piece = next(self._pieces)

    @property
    def options(self):
        return len(PLACEMENTS[self.piece])

    def step(self, choice):
        """Place the current piece on every alive board. Dead boards ignore their choice.

        Leaves last_piece, last_choice, last_y (landing row, -1 if it did not fit),
        last_locked (boards before line clears) and last_full (cleared rows) for renderers.
        """
        choice = np.asarray(choice).astype(np.int64)
        assert (choice[self.alive] >= 0).all() and (choice[self.alive] < self.options).all()
        self.last_piece, self.last_choice = self.piece, choice
        self.last_y = np.full(self.n, -1)
        self.last_locked = self.boards.copy()
        self.last_full = np.zeros((self.n, H), bool)
        for j in np.unique(choice[self.alive]):
            idx = np.nonzero(self.alive & (choice == j))[0]
            locked, y, fits = lock(self.boards[idx], self.piece, j)
            nb, lines, full = clear(locked)
            dead = ~fits | (nb[:, :HIDDEN] != 0).any((1, 2))
            self.last_y[idx] = np.where(fits, y, -1)
            self.last_locked[idx] = locked
            self.last_full[idx] = full
            self.boards[idx] = nb
            self.lines[idx] += lines
            self.score[idx] += LINE_SCORE[lines]
            self.placed[idx] += 1
            self.alive[idx[dead]] = False
        self.piece = next(self._pieces)
