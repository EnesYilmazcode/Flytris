"""V3 after-state policy and evaluation helpers.

V3 deliberately learns without the fly brain first.  This separates the Tetris
learning problem from the representation problem: if this policy cannot clear 75
lines in 300 pieces on untouched seeds, no GPU brain run is allowed to start.
"""
from dataclasses import dataclass

import numpy as np

from .tetris import COLS, H, HIDDEN, PLACEMENTS, clear, column_tops, drop, lock, piece_stream


FEATURE_NAMES = np.array([
    "aggregate_height",
    "lines_cleared",
    "holes",
    "bumpiness",
    "max_height",
    "wells",
    "row_transitions",
    "column_transitions",
    "blocks_over_holes",
])

# These ranges are intentionally disjoint from V1/V2 and from one another.
IMITATION_TRAIN_SEED0 = 500_000
IMITATION_TEST_SEED0 = 600_000
V3_EVAL_SEED0 = 700_000
V3_RACE_SEED0 = 1_000_000


def afterstate_features(boards, lines):
    """Return interpretable board features for cleared after-states."""
    boards = np.asarray(boards)
    filled = boards != 0
    top = column_tops(boards)
    heights = H - top
    below_top = np.arange(H)[None, :, None] > top[:, None, :]
    holes_mask = below_top & ~filled
    holes = holes_mask.sum((1, 2))
    bumpiness = np.abs(np.diff(heights, axis=1)).sum(1)

    left = np.pad(heights[:, :-1], ((0, 0), (1, 0)), constant_values=H)
    right = np.pad(heights[:, 1:], ((0, 0), (0, 1)), constant_values=H)
    well_depth = np.maximum(np.minimum(left, right) - heights, 0)
    wells = (well_depth * (well_depth + 1) // 2).sum(1)

    visible = filled[:, HIDDEN:]
    row_padded = np.pad(visible, ((0, 0), (0, 0), (1, 1)), constant_values=True)
    row_transitions = (row_padded[:, :, 1:] != row_padded[:, :, :-1]).sum((1, 2))
    col_padded = np.pad(visible, ((0, 0), (1, 1), (0, 0)), constant_values=True)
    column_transitions = (col_padded[:, 1:] != col_padded[:, :-1]).sum((1, 2))

    hole_below = np.cumsum(holes_mask[:, ::-1], axis=1)[:, ::-1] > 0
    blocks_over_holes = (filled & hole_below).sum((1, 2))
    return np.stack([
        heights.sum(1), np.asarray(lines), holes, bumpiness, heights.max(1), wells,
        row_transitions, column_transitions, blocks_over_holes,
    ], axis=1).astype(np.float64)


def enumerate_afterstates(board, piece):
    """Enumerate every legal placement for one board and piece."""
    n = len(PLACEMENTS[piece])
    parents = np.repeat(np.asarray(board)[None], n, axis=0)
    boards, lines, dead = [], [], []
    for move in range(n):
        nb, cleared, lost = drop(parents[move:move + 1], piece, move)
        boards.append(nb[0])
        lines.append(cleared[0])
        dead.append(lost[0])
    return np.stack(boards), np.asarray(lines), np.asarray(dead)


def enumerate_transitions(board, piece):
    """Return landed/flash boards and their corresponding cleared after-states."""
    n = len(PLACEMENTS[piece])
    parents = np.repeat(np.asarray(board)[None], n, axis=0)
    locked_boards, after_boards, lines, dead = [], [], [], []
    for move in range(n):
        locked, _, fits = lock(parents[move:move + 1], piece, move)
        after, cleared, _ = clear(locked)
        lost = ~fits | (after[:, :HIDDEN] != 0).any((1, 2))
        locked_boards.append(locked[0])
        after_boards.append(after[0])
        lines.append(cleared[0])
        dead.append(lost[0])
    return (np.stack(locked_boards), np.stack(after_boards),
            np.asarray(lines), np.asarray(dead))


@dataclass
class LinearAfterstatePolicy:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def scores(self, boards, lines):
        x = (afterstate_features(boards, lines) - self.mean) / self.scale
        return x @ self.weights + self.bias

    def choose(self, board, piece):
        boards, lines, dead = enumerate_afterstates(board, piece)
        score = self.scores(boards, lines)
        score[dead] = -np.inf
        return int(np.argmax(score)) if np.isfinite(score).any() else 0

    def save(self, path, **metadata):
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights,
                 bias=self.bias, feature_names=FEATURE_NAMES, **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["mean"], f["scale"], f["weights"], float(f["bias"]))


def play_policy(policy, seed, cap=300, record_states=False):
    """Play one deterministic game and return its outcome."""
    board = np.zeros((H, COLS), np.uint8)
    stream = piece_stream(seed)
    total_lines = 0
    states = []
    alive = True
    placed = 0
    shaping = 0.0
    moves = []
    for _ in range(cap):
        piece = next(stream)
        if record_states:
            states.append((board.copy(), piece))
        move = policy.choose(board, piece)
        moves.append(move)
        board, lines, dead = drop(board[None], piece, move)
        board = board[0]
        total_lines += int(lines[0])
        placed += 1
        shaping += board_quality(board)
        if dead[0]:
            alive = False
            break
    return {"lines": total_lines, "pieces": placed, "alive": alive,
            "final_board": board, "shaping": shaping,
            "moves": np.asarray(moves, np.uint8), "states": states}


def board_quality(board):
    """Dense per-placement signal; larger is better and zero is a perfect empty board."""
    f = afterstate_features(np.asarray(board)[None], np.zeros(1))[0]
    return float(-(4 * f[2] + f[0] + 0.5 * f[3] + 0.25 * f[4]))


def dense_fitness(lines, pieces, alive, shaping, cap):
    """Task reward plus a bounded trajectory-quality tie-breaker."""
    quality = shaping / max(pieces, 1)
    survival = 250 if alive and pieces >= cap else 0
    return float(lines * 10_000 + pieces * 10 + survival + np.clip(quality, -249, 0))


def racing_elites(primary, extra, shortlist=16, elites=8):
    """Shortlist on one seed, then rank finalists by their mean over all seeds."""
    primary = np.asarray(primary, dtype=np.float64)
    first = np.argsort(-primary, kind="stable")[:min(shortlist, len(primary))]
    extra = np.asarray(extra, dtype=np.float64)
    if extra.ndim == 1:
        extra = extra[:, None]
    aggregate = np.concatenate([primary[first, None], extra], axis=1).mean(1)
    return first[np.argsort(-aggregate, kind="stable")[:min(elites, len(first))]]
