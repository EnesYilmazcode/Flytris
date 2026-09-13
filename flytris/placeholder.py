"""Stand-in players for building the video pipeline before any fly exists.

A classic hand-tuned Tetris heuristic with a per-player temperature: low
temperature plays well, high temperature plays close to random.
"""
import numpy as np

from .tetris import H, PLACEMENTS, column_tops, drop

# Yiyuan Lee's genetic-algorithm weights: height, lines, holes, bumpiness
W = np.array([-0.510066, 0.760666, -0.35663, -0.184483])


def features(boards, lines):
    top = column_tops(boards)
    heights = H - top
    below = np.arange(H)[None, :, None] > top[:, None, :]
    holes = (below & (boards == 0)).sum((1, 2))
    bump = np.abs(np.diff(heights, axis=1)).sum(1)
    return np.stack([heights.sum(1), lines, holes, bump], 1)


def choose(batch, temperature, rng):
    scores = []
    for j in range(batch.options):
        nb, lines, dead = drop(batch.boards, batch.piece, j)
        s = features(nb, lines) @ W
        scores.append(np.where(dead, -1e9, s))
    scores = np.stack(scores, 1)
    gumbel = -np.log(-np.log(rng.random(scores.shape)))
    return (scores / temperature[:, None] + gumbel).argmax(1)
