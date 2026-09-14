"""Fit a fly-brain readout to rankings from the trained no-brain controller.

The frozen brain is run once per candidate afterstate. A ridge regression then
fits the 32 effective readout weights to the no-brain champion's within-board
preferences. Evolution can start around this useful policy instead of zero.

Run after GPU baseline training is stopped:
    python -X utf8 scripts/warmstart_after.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.eyes import board_channels  # noqa: E402
from flytris.tetris import PLACEMENTS, drop  # noqa: E402


def candidates(boards, pieces):
    out, groups = [], []
    for i, (board, piece) in enumerate(zip(boards, pieces)):
        for move in range(len(PLACEMENTS[int(piece)])):
            after, _, dead = drop(board[None], int(piece), move)
            if not dead[0]:
                out.append(after[0])
                groups.append(i)
    return np.asarray(out, np.uint8), np.asarray(groups, np.int32)


def center_groups(values, groups):
    sums = np.zeros((groups.max() + 1, *values.shape[1:]), np.float64)
    np.add.at(sums, groups, values)
    count = np.bincount(groups, minlength=len(sums)).reshape((-1,) + (1,) * (values.ndim - 1))
    return values - sums[groups] / count[groups]


def fit_ridge(x, y, lam):
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def group_accuracy(score, target, groups, chosen_groups):
    good = total = 0
    for group in chosen_groups:
        idx = np.flatnonzero(groups == group)
        if not len(idx):
            continue
        good += int(idx[np.argmax(score[idx])] == idx[np.argmax(target[idx])])
        total += 1
    return good / total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=str(ROOT / "runs/phase1b_boards.npz"))
    ap.add_argument("--teacher", default=str(ROOT / "runs/nobrain_after/champion.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs/warmstart_after.npz"))
    ap.add_argument("--parents", type=int, default=800)
    ap.add_argument("--max-batch", type=int, default=384)
    ap.add_argument("--cache", default=str(ROOT / "runs/warmstart_features.npz"))
    args = ap.parse_args(argv)

    with np.load(args.boards) as source:
        boards = np.concatenate([source["sel_b"], source["prac_b"]])[:args.parents]
        pieces = np.concatenate([source["sel_p"], source["prac_p"]])[:args.parents]
    after, groups = candidates(boards, pieces)
    print(f"generated {len(after)} valid afterstates from {len(boards)} parent boards", flush=True)

    with np.load(args.teacher) as data:
        teacher = data["params"]
    direct = board_channels(after, np.zeros(len(after), np.int64))[:, :20]
    target = direct @ teacher[:-1] + teacher[-1]

    cache = Path(args.cache)
    if cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved["groups"], groups) and len(saved["features"]) == len(after):
                features = saved["features"].astype(np.float64)
                print(f"loaded cached brain features from {cache}", flush=True)
            else:
                raise ValueError(f"cached features at {cache} do not match candidate boards")
    else:
        from flytris.fly_player import AfterstateFlyPlayer
        player = AfterstateFlyPlayer(max_batch=args.max_batch)
        features = player.board_features(after).astype(np.float64)
        np.savez_compressed(cache, features=features.astype(np.float32), groups=groups)
        print(f"cached brain features at {cache}", flush=True)

    x = center_groups(features, groups)
    y = center_groups(target[:, None], groups)[:, 0]
    parent_ids = np.arange(len(boards))
    rng = np.random.default_rng(2026)
    rng.shuffle(parent_ids)
    train_groups, test_groups = parent_ids[:int(0.8 * len(parent_ids))], parent_ids[int(0.8 * len(parent_ids)):]
    train = np.isin(groups, train_groups)

    best = None
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        w = fit_ridge(x[train], y[train], lam)
        acc = group_accuracy(x @ w, y, groups, test_groups)
        print(f"lambda {lam:g}: held-out top-move agreement {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = acc, lam
    weights = fit_ridge(x, y, best[1]).astype(np.float32)
    params = np.r_[weights, np.float32(0)]
    meta = {"parents": len(boards), "afterstates": len(after), "lambda": best[1],
            "heldout_top_move_agreement": best[0], "teacher": args.teacher}
    np.savez(args.out, params=params, meta=json.dumps(meta))
    print(f"saved {args.out}: {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
