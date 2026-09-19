"""Distill the passing V3 policy into the cached fly-brain representation.

This is a representation gate, not an overnight trainer.  If held-out top-move
agreement is below --gate, the script still saves diagnostics and a warm start but
returns failure so weak brain features are not sent to a cloud evolution run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.tetris import PLACEMENTS, drop
from flytris.v3_policy import LinearAfterstatePolicy, afterstate_features


def candidates(boards, pieces):
    out, cleared, groups = [], [], []
    for i, (board, piece) in enumerate(zip(boards, pieces)):
        for move in range(len(PLACEMENTS[int(piece)])):
            after, lines, dead = drop(board[None], int(piece), move)
            if not dead[0]:
                out.append(after[0])
                cleared.append(lines[0])
                groups.append(i)
    return np.asarray(out, np.uint8), np.asarray(cleared), np.asarray(groups, np.int32)


def centered(values, groups):
    sums = np.zeros((groups.max() + 1, *values.shape[1:]), np.float64)
    np.add.at(sums, groups, values)
    count = np.bincount(groups, minlength=len(sums)).reshape((-1,) + (1,) * (values.ndim - 1))
    return values - sums[groups] / count[groups]


def accuracy(score, target, groups, selected):
    correct = total = 0
    for group in selected:
        idx = np.flatnonzero(groups == group)
        if not len(idx):
            continue
        correct += idx[np.argmax(score[idx])] == idx[np.argmax(target[idx])]
        total += 1
    return float(correct / total)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=str(ROOT / "runs" / "phase1b_boards.npz"))
    ap.add_argument("--features", default=str(ROOT / "runs" / "warmstart_features.npz"))
    ap.add_argument("--teacher", default=str(ROOT / "runs" / "v3_nobrain" / "model.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "v3_brain_init.npz"))
    ap.add_argument("--report", default=str(ROOT / "runs" / "v3_brain_gate.json"))
    ap.add_argument("--gate", type=float, default=0.50)
    args = ap.parse_args(argv)

    with np.load(args.boards) as source:
        boards = np.concatenate([source["sel_b"], source["prac_b"]])
        pieces = np.concatenate([source["sel_p"], source["prac_p"]])
    after, lines, groups = candidates(boards, pieces)
    with np.load(args.features) as cache:
        if not np.array_equal(groups, cache["groups"]):
            raise ValueError("cached brain features do not match the candidate boards")
        brain = cache["features"].astype(np.float64)
    teacher = LinearAfterstatePolicy.load(args.teacher)
    target = teacher.scores(after, lines)
    x, y = centered(brain, groups), centered(target[:, None], groups)[:, 0]

    ids = np.arange(len(boards))
    np.random.default_rng(20260915).shuffle(ids)
    split = int(0.8 * len(ids))
    train_ids, test_ids = ids[:split], ids[split:]
    train = np.isin(groups, train_ids)
    trials = []
    best = None
    for lam in (0.01, 0.1, 1, 10, 100, 1000):
        w = np.linalg.solve(x[train].T @ x[train] + lam * np.eye(x.shape[1]), x[train].T @ y[train])
        acc = accuracy(x @ w, y, groups, test_ids)
        trials.append({"lambda": lam, "heldout_top_move_agreement": acc})
        if best is None or acc > best[0]:
            best = acc, lam, w
    passed = best[0] >= args.gate
    params = np.r_[best[2], 0].astype(np.float32)
    np.savez(args.out, params=params, heldout_top_move_agreement=best[0],
             representation_gate=args.gate, passed=passed, teacher=args.teacher)
    report = {"passed": passed, "gate": args.gate, "heldout_top_move_agreement": best[0],
              "parents": len(boards), "afterstates": len(after), "features": brain.shape[1],
              "best_lambda": best[1], "trials": trials,
              "next_action": ("cloud candidate racing is allowed" if passed else
                              "improve brain representation before cloud evolution")}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Brain representation gate failed; cloud evolution remains locked")


if __name__ == "__main__":
    main()
