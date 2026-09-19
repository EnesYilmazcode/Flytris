"""Learn and gate the V3 no-brain policy by imitation on disjoint seed sets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.placeholder import W
from flytris.v3_policy import (
    FEATURE_NAMES, IMITATION_TEST_SEED0, IMITATION_TRAIN_SEED0, V3_EVAL_SEED0,
    LinearAfterstatePolicy, afterstate_features, enumerate_afterstates, play_policy,
)


class Teacher:
    def choose(self, board, piece):
        boards, lines, dead = enumerate_afterstates(board, piece)
        score = afterstate_features(boards, lines)[:, :4] @ W
        score[dead] = -np.inf
        return int(np.argmax(score)) if np.isfinite(score).any() else 0


def collect(seed0, parents, cap=300):
    teacher = Teacher()
    states = []
    seed = seed0
    while len(states) < parents:
        result = play_policy(teacher, seed, cap=cap, record_states=True)
        states.extend(result["states"])
        seed += 1
    return states[:parents]


def design(states):
    xs, ys, groups = [], [], []
    for group, (board, piece) in enumerate(states):
        boards, lines, dead = enumerate_afterstates(board, piece)
        x = afterstate_features(boards, lines)
        keep = ~dead
        xs.append(x[keep])
        ys.append(x[keep, :4] @ W)
        groups.append(np.full(keep.sum(), group, np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def agreement(policy, states):
    teacher = Teacher()
    same = 0
    for board, piece in states:
        same += policy.choose(board, piece) == teacher.choose(board, piece)
    return same / len(states)


def summarize(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": int(a.min()), "max": int(a.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "v3_nobrain"))
    ap.add_argument("--train-parents", type=int, default=600)
    ap.add_argument("--test-parents", type=int, default=200)
    ap.add_argument("--eval-games", type=int, default=50)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--gate", type=float, default=75.0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.train_parents} teacher states...")
    train_states = collect(IMITATION_TRAIN_SEED0, args.train_parents, args.cap)
    x, y, _ = design(train_states)
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    fit = np.column_stack([z, np.ones(len(z))])
    solution = np.linalg.lstsq(fit, y, rcond=None)[0]
    policy = LinearAfterstatePolicy(mean, scale, solution[:-1], float(solution[-1]))

    print(f"Testing on {args.test_parents} unseen teacher states...")
    test_states = collect(IMITATION_TEST_SEED0, args.test_parents, args.cap)
    top_move_agreement = agreement(policy, test_states)
    model_path = out / "model.npz"
    policy.save(model_path, method="least_squares_teacher_imitation",
                train_parents=args.train_parents, test_parents=args.test_parents,
                top_move_agreement=top_move_agreement)

    print(f"Evaluating {args.eval_games} untouched 300-piece games...")
    learned, teacher = [], []
    learned_alive = teacher_alive = 0
    reference = Teacher()
    for i in range(args.eval_games):
        seed = V3_EVAL_SEED0 + i
        a = play_policy(policy, seed, args.cap)
        b = play_policy(reference, seed, args.cap)
        learned.append(a["lines"])
        teacher.append(b["lines"])
        learned_alive += a["alive"]
        teacher_alive += b["alive"]
        print(f"  {i + 1:02d}/{args.eval_games}: learned {a['lines']}, teacher {b['lines']}", flush=True)

    results = {
        "version": 3,
        "cap": args.cap,
        "gate_mean_lines": args.gate,
        "passed": float(np.mean(learned)) >= args.gate,
        "seed_range": [V3_EVAL_SEED0, V3_EVAL_SEED0 + args.eval_games - 1],
        "feature_names": FEATURE_NAMES.tolist(),
        "imitation": {"train_parents": args.train_parents, "test_parents": args.test_parents,
                      "unseen_top_move_agreement": top_move_agreement},
        "learned": {**summarize(learned), "survived": int(learned_alive), "lines": learned},
        "teacher": {**summarize(teacher), "survived": int(teacher_alive), "lines": teacher},
    }
    (out / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("feature_names",)}, indent=2))
    if not results["passed"]:
        raise SystemExit("V3 gate failed; do not start a cloud GPU run")


if __name__ == "__main__":
    main()
