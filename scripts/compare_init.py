"""Compare baseline/warm-start blends on fixed validation sequences.

Writes after every seed so an interrupted or detached terminal never loses the result.
This is a model-selection diagnostic only; the validation seeds are not used by the
subsequent evolutionary generations.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.fly_player import GROUPS_AFTER, VALID_SEEDS, AfterstateFlyPlayer, play_afterstate  # noqa: E402
from flytris.tetris import Batch  # noqa: E402


def load_params(path):
    with np.load(path) as data:
        return data["params"].astype(np.float32)


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "runs/train_after/champion.npz"))
    ap.add_argument("--warm", default=str(ROOT / "runs/warmstart_after.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs/init_comparison.json"))
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--max-batch", type=int, default=384)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    args = ap.parse_args(argv)

    baseline, warm = load_params(args.baseline), load_params(args.warm)
    if baseline.shape != warm.shape:
        raise ValueError(f"parameter shapes differ: {baseline.shape} vs {warm.shape}")
    alphas = [float(x) for x in args.alphas.split(",")]
    params = np.stack([(1 - a) * baseline + a * warm for a in alphas]).astype(np.float32)
    out = Path(args.out)
    result = {
        "baseline": args.baseline,
        "warm": args.warm,
        "cap": args.cap,
        "alphas": alphas,
        "seeds": [],
        "lines": [],
        "pieces": [],
    }
    player = AfterstateFlyPlayer(groups_file=str(GROUPS_AFTER), max_batch=args.max_batch)
    for seed in VALID_SEEDS:
        batch = Batch(len(params), seed)
        play_afterstate([batch], [params], player, args.cap)
        result["seeds"].append(seed)
        result["lines"].append(batch.lines.astype(int).tolist())
        result["pieces"].append(batch.placed.astype(int).tolist())
        array = np.asarray(result["lines"])
        result["mean_lines"] = array.mean(0).tolist()
        save_json(out, result)
        print(f"seed {seed}: lines {result['lines'][-1]}; means {result['mean_lines']}", flush=True)

    fitness = np.asarray(result["lines"]) * 1000 + np.asarray(result["pieces"])
    result["mean_fitness"] = fitness.mean(0).tolist()
    winner = int(np.argmax(result["mean_fitness"]))
    result["winner"] = {"index": winner, "alpha": alphas[winner]}
    save_json(out, result)
    np.savez(out.with_suffix(".npz"), params=params[winner], alpha=alphas[winner],
             baseline=args.baseline, warm=args.warm)
    print(f"winner alpha={alphas[winner]:g}; saved {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
