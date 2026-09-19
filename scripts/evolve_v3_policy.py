"""Resumable V3 CEM with multi-seed candidate racing and dense tie-breaking.

This is the CPU reference trainer.  It starts from the imitation model, evaluates
every candidate on one fresh seed, then evaluates only the shortlist on four more
seeds.  There is no clock-based shutdown: stop it manually with Ctrl+C and rerun the
same command to resume from state.npz.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.v3_policy import (
    V3_RACE_SEED0, LinearAfterstatePolicy, dense_fitness, play_policy, racing_elites,
)


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def vector(policy):
    return np.r_[policy.weights, policy.bias].astype(np.float64)


def from_vector(base, params):
    return LinearAfterstatePolicy(base.mean, base.scale, params[:-1], float(params[-1]))


def evaluate(base, params, seed, cap):
    result = play_policy(from_vector(base, params), seed, cap)
    return dense_fitness(result["lines"], result["pieces"], result["alive"],
                         result["shaping"], cap), result["lines"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "v3_nobrain" / "model.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "v3_evolution"))
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--race-seeds", type=int, default=5)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--init-std", type=float, default=0.08)
    args = ap.parse_args(argv)
    if not 0 < args.elites <= args.shortlist <= args.pop:
        ap.error("require elites <= shortlist <= pop")
    if args.race_seeds < 2:
        ap.error("race-seeds must be at least 2")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = LinearAfterstatePolicy.load(args.model)
    state_path = out / "state.npz"
    if state_path.exists():
        with np.load(state_path) as f:
            generation = int(f["generation"])
            mean, std = f["mean"], f["std"]
            champion, champion_score = f["champion"], float(f["champion_score"])
        print(f"Resuming generation {generation}; champion {champion_score:.1f}")
    else:
        generation = 0
        mean = vector(base)
        std = np.full_like(mean, args.init_std)
        champion, champion_score = mean.copy(), -np.inf
        print("Starting V3 evolution from the passing imitation model")

    try:
        while generation < args.gens:
            rng = np.random.default_rng([31, generation])
            candidates = mean + rng.standard_normal((args.pop, len(mean))) * std
            candidates[0] = champion if np.isfinite(champion_score) else mean
            seed0 = V3_RACE_SEED0 + generation * args.race_seeds
            primary, primary_lines = zip(*[
                evaluate(base, p, seed0, args.cap) for p in candidates
            ])
            finalists = np.argsort(-np.asarray(primary), kind="stable")[:args.shortlist]
            extra = np.empty((args.shortlist, args.race_seeds - 1))
            extra_lines = np.empty_like(extra, dtype=np.int32)
            for row, idx in enumerate(finalists):
                for j in range(1, args.race_seeds):
                    extra[row, j - 1], extra_lines[row, j - 1] = evaluate(
                        base, candidates[idx], seed0 + j, args.cap)
            # racing_elites performs the shortlist independently as a consistency check.
            elite = racing_elites(primary, extra, args.shortlist, args.elites)
            scores = np.concatenate([np.asarray(primary)[finalists, None], extra], 1).mean(1)
            best_row = int(np.argmax(scores))
            best_idx = int(finalists[best_row])
            if scores[best_row] > champion_score:
                champion, champion_score = candidates[best_idx].copy(), float(scores[best_row])
                from_vector(base, champion).save(out / "champion.npz", generation=generation,
                                                 race_fitness=champion_score)
            mean = candidates[elite].mean(0)
            std = np.maximum(candidates[elite].std(0), 0.01)
            atomic_npz(out / f"gen_{generation:04d}.npz", seed0=seed0,
                       candidates=candidates, primary_fitness=primary,
                       primary_lines=primary_lines, finalists=finalists,
                       extra_fitness=extra, extra_lines=extra_lines, elites=elite)
            best_lines = np.r_[primary_lines[best_idx], extra_lines[best_row]]
            print(f"gen {generation}: primary best {max(primary_lines)} lines; "
                  f"{args.race_seeds}-seed best mean {best_lines.mean():.1f}; "
                  f"champion fitness {champion_score:.1f}", flush=True)
            generation += 1
            atomic_npz(state_path, generation=generation, mean=mean, std=std,
                       champion=champion, champion_score=champion_score,
                       config=json.dumps(vars(args)))
    except KeyboardInterrupt:
        print("Stopped manually. The last completed generation is checkpointed.")


if __name__ == "__main__":
    main()
