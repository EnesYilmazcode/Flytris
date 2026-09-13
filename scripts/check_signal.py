"""Phase 1: does a Tetris board shown to the eyes reach the descending neurons?"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flytris import placeholder
from flytris.brain import Brain, load_neurons
from flytris.eyes import COLS, Eyes
from flytris.tetris import Batch

ap = argparse.ArgumentParser()
ap.add_argument("--thresholds", default="10,5")
ap.add_argument("--w-scale", type=float, default=1.0)
ap.add_argument("--board-hz", type=float, default=200.0)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--inhibitory-photoreceptors", action="store_true")
ap.add_argument("--speed-only", action="store_true")
args = ap.parse_args()

NAMED = ["DNa01", "DNa02", "DNp20", "DNb05", "DNb06", "DNa10", "DNp01", "DNpe017"]
neurons = load_neurons(["idx", "type", "side", "superclass"])
dn = neurons[neurons.superclass == "descending_neuron"].reset_index(drop=True)
dn_idx = torch.as_tensor(dn.idx.values, device="cuda")
print(f"descending neurons: {len(dn)}")

t0 = time.time()
eyes = Eyes(board_hz=args.board_hz)
print(f"eye map {eyes.info}, inputs {len(eyes.in_idx)}, built/loaded in {time.time() - t0:.1f}s")


def values_for(boards, pieces):
    return eyes.channels(boards, pieces)


def timed_run(brain, p, **kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    counts, rate = brain.run(eyes.in_idx, p, args.steps, dn_idx, **kw)
    torch.cuda.synchronize()
    return counts.cpu().numpy(), float(rate), (time.perf_counter() - t) / args.steps


def cosdist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return 1 - a @ b / (na * nb)


for thr in [int(x) for x in args.thresholds.split(",")]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    brain = Brain(thr, args.w_scale, not args.inhibitory_photoreceptors)
    print(f"\n=== threshold syn>={thr}: {brain.nnz:,} edges, loaded in {time.time() - t0:.1f}s ===")

    full = np.ones((len(eyes.channels(np.zeros((1, 22, 10), np.uint8), [0])), 1), np.float32)
    full[2 * COLS:] = 0
    full[2 * COLS] = 1
    for B in (1, 64):
        p = eyes.probs(np.repeat(full, B, 1))
        timed_run(brain, p[:, :1])
        _, rate, per = timed_run(brain, p)
        print(f"B={B:3d}: {per * 1000:6.2f} ms/step, fly-sec/wall-sec {B * 1e-3 / per:6.2f}, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    if args.speed_only:
        del brain
        continue

    B = 32
    p = eyes.probs(np.repeat(full, B, 1))
    base, base_rate, _ = timed_run(brain, p, silent_inputs=True)
    strong, strong_rate, _ = timed_run(brain, p)
    frac = strong_rate / brain.n
    print(f"baseline: DNs firing {int((base.sum(1) > 0).sum())}, mean spikes/ms {base_rate:.1f}")
    print(f"strong input: DNs firing (any trial) {int((strong.sum(1) > 0).sum())}, "
          f"DNs firing every trial {int((strong > 0).all(1).sum())}, "
          f"mean spikes/ms {strong_rate:.0f} ({frac * 100:.2f}% of neurons per ms), "
          f"max DN rate {strong.max() / args.steps * 1000:.0f} Hz")

    left = np.zeros_like(full)
    right = np.zeros_like(full)
    left[:5], left[COLS:COLS + 5] = 1, 1
    right[5:COLS], right[COLS + 5:2 * COLS] = 1, 1
    left[2 * COLS] = right[2 * COLS] = 1
    cl, _, _ = timed_run(brain, eyes.probs(np.repeat(left, B, 1)))
    cr, _, _ = timed_run(brain, eyes.probs(np.repeat(right, B, 1)))
    diff = cl.mean(1) - cr.mean(1)
    se = np.sqrt(cl.var(1) / B + cr.var(1) / B) + 1e-9
    zs = diff / se
    active = (cl.sum(1) + cr.sum(1)) > 0
    print(f"left vs right: DNs active {int(active.sum())}, |z|>3: {int((np.abs(zs) > 3).sum())}")
    for name in NAMED:
        rows = dn.index[dn.type == name]
        cells = ", ".join(f"{dn.side[i]}: L-stim {cl[i].mean():.1f} R-stim {cr[i].mean():.1f}" for i in rows)
        print(f"  {name:8s} {cells}")

    rng = np.random.default_rng(3)
    batch = Batch(20, 11)
    temp = np.exp(rng.uniform(np.log(0.3), np.log(5), 20))
    for _ in range(int(rng.integers(40, 60))):
        batch.step(placeholder.choose(batch, temp, rng))
    reps = 4
    vals = eyes.channels(np.repeat(batch.boards, reps, 0), np.full(20 * reps, batch.piece))
    cb, _, _ = timed_run(brain, eyes.probs(vals))
    vecs = cb.T.reshape(20, reps, -1)
    within = [cosdist(vecs[i, a], vecs[i, b]) for i in range(20) for a in range(reps) for b in range(a + 1, reps)]
    between = [cosdist(vecs[i, 0], vecs[j, 1]) for i in range(20) for j in range(20) if i != j]
    print(f"boards: within-board cos dist {np.nanmean(within):.3f}, between-board {np.nanmean(between):.3f}, "
          f"ratio {np.nanmean(between) / max(np.nanmean(within), 1e-9):.2f}")
    del brain
