"""Phase 1b: does a readout neuron population carry the board?

Shows practice boards to the brain, pools spike counts by (type, side), picks readout
groups on a separate selection set (5/8 by how well the board layout explains a group,
3/8 by how well the piece does), then ridge-decodes relative column heights and the
current piece on held-out boards. Also runs a randomly rewired brain as a wiring control.

    python -X utf8 scripts/decode_check.py --readout visual_projection --save runs/readout_groups.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flytris import placeholder
from flytris.brain import Brain, load_neurons
from flytris.eyes import Eyes
from flytris.fly_player import PRACTICE_SEED0, SELECTION_SEED0, neuron_groups
from flytris.tetris import H, ROWS, Batch, column_tops

ap = argparse.ArgumentParser()
ap.add_argument("--readout", default="visual_projection")
ap.add_argument("--thresholds", default="5,10")
ap.add_argument("--rewire", type=int, default=1, help="also run a rewired copy of the first threshold")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--board-hz", type=float, default=500)
ap.add_argument("--piece-hz", type=float, default=500)
ap.add_argument("--w-scale", type=float, default=1.0)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--save", default="", help="write the chosen groups here")
ap.add_argument("--tag", default="")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
(root / "runs").mkdir(exist_ok=True)
CACHE = root / "runs/phase1b_boards.npz"


def practice_boards(seed0, n, rng):
    """One snapshot per game from placeholder games at mixed skill levels."""
    boards, seed = [], seed0
    while sum(len(b) for b in boards) < n:
        batch = Batch(20, seed)
        temp = np.exp(rng.uniform(np.log(0.3), np.log(30), 20))
        snap = rng.integers(3, 90, 20)
        for t in range(90):
            if not batch.alive.any():
                break
            take = batch.alive & (snap == t)
            if take.any():
                boards.append(batch.boards[take].copy())
            c = placeholder.choose(batch, temp, rng)
            batch.step(np.where(batch.alive, c, 0))
        seed += 1
    return np.concatenate(boards)[:n], rng.integers(0, 7, n)


def targets(boards, pieces):
    heights = np.minimum(H - column_tops(boards), ROWS).astype(np.float64)
    return heights - heights.mean(1, keepdims=True), np.eye(7)[pieces]


def explained(y, f):
    """Share of each feature column's variance explained by a linear fit on targets y."""
    yc, fc = y - y.mean(0), f - f.mean(0)
    beta, *_ = np.linalg.lstsq(yc, fc, rcond=None)
    r2 = 1 - ((fc - yc @ beta) ** 2).sum(0) / ((fc ** 2).sum(0) + 1e-9)
    return np.where(f.std(0) > 0, r2, -1)


def pick_groups(f_sel, yh, yp, k):
    n_height = k * 5 // 8
    chosen = list(np.argsort(-explained(yh, f_sel))[:n_height])
    chosen += [g for g in np.argsort(-explained(yp, f_sel)) if g not in chosen][:k - n_height]
    return np.array(chosen)


def ridge_fit(x, y, lam):
    xm, xs, ym = x.mean(0), x.std(0) + 1e-6, y.mean(0)
    z = (x - xm) / xs
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return lambda q: ((q - xm) / xs) @ w + ym


def r2_mean(pred, y):
    return float(np.mean(1 - ((y - pred) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-9)))


def acc(pred, y):
    return float((pred.argmax(1) == y.argmax(1)).mean())


def ridge_cv(x, y, score):
    folds = np.array_split(np.random.default_rng(0).permutation(len(x)), 5)
    best = None
    for lam in [0.1, 1, 10, 100, 1000, 10000]:
        s = np.mean([score(ridge_fit(np.delete(x, f, 0), np.delete(y, f, 0), lam)(x[f]), y[f]) for f in folds])
        if best is None or s > best[0]:
            best = (s, lam)
    return best[1]


def decode(ftr, fte, htr, hte, ptr, pte):
    lh, lp = ridge_cv(ftr, htr, r2_mean), ridge_cv(ftr, ptr, acc)
    return {"rel_height_r2": round(r2_mean(ridge_fit(ftr, htr, lh)(fte), hte), 3),
            "piece_acc": round(acc(ridge_fit(ftr, ptr, lp)(fte), pte), 3)}


if CACHE.exists():
    z = np.load(CACHE)
    prac_b, prac_p, sel_b, sel_p = z["prac_b"], z["prac_p"], z["sel_b"], z["sel_p"]
else:
    rng = np.random.default_rng(1)
    prac_b, prac_p = practice_boards(PRACTICE_SEED0, 600, rng)
    sel_b, sel_p = practice_boards(SELECTION_SEED0, 200, rng)
    np.savez(CACHE, prac_b=prac_b, prac_p=prac_p, sel_b=sel_b, sel_p=sel_p)
yh_sel, yp_sel = targets(sel_b, sel_p)
yh, yp = targets(prac_b, prac_p)
tr, te = slice(0, 500), slice(500, 600)

neurons = load_neurons(["idx", "type", "side", "superclass"])
rec_np, group_np, names = neuron_groups(neurons, args.readout)
rec_idx = torch.as_tensor(rec_np, device="cuda")
group_of = torch.as_tensor(group_np, device="cuda")
eyes = Eyes(board_hz=args.board_hz, piece_hz=args.piece_hz)
boards = np.concatenate([sel_b, prac_b])
pieces = np.concatenate([sel_p, prac_p])
print(f"{args.readout}: {len(rec_np)} neurons in {len(names)} groups", flush=True)

thresholds = [int(x) for x in args.thresholds.split(",")]
runs = [(f"syn>={t}", t, None) for t in thresholds]
if args.rewire:
    runs.append((f"syn>={thresholds[0]} rewired", thresholds[0], 12345))

results, saved = {}, {}
for label, thr, rewire in runs:
    torch.cuda.empty_cache()
    t0 = time.time()
    brain = Brain(thr, args.w_scale, rewire_seed=rewire)
    feats = np.zeros((len(boards), len(names)), np.float32)
    rates = []
    for s in range(0, len(boards), 64):
        p = eyes.probs(eyes.channels(boards[s:s + 64], pieces[s:s + 64]))
        counts, rate = brain.run(eyes.in_idx, p, args.steps, rec_idx, eyes.phase)
        pooled = torch.zeros((len(names), counts.shape[1]), device="cuda")
        pooled.index_add_(0, group_of, counts)
        feats[s:s + 64] = pooled.T.cpu().numpy()
        rates.append(float(rate))
    n_neurons = brain.n
    del brain
    f_sel, f_prac = feats[:200], feats[200:]
    chosen = pick_groups(f_sel, yh_sel, yp_sel, args.k)
    r = {"share_of_neurons_spiking_per_ms": round(float(np.mean(rates)) / n_neurons, 4),
         "groups_firing": int((f_prac.sum(0) > 0).sum()),
         f"chosen_{args.k}": decode(f_prac[tr][:, chosen], f_prac[te][:, chosen], yh[tr], yh[te], yp[tr], yp[te]),
         "all_groups": decode(f_prac[tr], f_prac[te], yh[tr], yh[te], yp[tr], yp[te]),
         "seconds": round(time.time() - t0, 1)}
    c = r[f"chosen_{args.k}"]
    r["gate_pass"] = bool(c["rel_height_r2"] >= 0.3 and c["piece_acc"] >= 0.5)
    results[label] = r
    saved[label] = (thr, chosen, f_sel[:, chosen].mean(0), f_sel[:, chosen].std(0) + 1e-3)
    print(label, json.dumps(r), flush=True)

x_tr, x_te = eyes.channels(prac_b[tr], prac_p[tr]).T, eyes.channels(prac_b[te], prac_p[te]).T
results["reference: the 27 eye channel values, no brain"] = decode(x_tr, x_te, yh[tr], yh[te], yp[tr], yp[te])

pick = f"syn>={thresholds[0]}"
if len(thresholds) > 1:
    a = results[pick][f"chosen_{args.k}"]
    b = results[f"syn>={thresholds[1]}"][f"chosen_{args.k}"]
    if b["rel_height_r2"] > a["rel_height_r2"] + 0.1 and b["piece_acc"] >= a["piece_acc"] - 0.05:
        pick = f"syn>={thresholds[1]}"
results["chosen"] = pick
results["config"] = vars(args)

out = root / f"runs/phase1b{args.tag}.json"
out.write_text(json.dumps(results, indent=2))
if args.save:
    thr, chosen, mean, std = saved[pick]
    np.savez(root / args.save, threshold=thr, steps=args.steps, board_hz=args.board_hz,
             piece_hz=args.piece_hz, w_scale=args.w_scale, superclass=args.readout,
             groups=chosen, mean=mean, std=std,
             names=np.array([f"{names[g][0]}_{names[g][1]}" for g in chosen]))
print("chosen", pick, "->", args.save or "(not saved)", "results in", out.name)
