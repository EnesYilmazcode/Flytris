"""Phase 1d: does the visual system carry the board AFTER a piece lands, and how fast?

The after-state head shows the fly every place the current piece could land and scores
each view. Here: after-state boards (the phase 1b snapshots with a random valid placement
of their piece dropped in) go to the eyes with R7/R8 silent, and a ridge decoder reads
relative column heights, holes per column and bumpiness back out of the visual projection
groups at several simulation lengths. Also measures brain speed at the batch sizes the
after-state player uses.

    python -X utf8 scripts/decode_after.py --save runs/readout_groups_after.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flytris.brain import Brain, load_neurons
from flytris.eyes import Eyes
from flytris.fly_player import neuron_groups
from flytris.tetris import H, PLACEMENTS, ROWS, column_tops, drop

ap = argparse.ArgumentParser()
ap.add_argument("--readout", default="visual_projection")
ap.add_argument("--thresholds", default="5,10")
ap.add_argument("--steps", default="30,50,150")
ap.add_argument("--board-hz", type=float, default=500)
ap.add_argument("--w-scale", type=float, default=1.0)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--speed-batches", default="256,384")
ap.add_argument("--save", default="")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]


def after_states(boards, pieces, rng):
    out = boards.copy()
    for i in range(len(boards)):
        opts = rng.permutation(len(PLACEMENTS[pieces[i]]))
        for j in opts:
            nb, _, dead = drop(boards[i:i + 1], pieces[i], j)
            if not dead[0]:
                out[i] = nb[0]
                break
    return out


def targets(boards):
    top = column_tops(boards)
    heights = np.minimum(H - top, ROWS).astype(np.float64)
    below = np.arange(H)[None, :, None] > top[:, None, :]
    holes = (below & (boards == 0)).sum(1).astype(np.float64)
    bump = np.abs(np.diff(heights, axis=1)).sum(1, keepdims=True)
    return heights - heights.mean(1, keepdims=True), holes, bump


def explained(y, f):
    yc, fc = y - y.mean(0), f - f.mean(0)
    beta, *_ = np.linalg.lstsq(yc, fc, rcond=None)
    r2 = 1 - ((fc - yc @ beta) ** 2).sum(0) / ((fc ** 2).sum(0) + 1e-9)
    return np.where(f.std(0) > 0, r2, -1)


def ridge_fit(x, y, lam):
    xm, xs, ym = x.mean(0), x.std(0) + 1e-6, y.mean(0)
    z = (x - xm) / xs
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return lambda q: ((q - xm) / xs) @ w + ym


def r2_percol(pred, y):
    return float(np.mean(1 - ((y - pred) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-9)))


def r2_pooled(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def ridge_r2(xtr, xte, ytr, yte, score):
    folds = np.array_split(np.random.default_rng(0).permutation(len(xtr)), 5)
    best = None
    for lam in [0.1, 1, 10, 100, 1000, 10000]:
        s = np.mean([score(ridge_fit(np.delete(xtr, f, 0), np.delete(ytr, f, 0), lam)(xtr[f]), ytr[f])
                     for f in folds])
        if best is None or s > best[0]:
            best = (s, lam)
    return round(score(ridge_fit(xtr, ytr, best[1])(xte), yte), 3)


def decode(f, y, tr, te):
    (h, ho, b) = y
    return {"rel_height_r2": ridge_r2(f[tr], f[te], h[tr], h[te], r2_percol),
            "holes_r2": ridge_r2(f[tr], f[te], ho[tr], ho[te], r2_pooled),
            "bumpiness_r2": ridge_r2(f[tr], f[te], b[tr], b[te], r2_pooled)}


z = np.load(root / "runs/phase1b_boards.npz")
rng = np.random.default_rng(4)
sel_b = after_states(z["sel_b"], z["sel_p"], rng)
prac_b = after_states(z["prac_b"], z["prac_p"], rng)
boards = np.concatenate([sel_b, prac_b])
y_sel, y_prac = targets(sel_b), targets(prac_b)
tr, te = slice(0, 500), slice(500, 600)

neurons = load_neurons(["idx", "type", "side", "superclass"])
rec_np, group_np, names = neuron_groups(neurons, args.readout)
rec_idx = torch.as_tensor(rec_np, device="cuda")
group_of = torch.as_tensor(group_np, device="cuda")
eyes = Eyes(board_hz=args.board_hz, piece_hz=0.0)
zeros = np.zeros(len(boards), np.int64)
steps_list = [int(s) for s in args.steps.split(",")]
results = {"speed": {}, "decode": {}}
saved = {}

for thr in [int(t) for t in args.thresholds.split(",")]:
    torch.cuda.empty_cache()
    brain = Brain(thr, args.w_scale)
    for B in [int(b) for b in args.speed_batches.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        p = eyes.probs(eyes.channels(np.repeat(boards[:1], B, 0), zeros[:B]))
        brain.run(eyes.in_idx, p, 3, rec_idx, eyes.phase)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        brain.run(eyes.in_idx, p, 20, rec_idx, eyes.phase)
        torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / 20
        results["speed"][f"syn>={thr} B={B}"] = {
            "ms_per_step": round(per * 1000, 2), "fly_sec_per_wall_sec": round(B * 1e-3 / per, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(f"syn>={thr} B={B}", results["speed"][f"syn>={thr} B={B}"], flush=True)
    for T in steps_list:
        t0 = time.time()
        feats = np.zeros((len(boards), len(names)), np.float32)
        for s in range(0, len(boards), 256):
            p = eyes.probs(eyes.channels(boards[s:s + 256], zeros[s:s + 256]))
            counts, _ = brain.run(eyes.in_idx, p, T, rec_idx, eyes.phase)
            pooled = torch.zeros((len(names), counts.shape[1]), device="cuda")
            pooled.index_add_(0, group_of, counts)
            feats[s:s + 256] = pooled.T.cpu().numpy()
        f_sel, f_prac = feats[:200], feats[200:]
        chosen = np.argsort(-explained(np.concatenate(y_sel, 1), f_sel))[:args.k]
        r = {f"chosen_{args.k}": decode(f_prac[:, chosen], y_prac, tr, te),
             "all_groups": decode(f_prac, y_prac, tr, te),
             "groups_firing": int((f_prac.sum(0) > 0).sum()),
             "seconds": round(time.time() - t0, 1)}
        results["decode"][f"syn>={thr} T={T}"] = r
        saved[thr, T] = (chosen, f_sel[:, chosen].mean(0), f_sel[:, chosen].std(0) + 1e-3)
        print(f"syn>={thr} T={T}", json.dumps(r), flush=True)
    del brain

x = np.concatenate([eyes.channels(prac_b, zeros[:600]).T[:, :20]], 1)
results["reference: heights + holes channels, no brain"] = decode(x, y_prac, tr, te)


def score(thr, T):
    return results["decode"][f"syn>={thr} T={T}"][f"chosen_{args.k}"]["rel_height_r2"]


thrs = [int(t) for t in args.thresholds.split(",")]
pick_thr = thrs[0]
if len(thrs) > 1 and score(thrs[1], max(steps_list)) >= 0.9 * score(thrs[0], max(steps_list)):
    pick_thr = thrs[1]
full = score(pick_thr, max(steps_list))
pick_T = next(T for T in sorted(steps_list) if score(pick_thr, T) >= 0.8 * full)
results["chosen"] = {"threshold": pick_thr, "steps": pick_T, "rel_height_r2": score(pick_thr, pick_T)}
results["config"] = vars(args)
(root / "runs/phase1d.json").write_text(json.dumps(results, indent=2))
print("chosen", results["chosen"], flush=True)

if args.save:
    chosen, mean, std = saved[pick_thr, pick_T]
    np.savez(root / args.save, threshold=pick_thr, steps=pick_T, board_hz=args.board_hz, piece_hz=0.0,
             w_scale=args.w_scale, superclass=args.readout, groups=chosen, mean=mean, std=std,
             names=np.array([f"{names[g][0]}_{names[g][1]}" for g in chosen]))
    print("saved", args.save)
