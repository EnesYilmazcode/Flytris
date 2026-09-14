"""Phase 1c: strip input vs retinotopic image input, real vs rewired wiring, VPN vs DN readout.

Same boards, seeds, group selection and ridge decoder as decode_check.py. Two rewire
controls: "full" permutes every edge's target (photoreceptors can reach the readout
directly), "fair" keeps photoreceptor outputs real and permutes everything else.

    python -X utf8 scripts/phase1c.py
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
from flytris.eyes_image import EyesImage
from flytris.fly_player import neuron_groups
from flytris.tetris import H, ROWS, column_tops

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=int, default=5)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--board-hz", type=float, default=500)
ap.add_argument("--piece-hz", type=float, default=500)
ap.add_argument("--image-hz", type=float, default=None, help="board rate for image input (default --board-hz)")
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--inputs", default="strips,image")
ap.add_argument("--wirings", default="real,full,fair")
ap.add_argument("--readouts", default="visual_projection,descending_neuron")
ap.add_argument("--tag", default="")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]


def targets(boards, pieces):
    heights = np.minimum(H - column_tops(boards), ROWS).astype(np.float64)
    return heights - heights.mean(1, keepdims=True), np.eye(7)[pieces]


def explained(y, f):
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
    return {"r2": round(r2_mean(ridge_fit(ftr, htr, lh)(fte), hte), 3),
            "piece": round(acc(ridge_fit(ftr, ptr, lp)(fte), pte), 3)}


with np.load(root / "runs/phase1b_boards.npz") as z:
    prac_b, prac_p, sel_b, sel_p = z["prac_b"], z["prac_p"], z["sel_b"], z["sel_p"]
yh_sel, yp_sel = targets(sel_b, sel_p)
yh, yp = targets(prac_b, prac_p)
tr, te = slice(0, 500), slice(500, 600)
boards = np.concatenate([sel_b, prac_b])
pieces = np.concatenate([sel_p, prac_p])

neurons = load_neurons(["idx", "type", "side", "superclass"])
recs, groups, spans, off = [], [], {}, 0
for ro in args.readouts.split(","):
    rec, g, names = neuron_groups(neurons, ro)
    recs.append(rec)
    groups.append(g + off)
    spans[ro] = (off, off + len(names))
    off += len(names)
rec_idx = torch.as_tensor(np.concatenate(recs), device="cuda")
group_of = torch.as_tensor(np.concatenate(groups), device="cuda")

eyes = {"strips": Eyes(args.board_hz, args.piece_hz),
        "image": EyesImage(args.image_hz or args.board_hz, args.piece_hz)}
print("image map", eyes["image"].info, flush=True)

results = {"config": vars(args), "image_map": eyes["image"].info}
for wiring in args.wirings.split(","):
    torch.cuda.empty_cache()
    t0 = time.time()
    brain = Brain(args.threshold, 1.0, rewire_seed=None if wiring == "real" else 12345,
                  rewire_keep_sensory=wiring == "fair")
    print(f"{wiring}: loaded in {time.time() - t0:.0f}s", flush=True)
    for inp in args.inputs.split(","):
        e = eyes[inp]
        feats = np.zeros((len(boards), off), np.float32)
        rates = []
        for s in range(0, len(boards), 64):
            b, pc = boards[s:s + 64], pieces[s:s + 64]
            p = e.probs(e.channels(b, pc)) if inp == "strips" else e.probs(b, pc)
            counts, rate = brain.run(e.in_idx, p, args.steps, rec_idx, e.phase)
            pooled = torch.zeros((off, counts.shape[1]), device="cuda")
            pooled.index_add_(0, group_of, counts)
            feats[s:s + 64] = pooled.T.cpu().numpy()
            rates.append(float(rate))
        f_sel, f_prac = feats[:200], feats[200:]
        for ro, (lo, hi) in spans.items():
            fs, fp = f_sel[:, lo:hi], f_prac[:, lo:hi]
            chosen = pick_groups(fs, yh_sel, yp_sel, args.k)
            r = {"chosen": decode(fp[tr][:, chosen], fp[te][:, chosen], yh[tr], yh[te], yp[tr], yp[te]),
                 "all": decode(fp[tr], fp[te], yh[tr], yh[te], yp[tr], yp[te]),
                 "groups_firing": int((fp.sum(0) > 0).sum()), "groups": hi - lo,
                 "spiking_per_ms": round(float(np.mean(rates)) / brain.n, 4)}
            results[f"{inp}|{wiring}|{ro}"] = r
            print(f"{inp:6s} {wiring:4s} {ro:18s} {json.dumps(r)}", flush=True)
        print(f"  {inp} done at {time.time() - t0:.0f}s", flush=True)
    del brain

(root / f"runs/phase1c{args.tag}.json").write_text(json.dumps(results, indent=2))
print("wrote", f"runs/phase1c{args.tag}.json")
