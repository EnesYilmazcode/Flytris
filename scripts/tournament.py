"""Replay a 1,000-fly evolution field on one unseen sequence.

Individuals are sampled from saved generations, with later generations weighted
more heavily. The brain is evaluated in GPU-sized chunks; the resulting move
log is then rendered as the grid video.

    python -X utf8 scripts/tournament.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flytris.fly_player import AfterstateFlyPlayer, play_afterstate  # noqa: E402
from flytris.render import render  # noqa: E402
from flytris.tetris import Batch  # noqa: E402


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def select_field(files, n, seed):
    rng = np.random.default_rng(seed)
    records = []
    for path in files:
        data = load_npz(path)
        if "params" not in data:
            continue
        gen = int(path.stem.split("_")[-1])
        for i in range(len(data["params"])):
            records.append((gen, i, data["params"][i]))
    if not records:
        raise RuntimeError("no generation checkpoints with params found")
    # Later generations are more likely, but every saved generation remains
    # eligible so the field visibly contains the evolution trajectory.
    gens = np.array([r[0] for r in records], float)
    weights = 1.0 + 3.0 * (gens - gens.min()) / max(1.0, gens.max() - gens.min())
    replace = n > len(records)
    chosen = rng.choice(len(records), size=n, replace=replace, p=weights / weights.sum())
    return np.stack([records[i][2] for i in chosen]), np.array(
        [[records[i][0], records[i][1]] for i in chosen], dtype=np.int32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(ROOT / "runs/train_after"))
    ap.add_argument("--out", default=str(ROOT / "runs/tournament"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=400000)
    ap.add_argument("--sample-seed", type=int, default=400001)
    ap.add_argument("--max-batch", type=int, default=384)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.train_dir).glob("gen_*.npz"))
    params, source = select_field(files, args.n, args.sample_seed)
    moves = np.full((args.cap, args.n), 255, np.uint8)
    lines = np.zeros(args.n, np.int32)
    pieces = np.zeros(args.n, np.int32)

    player = AfterstateFlyPlayer(max_batch=args.max_batch)
    for start in range(0, args.n, args.chunk):
        end = min(args.n, start + args.chunk)
        batch = Batch(end - start, args.seed)
        log = play_afterstate([batch], [params[start:end]], player, args.cap)[0]
        moves[:len(log), start:end] = log
        lines[start:end] = batch.lines
        pieces[start:end] = batch.placed
        print(f"replayed {end}/{args.n} flies", flush=True)

    winner = int(np.lexsort((-pieces, -lines))[0])
    np.savez_compressed(out / "field.npz", seed=args.seed, moves=moves,
                        lines=lines, pieces=pieces, source=source,
                        winner=np.array(winner), params=params)
    meta = {"n": args.n, "cap": args.cap, "seed": args.seed,
            "winner": winner, "winner_lines": int(lines[winner]),
            "winner_pieces": int(pieces[winner])}
    (out / "results.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta), flush=True)

    if not args.no_render:
        render(args.seed, moves, str(out / "hero.mp4"),
               "1,000 flies. One sequence.",
               "Real evolution individuals replayed on an unseen sequence.")


if __name__ == "__main__":
    main()
