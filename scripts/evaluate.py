"""Score the after-state champion on unseen piece sequences, with controls.

Rows: champion fly, best generation-0 fly, champion with the brain silenced, champion with
its scores shuffled among its own landings, the no-brain after-state champion, and random
placement. Also records the champion's games on the first `--record` eval sequences for
the play page. Resumable: finished rows are kept in <out>/results.json.

    python -X utf8 scripts/evaluate.py
    python -X utf8 scripts/evaluate.py --dry-cpu --seeds 3 --cap 20 --record 3 --out runs/eval_dry
"""
import argparse
import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--dry-cpu", action="store_true", help="no-brain stand-in for the brain, no torch or GPU")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--cap", type=int, default=150)
ap.add_argument("--record", type=int, default=50)
ap.add_argument("--out", default=str(ROOT / "runs/eval"))
ap.add_argument("--train-dir", default=str(ROOT / "runs/train_after"))
ap.add_argument("--nobrain-dir", default=str(ROOT / "runs/nobrain_after"))
ap.add_argument("--max-batch", type=int, default=384)
args = ap.parse_args()

if args.dry_cpu:
    sys.modules["torch"] = types.ModuleType("torch")  # eyes.py and fly_player.py import it at top

from flytris.fly_player import DEAD, EVAL_SEED0, AfterstateNoBrainPlayer, play_afterstate  # noqa: E402
from flytris.tetris import PLACEMENTS, Batch  # noqa: E402

SEC_PER_BOARD = 0.014
MEAN_LANDINGS = np.mean([len(p) for p in PLACEMENTS])
ROWS = {
    "champion": "Champion fly",
    "gen0": "Best generation-0 fly",
    "silenced": "Champion, brain silenced",
    "shuffled": "Champion, shuffled scores",
    "nobrain": "No-brain readout (same head)",
    "random": "Random placement",
}

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log_file = open(out / "eval.log", "a", encoding="utf8")


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    log_file.write(line + "\n")
    log_file.flush()
    print(line, flush=True)


def replace_retry(tmp, path):
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    os.replace(tmp, path)


def save_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf8")
    replace_retry(tmp, path)


def save_npz(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    replace_retry(tmp, path)


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def best_of_generation(path):
    g = load(path)
    pieces = (g["moves"] != DEAD).sum(0)
    i = int(np.lexsort((-pieces, -g["lines"]))[0])
    return g["params"][i], int(g["lines"][i]), int(pieces[i])


def summarize(lines, pieces):
    lines, pieces = np.asarray(lines, float), np.asarray(pieces, float)
    boot = np.random.default_rng(0).choice(lines, (2000, len(lines))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"games": len(lines), "mean_lines": float(lines.mean()), "ci95_lines": [float(lo), float(hi)],
            "median_pieces": float(np.median(pieces)), "lines": lines.tolist(), "pieces": pieces.tolist()}


def pad_moves(logs):
    steps = max(len(m) for m in logs)
    moves = np.full((steps, len(logs)), DEAD, np.uint8)
    for j, m in enumerate(logs):
        moves[:len(m), j] = m
    return moves


keep_awake()
train_dir, nobrain_dir = Path(args.train_dir), Path(args.nobrain_dir)
source_dir = nobrain_dir if args.dry_cpu else train_dir
if not (source_dir / "champion.npz").exists():
    log(f"no champion at {source_dir / 'champion.npz'}; nothing to evaluate")
    sys.exit(1)
champ_file = load(source_dir / "champion.npz")
champ = champ_file["params"]
gen0, gen0_lines, gen0_pieces = best_of_generation(source_dir / "gen_0000.npz")
nobrain_champ = load(nobrain_dir / "champion.npz")["params"] if (nobrain_dir / "champion.npz").exists() else None

results_path = out / "results.json"
results = json.loads(results_path.read_text(encoding="utf8")) if results_path.exists() else {}
rows = results.setdefault("rows", {})
results["meta"] = {"dry_cpu": args.dry_cpu, "champion_gen": int(champ_file["gen"]), "cap": args.cap,
                   "eval_seeds": [EVAL_SEED0, EVAL_SEED0 + args.seeds - 1], "source": str(source_dir)}

seeds = [EVAL_SEED0 + i for i in range(args.seeds)]
record = args.record
champ_pieces_est = min(args.cap, 2.5 * float(np.mean(champ_file["validation_lines"])) + 25)


def estimate_minutes(n_record):
    boards = MEAN_LANDINGS * (args.seeds * (champ_pieces_est + 2 * gen0_pieces)
                              + max(0, n_record - args.seeds) * champ_pieces_est)
    return boards * SEC_PER_BOARD / 60


est = estimate_minutes(record)
log(f"start: champion gen {int(champ_file['gen'])}, est. champion {champ_pieces_est:.0f} pieces/game, "
    f"gen-0 best {gen0_pieces} pieces; brain rows ~{est:.0f} min "
    f"({MEAN_LANDINGS:.1f} landings x {SEC_PER_BOARD * 1000:.0f} ms/board)")
if est > 90 and record > 30:
    record = 30
    log(f"estimate over 90 min: recording {record} champion games instead of {args.record}")

if args.dry_cpu:
    player = AfterstateNoBrainPlayer()
    log("DRY RUN: no-brain stand-in for the brain player")
else:
    from flytris.fly_player import AfterstateFlyPlayer
    player = AfterstateFlyPlayer(groups_file=str(champ_file["groups"]), max_batch=args.max_batch)


def save_results():
    save_text(results_path, json.dumps(results, indent=1))
    lines = ["| Player | Games | Mean lines | 95% CI | Median pieces |", "|---|---|---|---|---|"]
    for key, name in ROWS.items():
        r = rows.get(key)
        if r:
            lines.append(f"| {name} | {r['games']} | {r['mean_lines']:.1f} | "
                         f"{r['ci95_lines'][0]:.1f} to {r['ci95_lines'][1]:.1f} | {r['median_pieces']:.0f} |")
    note = "\n\nDRY RUN: no-brain stand-in, not the fly.\n" if args.dry_cpu else "\n"
    save_text(out / "results.md", "\n".join(lines) + note)


part1 = out / "champion_games_part1.npz"
if not ("champion" in rows and "gen0" in rows and part1.exists()):
    t0 = time.time()
    batches = [Batch(2, s) for s in seeds]
    logs = play_afterstate(batches, [np.stack([champ, gen0])] * len(batches), player, args.cap)
    rows["champion"] = summarize([b.lines[0] for b in batches], [b.placed[0] for b in batches])
    rows["gen0"] = summarize([b.lines[1] for b in batches], [b.placed[1] for b in batches])
    keep = min(record, len(seeds))
    save_npz(part1, seeds=np.array(seeds[:keep]), moves=pad_moves([m[:, 0] for m in logs[:keep]]),
             lines=np.array([b.lines[0] for b in batches[:keep]]),
             pieces=np.array([b.placed[0] for b in batches[:keep]]))
    save_results()
    log(f"champion {rows['champion']['mean_lines']:.1f} lines, gen-0 best {rows['gen0']['mean_lines']:.1f} "
        f"| {time.time() - t0:.0f}s")

for key, mode in (("silenced", "silenced"), ("nobrain", None), ("random", None), ("shuffled", "shuffled")):
    if key in rows:
        continue
    t0 = time.time()
    batches = [Batch(1, s) for s in seeds]
    if key == "nobrain":
        if nobrain_champ is None:
            log("no no-brain champion found, skipping that row")
            continue
        play_afterstate(batches, [nobrain_champ[None]] * len(batches), AfterstateNoBrainPlayer(), args.cap)
    elif key == "random":
        rng = np.random.default_rng(1)
        for b in batches:
            while b.alive[0] and b.placed[0] < args.cap:
                b.step([int(rng.integers(b.options))])
    else:
        play_afterstate(batches, [champ[None]] * len(batches), player, args.cap, mode=mode, seed=1)
    rows[key] = summarize([b.lines[0] for b in batches], [b.placed[0] for b in batches])
    save_results()
    log(f"{ROWS[key]}: {rows[key]['mean_lines']:.1f} lines | {time.time() - t0:.0f}s")

part2 = out / "champion_games_part2.npz"
extra = [EVAL_SEED0 + i for i in range(len(seeds), record)]
if extra and not part2.exists():
    t0 = time.time()
    batches = [Batch(1, s) for s in extra]
    logs = play_afterstate(batches, [champ[None]] * len(batches), player, args.cap)
    save_npz(part2, seeds=np.array(extra), moves=pad_moves([m[:, 0] for m in logs]),
             lines=np.array([b.lines[0] for b in batches]), pieces=np.array([b.placed[0] for b in batches]))
    log(f"recorded {len(extra)} more champion games | {time.time() - t0:.0f}s")

parts = [load(part1)] + ([load(part2)] if extra and part2.exists() else [])
moves = pad_moves([p["moves"][:, j] for p in parts for j in range(p["moves"].shape[1])])
save_npz(out / "champion_games.npz", seeds=np.concatenate([p["seeds"] for p in parts]), moves=moves,
         lines=np.concatenate([p["lines"] for p in parts]), pieces=np.concatenate([p["pieces"] for p in parts]))
log(f"done: {moves.shape[1]} champion games recorded, table in {out / 'results.md'}")
log_file.close()
