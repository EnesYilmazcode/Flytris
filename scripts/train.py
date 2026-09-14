"""Evolve the readout with the noisy cross-entropy method (Szita and Lorincz 2006).

Each generation samples `pop` readouts from a per-weight Gaussian, plays them all on one
new piece sequence, keeps the best `elites`, and refits the Gaussian with extra variance
Z_t = max(5 - t/10, 0) * NOISE_SCALE. Szita and Lorincz started at variance 100 with that
Z_t; our features are z-scored and we start at variance 1, so NOISE_SCALE = 1/100 keeps the
same ratio of added noise to starting spread.

--head afterstate scores every landing of the current piece instead of choosing column and
rotation directly (params [F+1]); see fly_player.play_afterstate.

Resumable: state is checkpointed after every generation. Every generation's moves go to
<out>/gen_XXXX.npz (seed, moves uint8 [steps, flies] with 255 for dead, death, lines, params).

    python -X utf8 scripts/train.py --until 04:30
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flytris.fly_player import GROUPS, GROUPS_AFTER, N_OUT, TRAIN_SEED0, VALID_SEEDS, death_index, play, play_afterstate
from flytris.tetris import Batch

NOISE_SCALE = 1 / 100
SAMPLE_SEED = 7


def save_npz(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader or antivirus holds the file for a moment
            time.sleep(0.5)
    os.replace(tmp, path)


def deadline(hhmm):
    if not hhmm:
        return None
    h, m = map(int, hhmm.split(":"))
    now = datetime.now()
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= now:
        d += timedelta(days=1)
    # a restart just after the deadline would otherwise train until the same time tomorrow
    return now if d - now > timedelta(hours=16) else d


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def fitness(batch):
    return batch.lines.astype(np.int64) * 1000 + batch.placed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", choices=["brain", "nobrain"], default="brain")
    ap.add_argument("--out", default=None)
    ap.add_argument("--head", choices=["columns", "afterstate"], default="columns")
    ap.add_argument("--groups", default=None, help="readout groups file (default depends on --head)")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--gens", type=int, default=10_000)
    ap.add_argument("--until", default="", help="stop before starting a generation after HH:MM")
    ap.add_argument("--validate-every", type=int, default=5)
    ap.add_argument("--max-batch", type=int, default=None, help="brain batch (64 columns, 384 afterstate)")
    args = ap.parse_args(argv)
    after = args.head == "afterstate"
    groups_file = args.groups or str(GROUPS_AFTER if after else GROUPS)
    sub_dir = ("train" if args.player == "brain" else "nobrain") + ("_after" if after else "")
    out = Path(args.out or ROOT / "runs" / sub_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", encoding="utf8")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        log_file.write(line + "\n")
        log_file.flush()
        print(line, flush=True)

    stop_at = deadline(args.until)
    keep_awake()
    if args.player == "brain" and after:
        from flytris.fly_player import AfterstateFlyPlayer
        player = AfterstateFlyPlayer(groups_file=groups_file, max_batch=args.max_batch or 384)
    elif args.player == "brain":
        from flytris.fly_player import FlyPlayer
        player = FlyPlayer(groups_file=groups_file, max_batch=args.max_batch or 64)
    elif after:
        from flytris.fly_player import AfterstateNoBrainPlayer
        player = AfterstateNoBrainPlayer()
    else:
        from flytris.fly_player import NoBrainPlayer
        player = NoBrainPlayer()
    shape = (player.n_features + 1,) if after else (N_OUT, player.n_features + 1)
    play_fn = play_afterstate if after else play

    state_path = out / "state.npz"
    if state_path.exists():
        with np.load(state_path) as s:
            gen, mu, var = int(s["gen"]), s["mu"], s["var"]
            champ, champ_score, champ_gen = s["champion"], float(s["champion_score"]), int(s["champion_gen"])
        log(f"resuming at generation {gen}")
    else:
        gen, mu, var = 0, np.zeros(shape, np.float32), np.ones(shape, np.float32)
        champ, champ_score, champ_gen = mu.copy(), -1.0, -1
        log(f"new run: {vars(args)}")

    while gen < args.gens:
        if stop_at and datetime.now() >= stop_at:
            log(f"reached --until {args.until}, stopping before generation {gen}")
            break
        t0 = time.time()
        rng = np.random.default_rng([SAMPLE_SEED, gen])
        params = (mu + np.sqrt(var) * rng.standard_normal((args.pop, *shape))).astype(np.float32)
        seed = TRAIN_SEED0 + gen
        batch = Batch(args.pop, seed)
        moves = play_fn([batch], [params], player, args.cap)[0]
        fit = fitness(batch)
        elite = np.argsort(-fit, kind="stable")[:args.elites]
        save_npz(out / f"gen_{gen:04d}.npz", seed=seed, moves=moves, death=death_index(batch, args.cap),
                 lines=batch.lines.astype(np.int32), params=params)

        mu = params[elite].mean(0)
        extra = max(5 - gen / 10, 0) * NOISE_SCALE
        if after:
            # ~33 weights and one game per fly collapse the elite spread in tens of generations; std >= 0.1 keeps searching
            extra = max(extra, 0.01)
        var = params[elite].var(0) + extra
        msg = (f"gen {gen}: best {batch.lines.max()} lines / {batch.placed.max()} pieces, "
               f"elite mean {batch.lines[elite].mean():.1f} lines, pop mean {batch.lines.mean():.1f}, "
               f"mean std {np.sqrt(var).mean():.3f}")

        if (gen + 1) % args.validate_every == 0:
            cands = np.concatenate([mu[None], params[elite[:4]]])
            vb = [Batch(len(cands), s) for s in VALID_SEEDS]
            play_fn(vb, [cands] * len(vb), player, args.cap)
            score = np.mean([fitness(b) for b in vb], 0)
            best = int(np.argmax(score))
            if score[best] > champ_score:
                champ, champ_score, champ_gen = cands[best].copy(), float(score[best]), gen
                save_npz(out / "champion.npz", params=champ, validation_fitness=champ_score, gen=champ_gen,
                         validation_lines=np.array([b.lines[best] for b in vb]), groups=groups_file,
                         head=args.head)
            msg += f" | validation best {score[best] / 1000:.1f}, champion {champ_score / 1000:.1f} (gen {champ_gen})"

        gen += 1
        save_npz(state_path, gen=gen, mu=mu, var=var, champion=champ, champion_score=champ_score,
                 champion_gen=champ_gen, config=json.dumps(vars(args)))
        log(f"{msg} | {time.time() - t0:.0f}s")
    log_file.close()


if __name__ == "__main__":
    main()
