"""Pipeline preview: 1000 placeholder players (not flies) rendered as the tournament video."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flytris import placeholder
from flytris.render import render
from flytris.tetris import Batch

N, SEED, MAX_STEPS = 1000, 7, 300
root = Path(__file__).resolve().parents[1]
replay = root / "runs/preview_replay.npz"
(root / "media").mkdir(exist_ok=True)
(root / "runs").mkdir(exist_ok=True)

if replay.exists():
    choices = np.load(replay)["choices"][:MAX_STEPS]
else:
    rng = np.random.default_rng(SEED)
    temperature = np.exp(rng.uniform(np.log(0.15), np.log(40), N))
    batch, choices = Batch(N, SEED), []
    while batch.alive.any() and len(choices) < MAX_STEPS:
        c = placeholder.choose(batch, temperature, rng)
        choices.append(c)
        batch.step(c)
    choices = np.array(choices, np.int8)
    np.savez_compressed(replay, seed=SEED, choices=choices)

t0 = time.time()
media = root / "media"
winner = render(SEED, choices, str(media / "preview.mp4"),
                f"{N:,} fly brains playing Tetris", "PREVIEW: stand-in players, not flies yet",
                stills={k: str(media / f"preview_{k}.png") for k in ("close", "pull", "grid", "final")})
print(f"rendered in {time.time() - t0:.1f}s, winner #{winner + 1}")
