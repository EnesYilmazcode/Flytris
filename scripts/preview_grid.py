"""Pipeline preview: 1000 placeholder players (not flies) rendered as a grid."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flytris import placeholder
from flytris.render import render
from flytris.tetris import Batch

N, SEED, MAX_STEPS = 1000, 7, 900
root = Path(__file__).resolve().parents[1]
(root / "media").mkdir(exist_ok=True)
(root / "runs").mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)
temperature = np.exp(rng.uniform(np.log(0.15), np.log(40), N))

t0 = time.time()
batch = Batch(N, SEED)
choices = []
while batch.alive.any() and len(choices) < MAX_STEPS:
    c = placeholder.choose(batch, temperature, rng)
    choices.append(c)
    batch.step(c)
choices = np.array(choices, np.int8)
print(f"played {len(choices)} pieces in {time.time() - t0:.1f}s, "
      f"alive at end {batch.alive.sum()}, best {batch.lines.max()} lines")
np.savez_compressed(root / "runs/preview_replay.npz", seed=SEED, choices=choices)

t0 = time.time()
winner = render(SEED, choices, str(root / "media/preview.mp4"),
                "1,000 fly brains play Tetris", "PREVIEW: placeholder players, not flies yet",
                still_png=str(root / "media/preview.png"))
print(f"rendered in {time.time() - t0:.1f}s, winner #{winner + 1}")
