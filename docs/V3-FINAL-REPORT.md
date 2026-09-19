# Flytris V3 result report

Date: 2026-09-15

## Headline result

The best verified connectome-mediated game cleared **86 lines in 257 pieces**.
Across 36 untouched games, the accepted 2,048-signal controller cleared **40.67 lines
on average** with a median of 39 and a range of 9 to 86. Twelve of 36 games reached at
least 50 lines, and two reached at least 75.

The exact 86-line move sequence was saved, replayed deterministically, and rendered.

## What the controller is

- A simulation of the 166,700-neuron MaleCNS fruit-fly connectome runs for every legal
  candidate placement.
- Engineered Tetris board features are injected through a fixed sensory interface.
- Each candidate is shown as one continuous landed-board to cleared-board transition.
- A regularized linear readout scores 2,048 distinct L1/L2 neural spike-count signals.
- The readout was trained to imitate a strong no-brain Tetris teacher on training boards.
- Only scores produced from the simulated neural activity count as fly results.

This is accurately described as a **fruit-fly-connectome-mediated Tetris controller**.
It is not a biological fly, an embodied vision system, or a controller that learned
motor actions from pixels end to end.

## Results

| Controller | Games | Mean lines | Best lines |
|---|---:|---:|---:|
| 1,024-signal fly, first untouched set | 12 | 22.33 | 63 |
| 1,024-signal fly, independent set | 24 | 25.42 | 70 |
| 2,048-signal fly, first untouched set | 12 | 49.08 | 86 |
| 2,048-signal fly, independent set | 24 | 36.46 | 73 |
| 2,048-signal fly, combined | 36 | **40.67** | **86** |

The 2,048-signal representation achieved 54.49% held-out top-move agreement with the
teacher, compared with 50.64% for 1,024 signals and 19.2% for the original 32 pooled
features.

## Causal controls

Controls used the same first 12 seeds as the independent intact validation.

| Condition | Mean lines | Best lines |
|---|---:|---:|
| Intact trained pathway | 33.67 on the paired seeds | 64 |
| Silenced readout | 0.08 | 1 |
| Shuffled readout weights | 5.42 | 8 |
| Shuffled sensory wiring | 5.25 | 11 |

The large collapse under all three controls shows that the trained neural signals and
their wiring are causally important. It does not prove biological plausibility beyond
the connectome simulation and stated interface.

## Reproducibility artifacts

- `runs/v3_modal/wide2048/readout.npz`: accepted neural readout and normalization.
- `runs/v3_modal/wide2048/gate.json`: held-out representation test.
- `runs/v3_modal/wide2048/eval_seed1800000.json`: first 12-game result.
- `runs/v3_modal/wide2048/eval_seed1900000.json`: independent 24-game result.
- `runs/v3_modal/wide2048/replay_seed1800000.npz`: exact 86-line replay batch.
- `media/fly_connectome_86_lines.mp4`: rendered champion replay.
- `media/fly_connectome_tournament_36.mp4`: 36-game tournament cut with the exact
  independent replays, elimination flashes, and final zoom to the 86-line winner.
- `media/fly_connectome_arcade_cinematic.mp4`: text-free cinematic cut with the 3D fly
  arcade intro, falling/ghost pieces, a 2,304-board repeated-replay wall, eliminations,
  and the authentic winner.
- `V3-DEVELOPMENT-HISTORY.md`: chronological record of failed approaches, bugs, gates,
  experiment counts, and why the accepted representation worked.

## Claim-safe post wording

> I wired Tetris into a simulation of a 166,700-neuron fruit-fly connectome and trained
> a readout from its neural activity. It averaged 40.7 lines over 36 unseen games and
> peaked at 86. Silencing or shuffling the neural pathway collapsed performance.

Do not claim that 750,000 flies played unless that many separately seeded games are
actually run. Candidate afterstate simulations are not separate flies.

## Remaining work

- The isolated 3,072-signal comparison tied at 54.49% held-out agreement and was
  rejected because it added complexity without improvement.
- Optionally test a randomized-connectome control in addition to the existing pathway
  controls.
- Record exact Modal billing from the dashboard. Tracked L4 runtime so far implies a
  compute cost far below the $30 credit balance, but dashboard billing is authoritative.
