# Flytris development history

Date reconstructed: 2026-09-15

This document records not only the accepted result, but also the approaches that failed,
the implementation mistakes discovered along the way, and the evidence used to decide
what to try next.

## 1. Experimental definition

A score counts as a fly-connectome result only when legal Tetris placements are scored
from neural activity produced by the 166,700-neuron MaleCNS connectome simulation.

The final system is not a biological fly and is not end-to-end vision and motor control.
It has three explicit engineered interfaces:

1. Tetris board information is converted into sensory input rates.
2. Every legal placement is presented to the connectome as a continuous transition from
   the landed board to the line-cleared board.
3. A trained linear readout converts selected L1/L2 spike counts into placement scores.

The Tetris engine enumerates legal placements. The connectome-mediated controller chooses
among them. This distinction must remain visible in any public claim.

## 2. V1/V2 evolutionary baseline

The first approach directly evolved small readouts attached to fly activity. It generated
many real connectome-controlled games but learned slowly and unstably.

- `runs/train_after`: 13 generations x 32 candidates = 416 primary games, plus 25
  validation games.
- `runs/train_after_v2`: 105 generations x 64 candidates = 6,720 primary games, plus
  525 validation games.
- V2's one-off high was 13 lines, but its five-seed mean was only 3.8 lines.
- Held-out generation 69 averaged 2.1 lines versus 1.8 for generation 0, which was too
  small to establish meaningful learning.

Lesson: blind evolution was spending most of its compute searching over a weak and highly
compressed representation. More GPU throughput alone would not fix that bottleneck.

## 3. Building a teacher without mislabeling it

A nine-feature linear afterstate policy was trained without the fly connectome to answer a
diagnostic question: was the Tetris environment itself learnable with the available board
features?

The answer was yes. The teacher averaged 113.72 lines, had a median of 115, ranged from 93
to 119, and survived 48 of 50 games to the 300-piece cap.

This briefly caused confusion because a 117/119-line rendering looked like an enormous
fly improvement. It was not a fly controller. From that point onward, all reports and
renderings explicitly separated:

- teacher/no-brain scores;
- fly-connectome scores;
- control or ablation scores.

Lesson: a strong non-neural teacher is useful for distillation and debugging, but must
never be presented as the experimental result.

## 4. Modal benchmark and exact simulation

Modal's L4 was selected after measuring the actual sparse-connectome workload:

- batch size: 384;
- simulated duration: 50 ms;
- median wall time: 1.886 seconds;
- throughput: 10.18 simulated-fly-seconds per wall-second;
- peak memory: 1.65 GiB;
- deterministic repeatability error: exactly zero.

This showed that memory was not the limiting factor and that expensive H100-class GPUs
were unnecessary. The workspace had a $0 cash spend limit and used promotional credits.

## 5. Representation experiments

All representation gates used disjoint parent-board trajectories for training and held-out
testing. The metric was top-move agreement with the passing teacher.

| Representation | Held-out agreement | Decision |
|---|---:|---|
| Original 32 pooled visual features | 19.2% | Reject |
| Individual visual-projection neurons, 6 temporal bins | 23.72% | Reject |
| Raw retinotopic R1-R6 pixel eyes | 14.74% | Reject |
| Individual L1/L2 neurons, 6 temporal bins | 23.72% | Reject |
| Landed-to-cleared transition, L1/L2, 512 signals | 25.0% | Continue only as baseline |
| L1-L5 transition, 512 of 8,884 signals | 25.0% | Reject added complexity |
| L1/L2 transition, 1,024 aggregate signals | 50.64% | Accept |
| L1/L2 transition, 2,048 aggregate signals | 54.49% | Accept |
| L1/L2 transition, 3,072 aggregate signals | 54.49% | Reject tie/complexity |

The critical discovery was that dividing activity into many temporal bins while retaining
only 512 total signals discarded the spatially distributed code. Aggregating the full
150 ms response into one count per neuron allowed 1,024 and then 2,048 distinct neurons
to survive selection. That change, rather than a more expensive GPU, produced the jump.

## 6. Failed improvement paths

### DAgger

Fly-visited states were added to teacher states and several heads were retrained:

- linear DAgger: 26.32% offline agreement, 2.125 mean lines, 5 best;
- ordinary MLP: 29.82% offline, 1.5 mean, 5 best;
- listwise-ranking MLP iteration 1: 36.0% offline, 2.08 mean, 5 best;
- aggregated listwise iteration 2: 38.98% offline, 1.92 mean, 5 best.

Higher offline accuracy did not translate into longer survival. These heads were rejected.

### Runtime feature-centering bug

The listwise MLP was trained on candidate features centered within each parent board, but
the first gameplay implementation passed uncentered features at runtime. The runtime was
fixed to center candidates by parent group. Rerunning still produced weak gameplay, so the
bug explained part of the discrepancy but did not rescue the approach.

### Direct CEM evolution

A 512-signal readout was evolved directly using fly-game fitness. Training-seed performance
rose from 2.33 to 5.0 lines, but fresh 12-seed validation fell to 2.17 mean and 4 best.
This was seed overfitting, not general improvement.

Lesson: gameplay validation on untouched seeds must decide acceptance. Training fitness or
offline agreement alone is insufficient.

## 7. Engineering failures and safeguards

- Representation filenames originally omitted feature count and temporal-bin settings,
  allowing new runs to overwrite earlier artifacts. Tagged artifact names were added and
  important models were immediately downloaded locally.
- Evaluation files also collided by readout stem. A `run_tag` was added.
- The first evaluator saved visited states but not exact moves. Move matrices, seeds, lines,
  and piece counts are now saved in every replay archive.
- Evolution used a fixed checkpoint directory. Named run directories and parameter-shape
  checks were added so a 512-parameter checkpoint cannot corrupt a 1,024/2,048 run.
- Modal preempted some L4 workers. Deterministic seeds allowed safe automatic restarts.
- Long evaluations initially printed only on completion. Progress logging every 25 pieces
  was added.
- The raw-pixel eye mapping performed worse than engineered sensory channels. It remains an
  experimental path, not the accepted result.

## 8. Breakthrough and validation

The 1,024-signal model passed the 50% representation gate and immediately established that
the change mattered in games:

- first 12 games: 22.33 mean, 63 best;
- independent 24 games: 25.42 mean, 70 best.

The 2,048-signal model then improved both representation accuracy and gameplay:

- first 12 games: 49.08 mean, 86 best;
- independent 24 games: 36.46 mean, 73 best;
- combined 36 games: 40.67 mean, median 39, range 9-86;
- 12 of 36 reached at least 50 lines;
- 2 of 36 reached at least 75 lines.

The 86-line seed was replayed deterministically from its saved move sequence and finished
with exactly 257 pieces again.

## 9. Causal controls

On the same first 12 seeds used by the independent validation:

| Condition | Mean | Best |
|---|---:|---:|
| Intact trained connectome pathway | 33.67 | 64 |
| Silenced readout | 0.08 | 1 |
| Shuffled readout weights | 5.42 | 8 |
| Shuffled sensory wiring | 5.25 | 11 |

The collapse shows that the trained neural signals, their identities, and the sensory
routing are causally important. It does not establish biological behavior beyond the
stated connectome simulation and engineered interfaces.

## 10. How many simulated flies?

There are several non-equivalent counts:

- Approximately **8,200 full simulated-fly game episodes** were run across V1, V2, V3,
  validation, and controls. This is the most defensible interpretation of “flies that
  played Tetris.”
- Approximately **7,200 distinct evolved policy candidates** account for most of those
  episodes; some candidates were replayed on multiple seeds.
- Each afterstate game simulates the connectome once for every legal candidate placement,
  not merely once per falling piece. Using recorded game lengths and typical legal-move
  counts gives an estimated **6-8 million candidate-placement connectome simulations**.
- Representation fitting added hundreds of thousands of isolated afterstate simulations,
  but those are not complete flies or complete games.

Therefore “roughly 8,000 simulated fly-connectome agents played Tetris” is a reasonable
rounded description. “750,000 flies” is not supported by the recorded full-game count.

## 11. Final accepted artifacts

- `runs/v3_modal/wide2048/readout.npz`: accepted 2,048-signal readout.
- `runs/v3_modal/wide2048/gate.json`: representation gate.
- `runs/v3_modal/wide2048/replay_seed1800000.npz`: batch containing the 86-line game.
- `runs/v3_modal/wide2048/replay_seed1900000.npz`: independent 24-game batch.
- `media/fly_connectome_86_lines.mp4`: single-game replay.
- `media/fly_connectome_tournament_36.mp4`: authentic 36-game tournament cut.
- `V3-FINAL-REPORT.md`: compact result report and claim-safe wording.

## 12. Remaining scientific limitations

- The sensory encoder is engineered rather than a rendered camera-to-photoreceptor model.
- Placement enumeration bypasses low-level left/right/rotate motor control.
- The readout is trained with supervision from a no-brain teacher.
- The connectome model uses simplified spike dynamics and fixed parameters.
- Thirty-six accepted-model validation games provide a useful estimate, but a larger
  preregistered evaluation would narrow uncertainty.
- A fully rewired-connectome control would complement the existing input/readout shuffles.

These limitations do not erase the result; they define the precise result that was shown.
