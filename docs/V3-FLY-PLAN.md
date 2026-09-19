# Flytris V3: fly-brain training plan

Only games whose placement scores are computed from simulated connectome spikes count
as fly results. The 113.72-line learned Tetris controller is a teacher/control, not a
fly result. The verified fly-connectome record is now **86 lines**.

## Milestones

1. **Observable transition stimulus**
   - Show the landed board and completed rows, then the cleared board.
   - Preserve membrane, synaptic, refractory, delay-buffer, and input-phase state.
   - Gate: improve held-out teacher agreement over the 23.7% afterstate baseline.

2. **On-policy dataset (DAgger)**
   - Start with teacher trajectories, run the current fly, label the states it actually
     visits, retrain, and repeat.
   - Keep training and held-out piece-sequence seeds disjoint.
   - Gate: agreement improves on both teacher states and fly-visited states for two
     consecutive iterations.

3. **Readout comparison**
   - Compare L1/L2, L1-L5, visual-projection, central-brain, and descending-neuron
     activity using the same parent boards and split.
   - Compare regularized linear ranking with a small nonlinear ranking head.
   - Gameplay inputs remain neural spike counts only.
   - Gate: at least 50% held-out top-move agreement or a verified fly mean above the
     current V2 mean on 20 untouched games.

4. **Fly gameplay ladder**
   - Evaluate every accepted representation on untouched 300-piece games.
   - Gates: beat 13-line high, then 25, then 50, then 75.
   - No teacher/no-brain score is reported as a fly score.

5. **Modal evolution**
   - One primary seed for every candidate; four additional seeds for the shortlist.
   - Dense trajectory shaping only as a tie-breaker to lines and survival.
   - Retain the champion and checkpoint every completed generation.
   - No scheduled shutdown; manual stop and resume.
   - Do not start a long run before the representation/gameplay gate passes.

6. **Final audit and rendering**
   - Fifty untouched games at a 300-piece cap.
   - Silenced-input, shuffled-feature, rewired-connectome, teacher, and no-brain controls.
   - Replay the exact recorded moves of the verified fly champion.

## September 15 checkpoint

- Representation gate passed at 54.49% held-out top-move agreement.
- Genuine 12-game result: 49.08 mean, 86 best, 257 pieces in the champion game.
- The 86-line replay was reproduced and rendered from its recorded moves.
- Silenced and shuffle controls all failed strongly, supporting dependence on the
  trained connectome-mediated pathway.
- The larger untouched validation finished at 36.46 mean and 73 best across 24 games;
  combined validation is 40.67 mean and 86 best across 36 games.
- The 3,072-signal challenger tied the 2,048-signal model at 54.49% agreement and was
  rejected for adding complexity without improvement.

## Reports

- `sensory-transition-report.json`: stimulus timing, determinism, and decoding.
- `representation-report.json`: neural populations, feature count, held-out agreement.
- `dagger-report.json`: dataset iterations and teacher/fly-state performance.
- `fly-gameplay-report.json`: exact seeds, lines, pieces, deaths, and champions.
- `evolution-report.json`: generations, candidates, seed racing, checkpoints.
- `controls-report.json`: silenced, shuffled, rewired, and non-brain comparisons.
- `modal-cost-report.json`: GPU type, wall time, measured throughput, and estimated cost.
- `final-report.md`: claim-safe result, limitations, actual fly count, and replay links.

## Current score outlook

- Achieved target-range high score: **86 lines**.
- Current 12-game mean: **49.08 lines**.
- 100+ lines is now plausible, but should not be claimed until observed on an untouched
  seed and reproduced from a recorded replay.
- 119 lines remains the no-brain teacher's near-cap result, not a fly score.
