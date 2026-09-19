# Flytris on Modal

The Modal workspace is `enesyilmaz5157`. Its out-of-pocket spend limit is set to `$0`,
and the Starter workspace shows `$30` in credits.

## Cloud resources

- `flytris-data`: converted MaleCNS `edges.npz`, `neurons.parquet`, and `eye_map.npz`.
- `flytris-runs`: readout selection, practice boards, V2 initialization, warm start,
  generation-104 checkpoint, champion, and future benchmark results.

The raw 1.1 GB MaleCNS download and obsolete FlyWire fallback were deliberately not
uploaded. The converted files are sufficient to run the current simulator.

## Benchmark

No GPU starts when the app is imported or built. Run one explicitly:

```powershell
modal run scripts/modal_flytris.py --gpu L4 --batch 384 --repeats 3
```

Valid GPU values are `L4`, `A10`, `L40S`, `A100`, and `H100`. Every invocation is limited
to one hour, retries once, checks deterministic output, and saves a JSON result under
`benchmarks/` in `flytris-runs`.

List saved results without starting compute:

```powershell
python -X utf8 -m modal volume ls flytris-runs /benchmarks
```

Do not run long V2 training on Modal. V3 uses the L4 because the exact-connectome
benchmark showed ample memory and good throughput for this workload.

## V3 status and gates

The learned no-brain policy passed the final untouched-seed gate. Its source model and
full results are in `runs/v3_nobrain/` and are mirrored to `flytris-runs/v3/`.

Reproduce that gate locally:

```powershell
python -X utf8 scripts/train_v3_policy.py
```

The original 32 pooled fly-brain features failed the representation gate, with only
19.2% held-out top-move agreement against the passing V3 teacher. The successful model
instead reads 2,048 distinct L1/L2 neural signals from the full connectome simulation.
Reproduce the old local baseline with:

```powershell
python -X utf8 scripts/distill_v3_brain.py
```

`scripts/evolve_v3_policy.py` is the resumable CPU reference implementation for the new
training loop. It uses one primary seed, promotes only a shortlist to four additional
seeds, checkpoints every generation, and has no automatic shutdown time. It should be
used to validate future representation changes before porting the same loop to Modal.

The representation gate is now passed. Long evolution is optional rather than required;
the unevolved 2,048-signal model already reached the requested 75 to 100-line range.

### September 15 V3 neural experiments

- L4 benchmark: 384 brains x 50 ms in 1.886 seconds, deterministic, 1.65 GiB peak VRAM.
- Individual visual-projection neurons, 150 ms/6 bins: 23.7% held-out agreement.
- Retinotopic raw-pixel eyes: 14.7%; rejected.
- Individual L1/L2 neurons, 150 ms/6 bins: 23.7%.
- Continuous landed-board to cleared-board L1/L2 stimulus: 25.0%.
- A 1,024-signal aggregate L1/L2 representation reached 50.64% held-out agreement.
- Its first 12 untouched games averaged 22.33 lines and set a genuine 63-line record.
- An independent 24-game set averaged 25.42 lines and reached 70.
- A 2,048-signal representation reached 54.49% held-out agreement.
- Its first 12 untouched games averaged 49.08 lines and reached **86 lines**.
- Paired controls collapsed: silenced readout 0.08 mean/1 best, shuffled readout
  5.42/8, and shuffled sensory wiring 5.25/11.

The honest description is a 166,700-neuron fruit-fly connectome simulation with
engineered Tetris sensory encoding and a trained linear readout from L1/L2 activity.
Only connectome-mediated games count as fly scores. The next stage is a larger untouched
validation, followed by optional isolated evolution if it beats the unevolved baseline.
