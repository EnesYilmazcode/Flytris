# Colab TPU probe

This experiment answers one question before Flytris is ported: can a free
Colab TPU execute the real 166,700-neuron, threshold-5 MaleCNS recurrent step
quickly enough to justify an XLA backend?

## What is and is not tested

`scripts/tpu_probe.py` preserves the real edge weights and recurrent LIF update,
but replaces CUDA's CSR `torch.sparse.mm` with an XLA-friendly edge gather and
`index_add`. It checks the replacement against dense multiplication on a small
graph, then reports TPU compile time and cached execution time on the real graph.

It does not train Tetris readouts yet. Porting the eyes and full game loop only
makes sense if this probe is both correct and fast.

## Colab steps

1. Open a new Colab notebook and select **Runtime > Change runtime type > TPU**.
2. Upload the local `Flytris-colab-probe.zip` bundle prepared beside this repo.
3. Run these notebook cells:

```python
!unzip -q Flytris-colab-probe.zip -d /content/Flytris
%cd /content/Flytris
```

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
!python scripts/tpu_probe.py --device tpu --batch 8 --steps 10
```

4. If batch 8 succeeds, measure the useful scaling range:

```python
!python scripts/tpu_probe.py --device tpu --batch 32 --steps 10
!python scripts/tpu_probe.py --device tpu --batch 64 --steps 10
```

Stop if the real-graph run is unsupported, exhausts memory, or is slower than
the laptop CUDA baseline. A successful result must print both `scatter parity:
PASS` and `repeatability: PASS`.

Free Colab runtimes can disappear. Once full training is ported, every generation
must checkpoint to Google Drive; the TPU probe itself writes nothing.
