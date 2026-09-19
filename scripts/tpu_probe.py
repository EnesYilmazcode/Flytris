"""Benchmark an XLA-friendly MaleCNS synapse step on a TPU.

Flytris' CUDA backend uses ``torch.sparse.mm`` with a CSR matrix. XLA sparse
layouts are not a drop-in replacement, so this probe expresses the same
``W @ spikes`` operation as an edge gather followed by ``index_add``. It is a
benchmark and compatibility test, not yet a training backend.

Local correctness smoke test::

    python scripts/tpu_probe.py --device cpu --synthetic --steps 3

Colab TPU test (after selecting a TPU runtime)::

    PJRT_DEVICE=TPU python scripts/tpu_probe.py --device tpu --batch 8 --steps 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "malecns"
V0, V_RST, V_TH = -52.0, -52.0, -45.0
K_V = 1 - np.exp(-1.0 / 20.0)
A_G = float(np.exp(-1.0 / 5.0))
REFRACTORY_STEPS = 2
W_SYN = 0.275


def scatter_mm(post, pre, weight, spikes, n):
    """Return W @ spikes for COO-style (post, pre, weight) edges."""
    messages = weight[:, None] * spikes.index_select(0, pre)
    return torch.zeros((n, spikes.shape[1]), dtype=spikes.dtype,
                       device=spikes.device).index_add(0, post, messages)


def synthetic_connectivity():
    rng = np.random.default_rng(3)
    n, nnz = 257, 2048
    pre = rng.integers(0, n, nnz, dtype=np.int64)
    post = rng.integers(0, n, nnz, dtype=np.int64)
    weight = rng.normal(size=nnz).astype(np.float32)
    return n, pre, post, weight


def malecns_connectivity(threshold):
    neurons_file, edges_file = DATA / "neurons.parquet", DATA / "edges.npz"
    if not neurons_file.exists() or not edges_file.exists():
        raise FileNotFoundError(
            "MaleCNS converted data is missing. See data/README.md or upload "
            "neurons.parquet and edges.npz into data/malecns/."
        )
    neurons = pd.read_parquet(neurons_file, columns=["type"])
    with np.load(edges_file) as z:
        keep = np.flatnonzero(z["syn"] >= threshold)
        pre = z["pre"][keep].astype(np.int64)
        post = z["post"][keep].astype(np.int64)
        syn = z["syn"][keep].astype(np.float32)
        sign = z["sign"][keep].astype(np.float32)
    # Match Brain.__init__: dark falling blocks make photoreceptor output
    # excitatory in this model.
    is_r = neurons.type.str.match(r"^R[1-8]", na=False).to_numpy()
    sign[is_r[pre]] = 1.0
    return len(neurons), pre, post, syn * sign * W_SYN


def select_device(name):
    if name == "cpu":
        return torch.device("cpu"), None
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA mode requested, but no CUDA GPU is available.")
        return torch.device("cuda"), None
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError(
            "TPU mode needs torch_xla. Select a TPU runtime in Colab first."
        ) from exc
    return xm.xla_device(), torch_xla


def materialize(value, torch_xla):
    if torch_xla is not None:
        # Current torch_xla exposes sync(); older Colab images materialize on
        # the host transfer below, so this remains version-tolerant.
        sync = getattr(torch_xla, "sync", None)
        if sync is not None:
            sync(wait=True)
    return value.detach().cpu()


def simulate(post, pre, weight, n, batch, steps, device, torch_xla):
    # Fixed shapes are intentional: they let XLA reuse the compiled graph.
    seed = torch.arange(n * batch, dtype=torch.int64).reshape(n, batch)
    delayed = ((seed % 97) == 0).to(torch.float32).to(device)
    buf = [delayed, torch.zeros_like(delayed)]
    v = torch.full((n, batch), V0, device=device)
    g = torch.zeros_like(v)
    rfc = torch.zeros((n, batch), dtype=torch.int8, device=device)
    total = torch.zeros((), device=device)
    for step in range(steps):
        g = g * A_G + scatter_mm(post, pre, weight, buf[step % 2], n)
        v = v * (1 - K_V) + g * K_V + V0 * K_V
        v = torch.where(rfc > 0, V_RST, v)
        spk = v > V_TH
        v = torch.where(spk, V_RST, v)
        g = torch.where(spk, 0.0, g)
        rfc = torch.where(spk, REFRACTORY_STEPS,
                          (rfc - 1).clamp(min=0)).to(torch.int8)
        buf[step % 2] = spk.to(torch.float32)
        total = total + spk.sum()
    return total


def check_scatter_math():
    n, pre_np, post_np, weight_np = synthetic_connectivity()
    batch = 5
    rng = np.random.default_rng(9)
    spikes = torch.from_numpy(rng.normal(size=(n, batch)).astype(np.float32))
    pre = torch.from_numpy(pre_np)
    post = torch.from_numpy(post_np)
    weight = torch.from_numpy(weight_np)
    got = scatter_mm(post, pre, weight, spikes, n)
    dense = torch.zeros((n, n), dtype=torch.float32)
    dense.index_put_((post, pre), weight, accumulate=True)
    torch.testing.assert_close(got, dense @ spikes, rtol=2e-5, atol=2e-5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda", "tpu"], default="tpu")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)

    check_scatter_math()
    print("scatter parity: PASS", flush=True)
    if args.synthetic:
        n, pre_np, post_np, weight_np = synthetic_connectivity()
    else:
        n, pre_np, post_np, weight_np = malecns_connectivity(args.threshold)

    device, torch_xla = select_device(args.device)
    pre = torch.from_numpy(pre_np).to(device)
    post = torch.from_numpy(post_np).to(device)
    weight = torch.from_numpy(weight_np).to(device)
    print(f"device={device} neurons={n:,} edges={len(weight_np):,} "
          f"batch={args.batch} steps={args.steps}", flush=True)

    times = []
    totals = []
    for label in ("compile/warmup", "cached run"):
        started = time.perf_counter()
        result = simulate(post, pre, weight, n, args.batch, args.steps,
                          device, torch_xla)
        totals.append(float(materialize(result, torch_xla)))
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"{label}: {elapsed:.2f}s, spikes={totals[-1]:.0f}", flush=True)

    if totals[0] != totals[1]:
        raise RuntimeError(f"repeatability failed: {totals}")
    print(f"repeatability: PASS | cached {args.steps / times[1]:.2f} brain-ms/s",
          flush=True)


if __name__ == "__main__":
    main()
