"""MaleCNS leaky integrate-and-fire model, batched over flies on the GPU.

All flies share the real wiring; only their inputs differ. Inputs are regular
spike trains (a phase accumulator per input cell), so a run is deterministic.
Constants are from
Shiu et al. 2024 (philshiu/Drosophila_brain_model, model.py), adapted to a
1 ms Euler step: decays become exp(-dt/tau), the 1.8 ms synaptic delay rounds
to 2 steps and the 2.2 ms refractory period to 2 steps.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parents[1] / "data" / "malecns"

V0, V_RST, V_TH = -52.0, -52.0, -45.0   # mV
T_MBR, TAU_SYN = 20.0, 5.0              # ms
W_SYN, F_POI = 0.275, 250               # mV per synapse; input kick = F_POI * W_SYN
DELAY_STEPS, REFRACTORY_STEPS = 2, 2
DT = 1.0                                # ms


def load_neurons(columns=None):
    return pd.read_parquet(DATA / "neurons.parquet", columns=columns)


def load_edges(threshold):
    z = np.load(DATA / "edges.npz")
    syn = z["syn"]
    keep = np.nonzero(syn >= threshold)[0]
    syn = syn[keep]
    out = {"syn": syn}
    for k in ("pre", "post", "sign"):
        a = z[k]
        out[k] = a[keep]
        del a
    return out


class Brain:
    def __init__(self, threshold=5, w_scale=1.0, excitatory_photoreceptors=True,
                 rewire_seed=None, rewire_keep_sensory=False, device="cuda"):
        self.device = device
        neurons = load_neurons(["idx", "type", "superclass"])
        self.n = len(neurons)
        e = load_edges(threshold)
        if rewire_seed is not None:
            # Wiring control: same neurons, in/out degrees and weights, random targets.
            rng = np.random.default_rng(rewire_seed)
            if rewire_keep_sensory:
                # Fair control: photoreceptor outputs stay real, everything downstream is shuffled.
                is_r = neurons.type.str.match(r"^R[1-8]", na=False).to_numpy()
                m = np.nonzero(~is_r[e["pre"]])[0]
                e["post"][m] = rng.permutation(e["post"][m])
            else:
                e["post"] = rng.permutation(e["post"])
        sign = e["sign"].astype(np.float32)
        if excitatory_photoreceptors:
            # Histamine from R1-R8 is inhibitory, so from rest it can never drive the lamina.
            # Flipping it models the dark falling blocks, which depolarize lamina cells.
            is_r = neurons.type.str.match(r"^R[1-8]", na=False).to_numpy()
            sign[is_r[e["pre"]]] = 1.0
        w = e["syn"].astype(np.float32) * sign * (W_SYN * w_scale)
        order = np.lexsort((e["pre"], e["post"]))
        post, pre, w = e["post"][order], e["pre"][order], w[order]
        del e, sign, order
        crow = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(post, minlength=self.n), out=crow[1:])
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(crow), torch.from_numpy(pre.astype(np.int64)),
            torch.from_numpy(w), (self.n, self.n)).to(device)
        self.nnz = len(w)
        del post, pre, w
        self.k_v = 1 - np.exp(-DT / T_MBR)
        self.a_g = float(np.exp(-DT / TAU_SYN))

    @torch.no_grad()
    def run(self, in_idx, p_in, steps, record_idx, phase, silent_inputs=False):
        """Simulate `steps` ms from rest.

        in_idx [n_in] input neuron ids; p_in [n_in, B] input spikes per step (0..1);
        phase [n_in] fixed starting phase of each input cell's spike train.
        Returns (spike counts [n_rec, B], mean spikes per step across the batch).
        """
        B = p_in.shape[1]
        dev = self.device
        v = torch.full((self.n, B), V0, device=dev)
        g = torch.zeros((self.n, B), device=dev)
        rfc = torch.zeros((self.n, B), dtype=torch.int8, device=dev)
        buf = [torch.zeros((self.n, B), device=dev) for _ in range(DELAY_STEPS)]
        counts = torch.zeros((len(record_idx), B), device=dev)
        total = torch.zeros((), device=dev)
        kick = F_POI * W_SYN
        acc = phase[:, None].expand(-1, B).clone()
        for t in range(steps):
            g.mul_(self.a_g).add_(torch.sparse.mm(self.W, buf[t % DELAY_STEPS]))
            v.mul_(1 - self.k_v).add_(g, alpha=self.k_v).add_(V0 * self.k_v)
            v.masked_fill_(rfc > 0, V_RST)
            if not silent_inputs:
                acc.add_(p_in)
                fire = (acc >= 1.0).float()
                acc.sub_(fire)
                v.index_add_(0, in_idx, fire * kick)
            spk = v > V_TH
            v.masked_fill_(spk, V_RST)
            g.masked_fill_(spk, 0.0)
            rfc = torch.where(spk, REFRACTORY_STEPS, (rfc - 1).clamp_(min=0)).to(torch.int8)
            s = spk.float()
            buf[t % DELAY_STEPS] = s
            counts += s[record_idx]
            total += s.sum()
        return counts, total / (steps * B)
