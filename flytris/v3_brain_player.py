"""Runtime for V3 individual-neuron, temporally binned fly-brain readouts."""
from pathlib import Path

import numpy as np
import torch

from .brain import Brain
from .eyes import Eyes, PixelEyes
from .v3_policy import enumerate_transitions


class TemporalNeuronPlayer:
    """Turn exact connectome spikes into normalized features for after-state scoring."""

    def __init__(self, readout_file, max_batch=384, cache_size=400_000, device="cuda"):
        with np.load(Path(readout_file)) as saved:
            neuron_idx = saved["neuron_idx"].astype(np.int64)
            time_bin = saved["time_bin"].astype(np.int64)
            self.mean = saved["mean"].astype(np.float32)
            self.scale = saved["scale"].astype(np.float32)
            self.weights = saved["weights"].astype(np.float32)
            self.head = str(saved["head"]) if "head" in saved else "linear"
            if self.head.startswith("mlp"):
                self.mlp_w1 = saved["mlp_w1"].astype(np.float32)
                self.mlp_b1 = saved["mlp_b1"].astype(np.float32)
                self.mlp_w2 = saved["mlp_w2"].astype(np.float32)
                self.mlp_b2 = float(saved["mlp_b2"])
            self.steps = int(saved["steps"])
            self.bin_steps = int(saved["bin_steps"])
            eyes_mode = str(saved["eyes_mode"])
            self.stimulus_mode = str(saved["stimulus_mode"]) if "stimulus_mode" in saved else "after"
        unique, inverse = np.unique(neuron_idx, return_inverse=True)
        self.rec_idx = torch.as_tensor(unique, device=device)
        self.selector = torch.as_tensor(time_bin * len(unique) + inverse, device=device)
        self.brain = Brain(threshold=5, w_scale=1.0, device=device)
        self.eyes = PixelEyes(board_hz=500, device=device) if eyes_mode == "pixel" else Eyes(
            board_hz=500, piece_hz=0.0, device=device)
        self.n_features = len(self.weights)
        self.max_batch = max_batch
        self.cache_size = cache_size
        self.cache = {}

    def score_features(self, features, groups=None):
        if self.head.startswith("mlp"):
            if groups is not None:
                features = features.copy()
                for group in np.unique(groups):
                    idx = groups == group
                    features[idx] -= features[idx].mean(0)
            hidden = np.tanh(features @ self.mlp_w1 + self.mlp_b1)
            return hidden @ self.mlp_w2 + self.mlp_b2
        return features @ self.weights

    def _simulate(self, before, after=None):
        values = np.empty((len(before), self.n_features), np.float32)
        for start in range(0, len(before), self.max_batch):
            stop = min(start + self.max_batch, len(before))
            zeros = np.zeros(stop - start, np.int64)
            p = self.eyes.probs(self.eyes.channels(before[start:stop], zeros))
            kwargs = {}
            if after is not None:
                kwargs = {
                    "p_in_after": self.eyes.probs(self.eyes.channels(after[start:stop], zeros)),
                    "switch_step": self.steps // 2,
                }
            counts, _ = self.brain.run(self.eyes.in_idx, p, self.steps, self.rec_idx,
                                       self.eyes.phase, bin_steps=self.bin_steps, **kwargs)
            raw = counts[self.selector].T.cpu().numpy()
            values[start:stop] = (raw - self.mean) / self.scale
        return values

    def board_features(self, boards):
        keys = [board.tobytes() for board in boards]
        out = np.empty((len(boards), self.n_features), np.float32)
        missing = []
        for i, key in enumerate(keys):
            value = self.cache.get(key)
            if value is None:
                missing.append(i)
            else:
                out[i] = value
        if missing:
            first = {}
            for i in missing:
                first.setdefault(keys[i], i)
            indices = list(first.values())
            fresh_values = self._simulate(boards[indices])
            fresh = dict(zip(first.keys(), fresh_values))
            if len(self.cache) + len(fresh) > self.cache_size:
                self.cache.clear()
            self.cache.update(fresh)
            for i in missing:
                out[i] = fresh[keys[i]]
        return out

    def transition_features(self, landed, after):
        keys = [a.tobytes() + b.tobytes() for a, b in zip(landed, after)]
        out = np.empty((len(landed), self.n_features), np.float32)
        missing = []
        for i, key in enumerate(keys):
            value = self.cache.get(key)
            if value is None:
                missing.append(i)
            else:
                out[i] = value
        if missing:
            first = {}
            for i in missing:
                first.setdefault(keys[i], i)
            indices = list(first.values())
            values = self._simulate(landed[indices], after[indices])
            fresh = dict(zip(first.keys(), values))
            if len(self.cache) + len(fresh) > self.cache_size:
                self.cache.clear()
            self.cache.update(fresh)
            for i in missing:
                out[i] = fresh[keys[i]]
        return out


def play_transition_games(batches, player, cap, record_states=False):
    """Play one-fly games using only continuous transition-evoked neural activity."""
    if any(batch.n != 1 for batch in batches):
        raise ValueError("transition gameplay currently requires one fly per Batch")
    states = [[] for _ in batches]
    moves = [[] for _ in batches]
    for step in range(cap):
        landed_all, after_all, owner_all, move_all = [], [], [], []
        fallback = {}
        for owner, batch in enumerate(batches):
            if not batch.alive[0]:
                continue
            if record_states:
                states[owner].append((batch.boards[0].copy(), int(batch.piece)))
            landed, after, _, dead = enumerate_transitions(batch.boards[0], batch.piece)
            valid = np.flatnonzero(~dead)
            fallback[owner] = 0
            if len(valid):
                landed_all.append(landed[valid])
                after_all.append(after[valid])
                owner_all.append(np.full(len(valid), owner, np.int32))
                move_all.append(valid)
        chosen = fallback.copy()
        if landed_all:
            landed = np.concatenate(landed_all)
            after = np.concatenate(after_all)
            owners = np.concatenate(owner_all)
            options = np.concatenate(move_all)
            score = player.score_features(player.transition_features(landed, after), owners)
            for owner in np.unique(owners):
                idx = np.flatnonzero(owners == owner)
                chosen[int(owner)] = int(options[idx[np.argmax(score[idx])]])
        if not chosen:
            break
        for owner, move in chosen.items():
            moves[owner].append(move)
            batches[owner].step([move])
        if (step + 1) % 25 == 0:
            alive = sum(bool(batch.alive[0]) for batch in batches)
            best = max(int(batch.lines[0]) for batch in batches)
            print(f"gameplay step {step + 1}: {alive}/{len(batches)} alive, "
                  f"best {best} lines", flush=True)
    return {"states": states, "moves": moves}


def play_transition_population(batch, params, player, cap):
    """Fair tournament: all neural readouts receive one shared piece sequence."""
    params = np.asarray(params)
    for step in range(cap):
        alive = np.flatnonzero(batch.alive)
        if not len(alive):
            break
        landed_all, after_all, owners_all, moves_all = [], [], [], []
        for owner in alive:
            landed, after, _, dead = enumerate_transitions(batch.boards[owner], batch.piece)
            valid = np.flatnonzero(~dead)
            if len(valid):
                landed_all.append(landed[valid])
                after_all.append(after[valid])
                owners_all.append(np.full(len(valid), owner, np.int32))
                moves_all.append(valid)
        choice = np.zeros(batch.n, np.int64)
        if landed_all:
            owners = np.concatenate(owners_all)
            moves = np.concatenate(moves_all)
            features = player.transition_features(np.concatenate(landed_all), np.concatenate(after_all))
            value = (features * params[owners, :-1]).sum(1) + params[owners, -1]
            for owner in np.unique(owners):
                idx = np.flatnonzero(owners == owner)
                choice[owner] = moves[idx[np.argmax(value[idx])]]
        batch.step(choice)
        if (step + 1) % 25 == 0:
            print(f"gameplay step {step + 1}: {int(batch.alive.sum())}/{batch.n} alive, "
                  f"best {int(batch.lines.max())} lines", flush=True)
    return batch
