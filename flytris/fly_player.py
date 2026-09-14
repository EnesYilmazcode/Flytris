"""Brain + readout -> Tetris placements.

A readout is [14, F + 1]: rows score the 10 columns then 4 rotations from F features
(last column is the bias). A placement scores its rotation plus the mean score of the
columns it covers; the best valid placement is played.
"""
from pathlib import Path

import numpy as np
import torch

from .eyes import board_channels
from .tetris import COLS, PLACEMENTS, drop

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "runs" / "readout_groups.npz"

# Piece-sequence seed ranges; they never overlap.
TRAIN_SEED0 = 100_000          # generation t trains on TRAIN_SEED0 + t
VALID_SEEDS = tuple(range(200_001, 200_006))
EVAL_SEED0 = 300_000           # 50 unseen sequences: 300000..300049
TOURNAMENT_SEED = 400_000
SELECTION_SEED0 = 800_000      # picks the readout groups
PRACTICE_SEED0 = 900_000       # decode test boards
DEAD = 255

N_OUT = COLS + 4


def _placement_matrix(opts):
    a = np.zeros((len(opts), N_OUT), np.float32)
    for j, (r, x, cells) in enumerate(opts):
        w = cells[:, 1].max() + 1
        a[j, x:x + w] = 1 / w
        a[j, COLS + r % 4] = 1
    return a


PLACE_SCORE = [_placement_matrix(opts) for opts in PLACEMENTS]


def neuron_groups(neurons, superclass):
    """Neurons of one superclass pooled by (type, side). Returns (neuron idx, group of each, names)."""
    d = neurons[neurons.superclass == superclass].sort_values("idx")
    keys = list(zip(d.type.fillna(""), d.side.fillna("")))
    names = sorted(set(keys))
    lookup = {k: i for i, k in enumerate(names)}
    return d.idx.values.astype(np.int64), np.array([lookup[k] for k in keys], np.int64), names


class FlyPlayer:
    def __init__(self, brain=None, eyes=None, groups_file=GROUPS, max_batch=64, device="cuda"):
        from .brain import Brain, load_neurons
        from .eyes import Eyes

        with np.load(groups_file) as f:
            g = {k: f[k] for k in f.files}
        self.steps, self.threshold = int(g["steps"]), int(g["threshold"])
        self.brain = brain or Brain(self.threshold, float(g["w_scale"]), device=device)
        self.eyes = eyes or Eyes(float(g["board_hz"]), float(g["piece_hz"]), device=device)
        rec, group_of, names = neuron_groups(load_neurons(["idx", "type", "side", "superclass"]),
                                             str(g["superclass"]))
        chosen = g["groups"]
        slot = np.full(len(names), -1)
        slot[chosen] = np.arange(len(chosen))
        keep = slot[group_of] >= 0
        self.rec_idx = torch.as_tensor(rec[keep], device=device)
        self.slot = torch.as_tensor(slot[group_of[keep]], device=device)
        self.n_features = len(chosen)
        self.mean = torch.as_tensor(g["mean"], device=device)[:, None]
        self.std = torch.as_tensor(g["std"], device=device)[:, None]
        self.max_batch, self.device = max_batch, device

    def features(self, boards, pieces, silenced=False):
        """[B, F] normalized pooled spike counts of the readout groups."""
        out = np.empty((len(boards), self.n_features), np.float32)
        for s in range(0, len(boards), self.max_batch):
            m = min(self.max_batch, len(boards) - s)
            pooled = torch.zeros((self.n_features, m), device=self.device)
            if not silenced:
                p = self.eyes.probs(self.eyes.channels(boards[s:s + m], pieces[s:s + m]))
                counts, _ = self.brain.run(self.eyes.in_idx, p, self.steps, self.rec_idx, self.eyes.phase)
                pooled.index_add_(0, self.slot, counts)
            out[s:s + m] = ((pooled - self.mean) / self.std).T.cpu().numpy()
        return out


class NoBrainPlayer:
    """Same head fed the 27 eye channel values directly."""
    n_features = 27

    def features(self, boards, pieces, silenced=False):
        f = board_channels(boards, pieces)
        return np.zeros_like(f) if silenced else f


def choose(params, feats, piece):
    """params [k, 14, F+1], feats [k, F] -> placement index [k]."""
    logits = np.einsum("kof,kf->ko", params[:, :, :-1], feats) + params[:, :, -1]
    return (logits @ PLACE_SCORE[piece].T).argmax(1)


def play(batches, params, player, cap, mode="brain"):
    """Play several Batches at once (each with its own piece sequence) until every game is
    dead or has placed `cap` pieces. params: one [n, 14, F+1] array per batch.
    mode: "brain", "silenced" (features from zero counts) or "wrong" (each fly sees another
    alive fly's board). Returns one uint8 move log [steps, n] per batch, DEAD for dead flies.
    """
    logs = [[] for _ in batches]
    for _ in range(cap):
        owners = [(i, np.nonzero(b.alive)[0]) for i, b in enumerate(batches) if b.alive.any()]
        if not owners:
            break
        boards = np.concatenate([batches[i].boards[idx] for i, idx in owners])
        pieces = np.concatenate([np.full(len(idx), batches[i].piece) for i, idx in owners])
        if mode == "wrong":
            boards = np.roll(boards, 1, axis=0)
        feats = player.features(boards, pieces, silenced=mode == "silenced")
        off = 0
        for i, idx in owners:
            b = batches[i]
            c = np.zeros(b.n, np.int64)
            c[idx] = choose(params[i][idx], feats[off:off + len(idx)], b.piece)
            off += len(idx)
            move = np.full(b.n, DEAD, np.uint8)
            move[idx] = c[idx]
            logs[i].append(move)
            b.step(c)
    return [np.stack(m) if m else np.zeros((0, b.n), np.uint8) for m, b in zip(logs, batches)]


GROUPS_AFTER = ROOT / "runs" / "readout_groups_after.npz"


class AfterstateFlyPlayer(FlyPlayer):
    """Shows the eyes each board a landing would leave (R7/R8 silent: piece_hz is 0 in the
    groups file). The brain is deterministic, so each distinct board is simulated once."""

    def __init__(self, groups_file=GROUPS_AFTER, max_batch=384, cache_size=400_000, device="cuda"):
        super().__init__(groups_file=groups_file, max_batch=max_batch, device=device)
        self.cache, self.cache_size = {}, cache_size

    def board_features(self, boards):
        keys = [b.tobytes() for b in boards]
        out = np.empty((len(boards), self.n_features), np.float32)
        miss = []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is None:
                miss.append(i)
            else:
                out[i] = hit
        if miss:
            first = {}
            for i in miss:
                first.setdefault(keys[i], i)
            f = self.features(boards[list(first.values())], np.zeros(len(first), np.int64))
            fresh = dict(zip(first.keys(), f))
            for i in miss:
                out[i] = fresh[keys[i]]
            if len(self.cache) + len(fresh) > self.cache_size:
                self.cache.clear()
            self.cache.update(fresh)
        return out


class AfterstateNoBrainPlayer:
    """Same after-state head fed the heights and holes channels directly."""
    n_features = 2 * COLS

    def board_features(self, boards):
        return board_channels(boards, np.zeros(len(boards), np.int64))[:, :2 * COLS]


def play_afterstate(batches, params, player, cap, mode="brain", seed=0):
    """Like play(), but every fly scores each landing of the current piece and plays the best.
    Landings that end the game are skipped unless there is no other. params: one [n, F+1]
    array per batch. mode: "brain", "silenced" (zero features) or "shuffled" (each fly's
    candidate features permuted among its own landings)."""
    rng = np.random.default_rng(seed)
    logs = [[] for _ in batches]
    for _ in range(cap):
        live = [i for i, b in enumerate(batches) if b.alive.any()]
        if not live:
            break
        boards, owner_b, owner_f, move = [], [], [], []
        for i in live:
            b = batches[i]
            idx = np.nonzero(b.alive)[0]
            for j in range(b.options):
                nb, _, dead = drop(b.boards[idx], b.piece, j)
                ok = ~dead
                boards.append(nb[ok])
                owner_b.append(np.full(ok.sum(), i))
                owner_f.append(idx[ok])
                move.append(np.full(ok.sum(), j))
        boards = np.concatenate(boards)
        owner_b, owner_f, move = map(np.concatenate, (owner_b, owner_f, move))
        group = owner_b * (max(b.n for b in batches) + 1) + owner_f
        if len(boards):
            if mode == "silenced":
                feats = np.zeros((len(boards), player.n_features), np.float32)
            else:
                feats = player.board_features(boards)
            if mode == "shuffled":
                by_group = np.argsort(group, kind="stable")
                feats[by_group] = feats[np.lexsort((rng.random(len(group)), group))]
            w = np.concatenate([params[i][owner_f[owner_b == i]] for i in live])
            order_in = np.concatenate([np.nonzero(owner_b == i)[0] for i in live])
            value = np.empty(len(boards), np.float32)
            value[order_in] = (feats[order_in] * w[:, :-1]).sum(1) + w[:, -1]
            best = np.lexsort((-value, group))
            first = best[np.r_[True, group[best][1:] != group[best][:-1]]]
        else:
            first = np.zeros(0, np.int64)
        for i in live:
            b = batches[i]
            c = np.zeros(b.n, np.int64)
            pick = first[owner_b[first] == i]
            c[owner_f[pick]] = move[pick]
            log = np.full(b.n, DEAD, np.uint8)
            log[b.alive] = c[b.alive]
            logs[i].append(log)
            b.step(c)
    return [np.stack(m) if m else np.zeros((0, b.n), np.uint8) for m, b in zip(logs, batches)]


def death_index(batch, cap):
    """Pieces placed before dying, or -1 if the game reached the cap alive."""
    return np.where(batch.alive, -1, batch.placed).astype(np.int16)
