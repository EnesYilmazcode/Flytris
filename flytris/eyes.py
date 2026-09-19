"""Tetris board -> input spike rates on the fly's photoreceptors.

Columns 0-4 go to the left eye's R1-R6 and columns 5-9 to the right eye's, as
five strips per eye. R1-R6 have no position in the data, so each takes the
optic-lobe hex column of its strongest L1/L2 partner. Strips are bands along
hex1 - hex2, the horizontal axis (hex1 + hex2 tracks dorsal-ventral soma position,
r = -0.98). Low hex1 - hex2 is treated as posterior, so the outer strips of
each eye see the board's outer columns. Each strip is split in half by body id:
one half signals column height, the other holes. R7/R8 in both eyes are split
into seven groups that signal the current piece.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tetris import COLS, H, ROWS, column_tops

DATA = Path(__file__).resolve().parents[1] / "data" / "malecns"
MAP = DATA / "eye_map.npz"
PIXEL_MAP = DATA / "pixel_eye_map.npz"
N_CHANNELS = 2 * COLS + 7
PHASE_SEED = 20260913


def board_channels(boards, pieces):
    """[B, N_CHANNELS] values in 0..1: column heights, holes per column, piece one-hot."""
    top = column_tops(boards)
    heights = np.minimum(H - top, ROWS) / ROWS
    below = np.arange(H)[None, :, None] > top[:, None, :]
    holes = np.minimum((below & (boards == 0)).sum(1), 8) / 8
    piece = np.eye(7)[np.asarray(pieces)]
    return np.concatenate([heights, holes, piece], 1).astype(np.float32)


def build_eye_map():
    nr = pd.read_parquet(DATA / "neurons.parquet", columns=["idx", "body_id", "type", "side"])
    ann = pd.read_feather(DATA / "raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                          columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    hex_axis = pd.Series((ann.assignedOlHex1 - ann.assignedOlHex2).values, index=ann.bodyId)
    lam = nr[nr.type.isin(["L1", "L2"])]
    lam_pos = np.full(len(nr), np.nan)
    lam_pos[lam.idx] = hex_axis.reindex(lam.body_id).values

    r16 = nr[nr.type == "R1-R6"]
    is_r = np.zeros(len(nr), bool)
    is_r[r16.idx] = True
    z = np.load(DATA / "edges.npz")
    pre = z["pre"]
    keep = np.nonzero(is_r[pre])[0]
    pre = pre[keep]
    post = z["post"][keep]
    syn = z["syn"][keep]
    ok = ~np.isnan(lam_pos[post])
    pre, post, syn = pre[ok], post[ok], syn[ok]
    order = np.lexsort((-syn, pre))
    first = np.unique(pre[order], return_index=True)[1]
    r_pos = np.full(len(nr), np.nan)
    r_pos[pre[order][first]] = lam_pos[post[order][first]]

    in_idx, channel, scale = [], [], []
    counts = {}
    for side in "LR":
        rs = r16[(r16.side == side) & ~np.isnan(r_pos[r16.idx])]
        counts[side] = len(rs)
        rs = rs.assign(pos=r_pos[rs.idx]).sort_values(["pos", "body_id"])
        strip = np.minimum(np.arange(len(rs)) * 5 // len(rs), 4)
        col = strip if side == "L" else COLS - 1 - strip
        half = np.zeros(len(rs), int)
        for s in range(5):
            m = strip == s
            half[m] = np.argsort(np.argsort(rs.body_id.values[m])) % 2
        in_idx.append(rs.idx.values)
        channel.append(col + COLS * half)
        r78 = nr[nr.type.str.match(r"^R[78]", na=False) & (nr.side == side)].sort_values("body_id")
        in_idx.append(r78.idx.values)
        channel.append(2 * COLS + np.arange(len(r78)) % 7)
    n_l, n_r = counts["L"], counts["R"]
    for i, side in enumerate("LR"):
        k = 2 * i
        s = np.ones(len(in_idx[k]), np.float32) * (max(n_l, n_r) / counts[side])
        scale.extend([s, np.ones(len(in_idx[k + 1]), np.float32)])
    np.savez(MAP, in_idx=np.concatenate(in_idx).astype(np.int64),
             channel=np.concatenate(channel).astype(np.int64),
             scale=np.concatenate(scale), n_r16_left=n_l, n_r16_right=n_r,
             n_r16_total=len(r16))
    return np.load(MAP)


class Eyes:
    def __init__(self, board_hz=200.0, piece_hz=150.0, device="cuda"):
        m = np.load(MAP) if MAP.exists() else build_eye_map()
        self.info = {k: int(m[k]) for k in ("n_r16_left", "n_r16_right", "n_r16_total")}
        self.in_idx = torch.from_numpy(m["in_idx"]).to(device)
        self.channel = torch.from_numpy(m["channel"]).to(device)
        self.scale = torch.from_numpy(m["scale"]).to(device)
        rng = np.random.default_rng(PHASE_SEED)
        self.phase = torch.from_numpy(rng.random(len(m["in_idx"]), dtype=np.float32)).to(device)
        self.board_hz, self.piece_hz, self.device = board_hz, piece_hz, device

    def channels(self, boards, pieces):
        """[N_CHANNELS, B] values in 0..1 from boards [B, H, COLS] and piece ids [B]."""
        return board_channels(boards, pieces).T

    def probs(self, values, board_gain=1.0):
        """values [N_CHANNELS, B] -> input spikes per step [n_in, B]."""
        v = torch.as_tensor(values, device=self.device)
        hz = torch.full((N_CHANNELS, 1), self.board_hz * board_gain, device=self.device)
        hz[2 * COLS:] = self.piece_hz
        p = (v * hz)[self.channel] * self.scale[:, None] * 1e-3
        return p.clamp_(0, 1)


def build_pixel_eye_map():
    """Map R1-R6 photoreceptors retinotopically onto the visible 20x10 board."""
    nr = pd.read_parquet(DATA / "neurons.parquet", columns=["idx", "body_id", "type", "side"])
    ann = pd.read_feather(DATA / "raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                          columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    pos = ann.set_index("bodyId")[["assignedOlHex1", "assignedOlHex2"]]
    lam = nr[nr.type.isin(["L1", "L2"])]
    h1 = np.full(len(nr), np.nan)
    h2 = np.full(len(nr), np.nan)
    h1[lam.idx] = pos.assignedOlHex1.reindex(lam.body_id).values
    h2[lam.idx] = pos.assignedOlHex2.reindex(lam.body_id).values

    r16 = nr[nr.type == "R1-R6"]
    is_r = np.zeros(len(nr), bool)
    is_r[r16.idx] = True
    z = np.load(DATA / "edges.npz")
    edge = np.nonzero(is_r[z["pre"]])[0]
    pre, post, syn = z["pre"][edge], z["post"][edge], z["syn"][edge]
    valid = ~np.isnan(h1[post]) & ~np.isnan(h2[post])
    pre, post, syn = pre[valid], post[valid], syn[valid]
    order = np.lexsort((-syn, pre))
    first = np.unique(pre[order], return_index=True)[1]
    partner = np.full(len(nr), -1, np.int64)
    partner[pre[order][first]] = post[order][first]

    def rank_bins(values, bins):
        rank = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return np.minimum(rank * bins // len(rank), bins - 1)

    in_idx, channel = [], []
    for side in "LR":
        rs = r16[(r16.side == side) & (partner[r16.idx] >= 0)].copy()
        p = partner[rs.idx]
        horizontal = h1[p] - h2[p]
        vertical = h1[p] + h2[p]
        col_half = rank_bins(horizontal, COLS // 2)
        col = col_half if side == "L" else COLS - 1 - col_half
        row = np.empty(len(rs), np.int64)
        for c in range(COLS // 2):
            use = np.flatnonzero(col_half == c)
            row[use] = rank_bins(vertical[use], ROWS)
        in_idx.append(rs.idx.to_numpy(np.int64))
        channel.append((row * COLS + col).astype(np.int64))
    np.savez(PIXEL_MAP, in_idx=np.concatenate(in_idx), channel=np.concatenate(channel),
             rows=ROWS, cols=COLS, method="strongest_L1_L2_partner_rank_bins")
    return np.load(PIXEL_MAP)


class PixelEyes:
    """Raw visible board occupancy projected onto inferred retinal coordinates."""
    n_channels = ROWS * COLS

    def __init__(self, board_hz=500.0, device="cuda"):
        m = np.load(PIXEL_MAP) if PIXEL_MAP.exists() else build_pixel_eye_map()
        self.in_idx = torch.from_numpy(m["in_idx"]).to(device)
        self.channel = torch.from_numpy(m["channel"]).to(device)
        rng = np.random.default_rng(PHASE_SEED)
        self.phase = torch.from_numpy(rng.random(len(m["in_idx"]), dtype=np.float32)).to(device)
        self.board_hz, self.device = board_hz, device

    def channels(self, boards, pieces=None):
        visible = (np.asarray(boards)[:, H - ROWS:] != 0).astype(np.float32)
        return visible.reshape(len(visible), -1).T

    def probs(self, values, board_gain=1.0):
        v = torch.as_tensor(values, device=self.device)
        p = v[self.channel] * (self.board_hz * board_gain * 1e-3)
        return p.clamp_(0, 1)
