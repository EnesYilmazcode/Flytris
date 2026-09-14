"""Tetris board -> a retinotopic picture on the fly's photoreceptors (experimental).

Each R1-R6 cell takes the 2D optic-lobe hex position of its strongest L1/L2 partner:
u = hex1 - hex2 (horizontal), v = hex1 + hex2 (vertical). Positions are rank-normalized
per eye (vertical rank within each column band) and binned onto the board: the left eye sees columns 0-4, the right eye 5-9,
both see all 20 visible rows. As in eyes.py, low u is the outer edge of each eye.
Low v is the TOP of the board (an arbitrary choice). A filled cell is light: its
photoreceptors fire at board_hz, empty cells are silent. The falling piece is drawn
at its spawn position in the top rows. R7/R8 carry the piece id as in eyes.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .eyes import PHASE_SEED
from .tetris import COLS, HIDDEN, PLACEMENTS, ROWS

DATA = Path(__file__).resolve().parents[1] / "data" / "malecns"
MAP = DATA / "eye_image_map.npz"


def _rank01(x):
    return (np.argsort(np.argsort(x, kind="stable"), kind="stable") + 0.5) / len(x)


def build_image_map():
    nr = pd.read_parquet(DATA / "neurons.parquet", columns=["idx", "body_id", "type", "side"])
    ann = pd.read_feather(DATA / "raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                          columns=["bodyId", "assignedOlHex1", "assignedOlHex2"]).set_index("bodyId")
    lam = nr[nr.type.isin(["L1", "L2"])]
    h1 = np.full(len(nr), np.nan)
    h2 = np.full(len(nr), np.nan)
    h1[lam.idx] = ann.assignedOlHex1.reindex(lam.body_id).values
    h2[lam.idx] = ann.assignedOlHex2.reindex(lam.body_id).values

    r16 = nr[nr.type == "R1-R6"]
    is_r = np.zeros(len(nr), bool)
    is_r[r16.idx] = True
    with np.load(DATA / "edges.npz") as z:
        pre = z["pre"]
        keep = np.nonzero(is_r[pre])[0]
        pre = pre[keep]
        post = z["post"][keep]
        syn = z["syn"][keep]
    ok = ~np.isnan(h1[post]) & ~np.isnan(h2[post])
    pre, post, syn = pre[ok], post[ok], syn[ok]
    order = np.lexsort((-syn, pre))
    first = np.unique(pre[order], return_index=True)[1]
    best = post[order][first]
    ru = np.full(len(nr), np.nan)
    rv = np.full(len(nr), np.nan)
    ru[pre[order][first]] = h1[best] - h2[best]
    rv[pre[order][first]] = h1[best] + h2[best]

    in_idx, pix, chan, scale, counts = [], [], [], [], {}
    for side in "LR":
        rs = r16[(r16.side == side) & ~np.isnan(ru[r16.idx])].sort_values("body_id")
        counts[side] = len(rs)
        c = np.minimum((_rank01(ru[rs.idx]) * 5).astype(int), 4)
        col = c if side == "L" else COLS - 1 - c
        v = np.zeros(len(rs))
        vs = rv[rs.idx]
        for k in range(5):
            v[c == k] = _rank01(vs[c == k])
        row = np.minimum((v * ROWS).astype(int), ROWS - 1)
        in_idx.append(rs.idx.values)
        pix.append(row * COLS + col)
        chan.append(np.full(len(rs), -1))
        r78 = nr[nr.type.str.match(r"^R[78]", na=False) & (nr.side == side)].sort_values("body_id")
        in_idx.append(r78.idx.values)
        pix.append(np.full(len(r78), -1))
        chan.append(np.arange(len(r78)) % 7)
    for i, side in enumerate("LR"):
        scale.extend([np.full(len(in_idx[2 * i]), max(counts.values()) / counts[side], np.float32),
                      np.ones(len(in_idx[2 * i + 1]), np.float32)])
    pix = np.concatenate(pix)
    covered = np.bincount(pix[pix >= 0], minlength=ROWS * COLS)
    np.savez(MAP, in_idx=np.concatenate(in_idx).astype(np.int64), pix=pix.astype(np.int64),
             chan=np.concatenate(chan).astype(np.int64), scale=np.concatenate(scale),
             n_left=counts["L"], n_right=counts["R"], empty_pixels=int((covered == 0).sum()),
             min_cells_per_pixel=int(covered.min()), max_cells_per_pixel=int(covered.max()))
    return np.load(MAP)


def board_image(boards, pieces):
    """[B, ROWS*COLS] 1 where a block (or the spawning piece) is, else 0."""
    img = (boards[:, HIDDEN:, :] != 0).astype(np.float32)
    for p in np.unique(pieces):
        _, _, cells = PLACEMENTS[p][0]
        x = (COLS - (cells[:, 1].max() + 1)) // 2
        b = np.nonzero(pieces == p)[0]
        img[b[:, None], cells[:, 0], cells[:, 1] + x] = 1
    return img.reshape(len(boards), -1)


class EyesImage:
    def __init__(self, board_hz=500.0, piece_hz=500.0, device="cuda"):
        m = np.load(MAP) if MAP.exists() else build_image_map()
        self.info = {k: int(m[k]) for k in ("n_left", "n_right", "empty_pixels",
                                            "min_cells_per_pixel", "max_cells_per_pixel")}
        self.in_idx = torch.from_numpy(m["in_idx"]).to(device)
        self.pix, self.chan = m["pix"], m["chan"]
        self.scale = torch.from_numpy(m["scale"]).to(device)
        rng = np.random.default_rng(PHASE_SEED)
        self.phase = torch.from_numpy(rng.random(len(m["in_idx"]), dtype=np.float32)).to(device)
        self.board_hz, self.piece_hz, self.device = board_hz, piece_hz, device

    def probs(self, boards, pieces):
        """boards [B, H, COLS], pieces [B] -> input spikes per step [n_in, B]."""
        pieces = np.asarray(pieces)
        img = board_image(boards, pieces)
        onehot = np.eye(7, dtype=np.float32)[pieces]
        vals = np.where(self.pix[None, :] >= 0,
                        img[:, np.maximum(self.pix, 0)] * self.board_hz,
                        onehot[:, np.maximum(self.chan, 0)] * self.piece_hz)
        p = torch.as_tensor(vals.T.astype(np.float32), device=self.device) * self.scale[:, None] * 1e-3
        return p.clamp_(0, 1)
