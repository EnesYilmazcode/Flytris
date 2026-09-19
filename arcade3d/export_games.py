"""Export every recorded fly game for the 3D arcade render.

Sources: the V2 evolution logs (runs/train_after, runs/train_after_v2) and the 36
V3 validation replays. Each game is replayed with the Python engine and must
reproduce its recorded lines and death before it is written.

Writes games.json (metadata) and games.bin (pieces then moves, one byte each).
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from flytris.tetris import Batch  # noqa: E402

OUT = Path(__file__).resolve().parent


def replay(seed, moves, n):
    """Replay n games sharing one seed. moves is [steps, n] with 255 for dead."""
    game = Batch(n, seed)
    pieces = []
    for row in moves:
        if not game.alive.any():
            break
        pieces.append(game.piece)
        game.step(np.where(game.alive, row, 0))
    return game, np.array(pieces, np.uint8)


def main():
    games, pieces_blob, moves_blob = [], [], []

    def add(source, seed, piece_seq, move_seq, lines, placed, died):
        games.append(dict(src=source, seed=int(seed), n=int(placed), lines=int(lines), died=bool(died),
                          off=sum(len(p) for p in pieces_blob)))
        pieces_blob.append(piece_seq[:placed].astype(np.uint8))
        moves_blob.append(move_seq[:placed].astype(np.uint8))

    for run in ["train_after", "train_after_v2"]:
        for path in sorted(glob.glob(str(ROOT / "runs" / run / "gen_*.npz"))):
            z = np.load(path)
            seed, moves, death, lines = int(z["seed"]), z["moves"], z["death"], z["lines"]
            game, pieces = replay(seed, moves, moves.shape[1])
            assert (game.lines == lines).all(), path
            died = death >= 0
            assert (game.placed[died] == death[died]).all(), path
            for i in range(moves.shape[1]):
                add(f"v2:{run}", seed, pieces, moves[:, i], lines[i], game.placed[i], died[i])

    for path in sorted(glob.glob(str(ROOT / "runs/v3_modal/wide2048/replay_seed*.npz"))):
        z = np.load(path)
        for i, seed in enumerate(z["seeds"]):
            game, pieces = replay(int(seed), z["moves"][:, i:i + 1], 1)
            assert game.lines[0] == z["lines"][i] and game.placed[0] == z["placed"][i], (path, seed)
            add("v3", seed, pieces, z["moves"][:, i], z["lines"][i], z["placed"][i], not game.alive[0])

    pieces = np.concatenate(pieces_blob)
    moves = np.concatenate(moves_blob)
    (OUT / "games.bin").write_bytes(pieces.tobytes() + moves.tobytes())
    meta = dict(total=int(len(pieces)), games=games)
    (OUT / "games.json").write_text(json.dumps(meta, separators=(",", ":")))
    placed = np.array([g["n"] for g in games])
    print(f"{len(games)} games verified, {len(pieces)} placements, "
          f"placed p50={np.median(placed):.0f} max={placed.max()}, "
          f"{sum(g['src'] == 'v3' for g in games)} from V3")


if __name__ == "__main__":
    main()
