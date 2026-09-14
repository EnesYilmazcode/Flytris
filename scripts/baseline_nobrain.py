"""No-brain baseline: the same readout head, population, generations and piece seeds as
train.py, fed the 27 eye channel values (heights, holes, piece) directly. CPU only.

    python -X utf8 scripts/baseline_nobrain.py --until 04:30
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

if __name__ == "__main__":
    train.main(["--player", "nobrain", *sys.argv[1:]])
