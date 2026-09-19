"""Render real MaleCNS neuron skeletons as a glowing image for the README.

Downloads SWC skeletons from the public MaleCNS v1.0 bucket (cached under
data/malecns/skeletons/) for the neurons Flytris uses (photoreceptors in, L1/L2 read out),
plus descending, visual projection and central neurons for the outline, and draws them.

  python scripts/render_connectome_art.py
"""
import concurrent.futures as cf
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/malecns/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
CACHE = ROOT / "data/malecns/skeletons"
URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"
OUT = ROOT / "media/connectome.png"

# (label, selector, count, color, brightness)
GROUPS = [
    ("central", lambda a: a.superclass == "cb_intrinsic", 260, (80, 110, 255), 0.05),
    ("visual projection", lambda a: a.superclass == "visual_projection", 160, (150, 90, 255), 0.12),
    ("descending", lambda a: a.superclass == "descending_neuron", 220, (255, 160, 50), 0.06),
    ("photoreceptors", lambda a: a.type.astype(str).str.match(r"^R[1-8]"), 260, (30, 220, 255), 0.6),
    ("L1/L2", lambda a: a.type.astype(str).isin(["L1", "L2"]), 240, (255, 50, 180), 0.6),
]


def fetch(body):
    path = CACHE / f"{body}.swc"
    if not path.exists():
        try:
            with urllib.request.urlopen(URL.format(body), timeout=60) as r:
                data = r.read()
        except Exception:
            return None
        if len(data) > 6_000_000:
            return None
        path.write_bytes(data)
    return path


def segments(path):
    rows = [l.split() for l in path.read_text().splitlines() if l and not l.startswith("#")]
    a = np.array(rows, dtype=float)
    idx = {int(i): k for k, i in enumerate(a[:, 0])}
    xyz = a[:, 2:5]
    par = np.array([idx.get(int(p), -1) for p in a[:, 6]])
    keep = par >= 0
    return xyz[keep], xyz[par[keep]]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ann = pd.read_feather(RAW, columns=["bodyId", "type", "superclass"])
    rng = np.random.default_rng(3)
    picks = []
    weights = []
    for label, sel, count, color, weight in GROUPS:
        ids = ann[sel(ann)].bodyId.to_numpy()
        ids = rng.choice(ids, size=min(count, len(ids)), replace=False)
        picks += [(int(b), color, label) for b in ids]
        weights += [weight] * len(ids)
    with cf.ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(lambda p: fetch(p[0]), picks))
    print(f"{sum(p is not None for p in paths)}/{len(picks)} skeletons")

    segs = [(segments(p), c, w) for p, (_, c, _), w in zip(paths, picks, weights) if p is not None]
    brain = np.vstack([s[0][0] for s in segs if len(s[0][0])])
    # Front view: x across, y down. Frame the brain as a wide banner.
    lo, hi = np.percentile(brain[:, :2], 0.3, axis=0), np.percentile(brain[:, :2], 99.7, axis=0)
    W, H = 3200, 1600
    scale = min(0.94 * W / (hi[0] - lo[0]), 0.94 * H / (hi[1] - lo[1]))
    off = np.array([W, H]) / 2 - scale * (lo + hi) / 2
    acc = np.zeros((H, W, 3))
    for (a, b), color, weight in segs:
        pa, pb = a[:, :2] * scale + off, b[:, :2] * scale + off
        n = np.maximum(1, np.ceil(np.linalg.norm(pb - pa, axis=1))).astype(int)
        t = np.concatenate([np.arange(k) / k for k in n])
        start = np.repeat(pa, n, axis=0)
        pts = start + (np.repeat(pb, n, axis=0) - start) * t[:, None]
        x, y = pts[:, 0].astype(int), pts[:, 1].astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        np.add.at(acc, (y[ok], x[ok]), np.array(color) / 255 * weight)
    from scipy.ndimage import gaussian_filter
    glow = gaussian_filter(acc, sigma=(10, 10, 0))
    light = acc * 0.9 + glow * 6.0
    img = 1 - np.exp(-light * 1.4)
    img = (np.clip(img, 0, 1) ** 0.9 * 255).astype(np.uint8)
    Image.fromarray(img).resize((1600, 800), Image.LANCZOS).save(OUT)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
