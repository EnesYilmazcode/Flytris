# Flytris data

## malecns/ (primary): MaleCNS v1.0, adult male fly brain + VNC

Source: FlyEM flat-connectome release, https://male-cns.janelia.org/download/
(files live in the public bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`).

| raw/ file | bytes | sha256 |
|---|---|---|
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14,483,314 | 2177e246...a9a3b2 |
| body-neurotransmitters-male-cns-v1.0.feather | 43,282,834 | 95c92892...879621 |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1,051,241,946 | e35da783...afc1 |

Base URL: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
Hashes match the ones pinned by nftechie/doomfly and cobanov/flyjump. `raw/` (1.1 GB) can be
deleted once the converted files exist. Rebuild with `python -X utf8 data/malecns/build.py`.

License: CC BY 4.0 ("The Male CNS is licensed under CC-BY", release page).
Cite: Berg et al., "Sexual dimorphism in the complete Drosophila male central nervous system
connectome", Cell 189, 5504-5526.e15 (2026), https://doi.org/10.1016/j.cell.2026.08.015.
Credit FlyEM/HHMI Janelia, Univ. of Cambridge, MRC LMB and Google Research.

### neurons.parquet (166,700 rows)
`idx` int32 (0..N-1, ascending body_id), `body_id` int64, `type`, `side` (L/R/M or ""),
`superclass`, `nt` (consensus_nt: acetylcholine, glutamate, gaba, histamine, unclear,
dopamine, octopamine, serotonin, or "missing" when the body has no NT row).

- Kept: annotation rows with a non-empty `superclass` and `status != "Glia"` (211,577 raw rows,
  33,013 unresolved + 11,864 glia dropped). Same rule as doomfly, same 166,700.
- `side` = `somaSide` if L/R/M, else `rootSide` if L/R, else the `_L/_R` instance suffix.
  Photoreceptors have no soma in the data, so their side comes from `rootSide`. 548 have no side.

### edges.npz (25,582,938 rows)
`pre` int32, `post` int32 (neuron idx), `syn` int32 (synapse count at minconf 0.5), `sign` int8.
All released edges with both ends kept; no weight threshold; 101 self-edges kept. Each row
is a unique (pre, post) pair. Source table has 151,856,684 rows (most touch unkept fragments).

### Sign rule (by presynaptic neuron's `nt`)
- acetylcholine +1; gaba -1; glutamate -1 (Shiu et al. 2024)
- histamine -1 (ort/HisCl1 chloride channels; it is the photoreceptor transmitter, so R->lamina is inhibitory). FlyWire had no histamine class, so this is our choice.
- everything else (dopamine, octopamine, serotonin, unclear, missing) +1, following Shiu et al.'s "only GABA/Glu are inhibitory" convention.
To change it, recompute: `sign = table[nt_code[pre]]` from `neurons.nt`.

## flywire/ (fallback): Shiu et al. 2024 LIF model inputs

From https://github.com/philshiu/Drosophila_brain_model (code MIT, see `flywire/LICENSE`):
`Connectivity_783.parquet` + `Completeness_783.csv` (FlyWire public v783, 138,639 neurons,
15,091,983 edges, has `Excitatory` +-1 and `Excitatory x Connectivity`), and the paper's v630
pair `2023_03_23_connectivity_630_final.parquet` + `2023_03_23_completeness_630_final.csv`.
Row index in the completeness CSV = `Presynaptic_Index`/`Postsynaptic_Index`.
Cite: Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing",
Nature 634, 210-219 (2024); FlyWire: Dorkenwald et al. 2024 and Schlegel et al. 2024 (Nature). FlyWire data is CC BY 4.0.

## Prior art read for loaders
- nftechie/doomfly (MIT): `doom/connectome.py`, `data-provenance/malecns_v1/`
- cobanov/flyjump (custom "Cobanov Template Attribution License 1.0"): `scripts/build-connectome.py`
- jonatasperaza/FlyPong (no license file): uses neuPrint `male-cns:v1.0` with a token instead
