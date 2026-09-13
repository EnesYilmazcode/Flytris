"""Convert MaleCNS v1.0 flat-connectome feathers (raw/) into neurons.parquet + edges.npz.

Node rule (same as nftechie/doomfly, gives 166,700): annotation rows with a non-empty
superclass and status != 'Glia'. idx is by ascending bodyId.
Edges: every released edge with both ends retained, streamed batch by batch (low RAM).
"""
import os, sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.feather as feather, pyarrow.ipc as ipc

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
ANN = os.path.join(RAW, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
NT = os.path.join(RAW, "body-neurotransmitters-male-cns-v1.0.feather")
EDG = os.path.join(RAW, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

# ACh +1, GABA/Glu -1 (Shiu et al. 2024). Histamine -1 (Cl- channel ort/HisCl1; it is the
# photoreceptor transmitter). Everything else, incl. unclear/missing, +1 (Shiu: "all others excitatory").
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}

a = feather.read_table(ANN, columns=["bodyId", "type", "instance", "somaSide", "rootSide", "superclass", "status"]).to_pandas()
a = a[a.superclass.notna() & (a.superclass != "") & (a.status != "Glia")].sort_values("bodyId", ignore_index=True)
side = a.somaSide.where(a.somaSide.isin(["L", "R", "M"]))
side = side.fillna(a.rootSide.where(a.rootSide.isin(["L", "R"])))
side = side.fillna(a.instance.fillna("").str.extract(r"_([LRM])$")[0]).fillna("")
nt = feather.read_table(NT, columns=["body", "consensus_nt"]).to_pandas().set_index("body").consensus_nt
nts = a.bodyId.map(nt).fillna("missing")
neurons = pd.DataFrame({
    "idx": np.arange(len(a), dtype=np.int32), "body_id": a.bodyId.astype(np.int64),
    "type": a.type.fillna(""), "side": side, "superclass": a.superclass, "nt": nts,
})
neurons.to_parquet(os.path.join(HERE, "neurons.parquet.tmp"), index=False)
os.replace(os.path.join(HERE, "neurons.parquet.tmp"), os.path.join(HERE, "neurons.parquet"))
ids = neurons.body_id.to_numpy()
sign_of = np.array([SIGN.get(x, 1) for x in nts], dtype=np.int8)
print("neurons", len(ids), flush=True)

def to_idx(b):
    j = np.searchsorted(ids, b)
    j[j >= len(ids)] = 0
    return j, ids[j] == b

P, Q, W = [], [], []
src_rows = src_syn = 0
with pa.memory_map(EDG, "r") as src:
    r = ipc.open_file(src)
    print("batches", r.num_record_batches, r.schema, flush=True)
    for k in range(r.num_record_batches):
        b = r.get_batch(k)
        pre = b.column("body_pre").to_numpy(); post = b.column("body_post").to_numpy(); w = b.column("weight").to_numpy()
        src_rows += len(w); src_syn += int(w.sum(dtype=np.int64))
        i, ok1 = to_idx(pre); j, ok2 = to_idx(post); keep = ok1 & ok2
        P.append(i[keep].astype(np.int32)); Q.append(j[keep].astype(np.int32)); W.append(w[keep].astype(np.int32))
        del b, pre, post, w, i, j
        if k % 400 == 0: print(k, src_rows, flush=True)
pre = np.concatenate(P); del P
post = np.concatenate(Q); del Q
syn = np.concatenate(W); del W
sign = sign_of[pre]
print("source rows", src_rows, "source syn", src_syn, "kept", len(pre), "kept syn", int(syn.sum(dtype=np.int64)), flush=True)
tmp = os.path.join(HERE, "edges.tmp.npz")
np.savez(tmp, pre=pre, post=post, syn=syn, sign=sign)
os.replace(tmp, os.path.join(HERE, "edges.npz"))
print("wrote", flush=True)
