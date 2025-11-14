# services/ingest_phys/export_to_faiss.py
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import os
import pickle
import pathlib
import numpy as np
import faiss

ES_HOST   = os.getenv("ES_HOST",  "http://localhost:9200")
ES_INDEX  = os.getenv("ES_INDEX", "physbot_units")
EMB_FIELD = os.getenv("EMB_FIELD", "embedding")  # field holding the vector

OUT_DIR   = pathlib.Path("artifacts/phys_demo")

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST)

# Stream docs from ES; only request fields we know exist in the mapping
docs = scan(
    es,
    index=ES_INDEX,
    query={"query": {"match_all": {}}},
    _source=[EMB_FIELD, "unit", "section", "content", "source", "chunk_index"],
    size=1000,
)

vecs, meta = [], []

for h in docs:
    s = h.get("_source", {})
    emb = s.get(EMB_FIELD)
    if emb is None:
        continue

    vecs.append(emb)

    meta.append({
        "unit":        s.get("unit"),          # e.g. "111"
        "section":     s.get("section"),       # e.g. "Work-Energy Theorem"
        "content":     s.get("content", ""),   # main text
        "source":      s.get("source", ""),    # filename or unit title
        "chunk_index": s.get("chunk_index"),   # int

        # convenience fields for UI / context builder:
        "doc":         s.get("source", ""),                # label for UI
        "page":        s.get("page", s.get("chunk_index")) # fallback "page-ish"
    })

if not vecs:
    raise RuntimeError(
        f"No embeddings found in index '{ES_INDEX}' under field '{EMB_FIELD}'."
    )

xb = np.array(vecs, dtype="float32").astype("float32")
# L2-normalize for cosine/IP
faiss.normalize_L2(xb)

d = xb.shape[1]
index = faiss.IndexFlatIP(d)
index.add(xb)

OUT_DIR.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(OUT_DIR / "store.faiss"))
with open(OUT_DIR / "store.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"Exported {len(meta)} vectors → {OUT_DIR}/store.faiss & store.pkl (dim={d})")
