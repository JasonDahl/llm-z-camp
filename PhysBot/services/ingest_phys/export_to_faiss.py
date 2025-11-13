# services/ingest_phys/export_to_faiss.py
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import os, pickle, pathlib, numpy as np, faiss

ES_HOST   = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX  = os.getenv("ES_INDEX", "physbot_units")
OUT_DIR   = pathlib.Path("artifacts/phys_demo")
EMB_FIELD = os.getenv("EMB_FIELD", "embedding")  # change if your field differs

# Connect
es = Elasticsearch(ES_HOST)

# Stream docs (match_all) pulling only what we need
# If your index is large, scan() is simpler than scroll plumbing.
docs = scan(
    es,
    index=ES_INDEX,
    query={"query": {"match_all": {}}},
    _source=[EMB_FIELD, "text", "doc", "page", "unit", "section", "title"],
    size=1000
)

vecs, meta = [], []
n = 0
for d in docs:
    src = d.get("_source", {})
    emb = src.get(EMB_FIELD)
    if not emb:
        continue
    vecs.append(emb)
    # keep metadata lean but useful
    meta.append({
        "text":   src.get("text", ""),
        "doc":    src.get("doc"),
        "page":   src.get("page"),
        "unit":   src.get("unit"),
        "section":src.get("section"),
        "title":  src.get("title"),
    })
    n += 1

if not vecs:
    raise RuntimeError(f"No embeddings found in index '{ES_INDEX}' under field '{EMB_FIELD}'.")

xb = np.array(vecs, dtype="float32")
# L2-normalize for cosine/IP
faiss.normalize_L2(xb)

# Build FAISS (inner-product)
d = xb.shape[1]
index = faiss.IndexFlatIP(d)
index.add(xb)

OUT_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(OUT_DIR / "store.faiss"))
with open(OUT_DIR / "store.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"Exported {len(meta)} vectors → {OUT_DIR}/store.faiss & store.pkl (dim={d})")
