# physbot_core — Shared RAG Inference Engine
*FAISS + OpenAI + optional ElasticSearch backend for multiple bots (PhysBot, OreBot, …)*

`physbot_core` provides the **runtime inference layer** for all bots in this monorepo.  
It abstracts away the vector backend (FAISS or ElasticSearch), assembly of prompts, embedding logic, and provenance formatting.

The Streamlit apps depend *only* on this module — never on ingestion code.

---

# 🧠 Responsibilities

`physbot_core` handles:

- Loading FAISS or ES indexes
- Creating OpenAI embeddings at query time
- Vector search (FAISS or ES)
- Context assembly and citation formatting
- Prompt construction with system/user messages
- Returning both:
  - the answer  
  - the ranked source chunks (for provenance display)

This layer makes the apps deployable *without Elasticsearch*, while still offering full ES support when available.

---

# 📦 Module Structure

```
physbot_core/
│
├── rag_utils.py      # core search + prompt logic
├── settings.py       # environment-backed configuration
└── path_utils.py     # utilities for locating repo root
```

---

# ⚙ settings.py — Unified Configuration

Uses Pydantic (`pydantic-settings`) for structured runtime config.

Fields include:

```python
class AppSettings(BaseSettings):
    app_name: str = "PhysBot"
    deploy_mode: str = "faiss"   # or "elastic"

    # OpenAI
    openai_api_key: str

    # FAISS
    faiss_index_path: str
    faiss_meta_path: str

    # ElasticSearch
    elastic_index: str = "physbot_units"
    elastic_host: str | None = None
    elastic_user: str | None = None
    elastic_pass: str | None = None
```

Supports both:
- `.env` files (local)
- Streamlit Cloud secrets (deployment)

---

# 🔍 rag_utils.py — Core RAG Pipeline

This file contains the inference logic used by *both* PhysBot and OreBot.

## 1. Embeddings

```python
emb = oai.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding
```

This produces **1536‑dimensional** vectors compatible with both ES and FAISS.

---

# 🔎 Vector Search Backends

## **A. FAISS Search (default for Streamlit Cloud)**

Loaded as a cached resource:

```python
idx = faiss.read_index(path)
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
```

Then searched with:

```python
D, I = idx.search(v, k)
results = [meta[i] for i in I[0]]
```

Reasons FAISS is favored in deployment:

- Zero infrastructure → works on Streamlit Cloud
- Ultra-fast in-memory vector search
- Index + metadata fits in repo under `artifacts/`

---

## **B. ElasticSearch Search (optional)**

When:

```env
DEPLOY_MODE=elastic
```

and `HAS_ES=True`, ES will be used instead.

Query uses **dense_vector** + cosine scoring:

```python
es.search(
  index=cfg.elastic_index,
  knn={
    "field": "embedding",
    "query_vector": emb,
    "k": k,
    "num_candidates": 256
  },
  _source=[...]
)
```

This mode enables:

- Large datasets beyond RAM
- Scalable multi-user deployments
- Fast filtering using ES metadata fields (unit, section, etc.)

---

# 📄 Context Assembly

Chunks are formatted:

```
[1] Unit 105 • p.14: The net work done on an object equals...
```

These are injected into the prompt to enforce grounded answers.

---

# 💬 Prompt Construction

`physbot_core` uses a strict system message:

```
You are PhysBot. Answer only with the provided context.
Cite sources like [1], [2].
If unsure, say so briefly.
```

This dramatically improves:

- factual grounding
- interpretability
- student use cases

---

# 📤 Output Format

```python
return answer_text, [
    {
      "rank": 1,
      "unit": "105",
      "section": "Work-Energy Theorem",
      "content": "...",
      "source": "Unit_105.pdf",
      "chunk_index": 14
    },
    ...
]
```

Streamlit expanders use these fields to display provenance.

---

# 🧪 Local Testing

```bash
python - <<'PY'
from physbot_core.settings import AppSettings
from physbot_core.rag_utils import generate_rag_response

cfg = AppSettings(
    deploy_mode="faiss",
    openai_api_key="sk-...",
    faiss_index_path="PhysBot/artifacts/phys_demo/store.faiss",
    faiss_meta_path="PhysBot/artifacts/phys_demo/store.pkl"
)

answer, chunks = generate_rag_response("What is Newton's first law?", cfg)
print(answer)
print(chunks[:2])
PY
```

---

# 🏎 Performance Notes

- FAISS load is cached with `@st.cache_resource`
- OpenAI client is cached the same way
- Metadata list is small (e.g., 430 rows for PhysBot demo)
- Cold start on Streamlit Cloud: ~8–12 seconds

ElasticSearch mode uses persistent connections and is much heavier.

---

# 📚 When to Use Each Backend

| Backend | Pros | Cons |
|--------|------|------|
| **FAISS** | Fast, serverless, works on Cloud, reproducible | Must fit in memory |
| **ElasticSearch** | Scales to millions of chunks, filtering, hybrid search | Needs infrastructure; slower cold starts |

---

# 🛣 Roadmap for physbot_core

- Add reranker (bge-reranker / Cohere reranker)
- Optional local embedding model
- Vision encoder for figure-context retrieval
- Add prompt templates per-bot (physics vs geology)

---

# 🎯 Summary

`physbot_core` is the **shared RAG engine** for the entire project.  
It keeps the Streamlit apps simple and makes it trivial to:

- deploy PhysBot  
- add OreBot  
- add future bots (ChemBot? MathBot?)  
- swap FAISS ↔ ES without changing app code  

It is designed for clarity, modularity, and long-term maintainability.
