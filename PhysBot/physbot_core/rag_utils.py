import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# Optional ES: only used when deploy_mode == "elastic"
try:
    from elasticsearch import Elasticsearch  # type: ignore
    HAS_ES = True
except Exception:
    HAS_ES = False

import faiss  # faiss-cpu
from .settings import AppSettings

# ---- Cached resources --------------------------------------------------------

@st.cache_resource
def _get_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

@st.cache_resource(show_spinner=False)
def _load_faiss(idx_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict]]:
    idx = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

# ---- Embedding & search ------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536d

def _embed(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def _faiss_search(embedding: List[float],
                  idx: faiss.Index,
                  meta: List[Dict],
                  k: int = 5) -> List[Dict]:
    v = np.array(embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    D, I = idx.search(v, k)
    return [meta[i] for i in I[0] if i >= 0]

def _es_search(embedding: List[float], cfg: AppSettings, k: int = 5) -> List[Dict]:
    if not HAS_ES:
        raise RuntimeError("Elasticsearch client not available in this environment.")
    es = Elasticsearch(
        hosts=[cfg.elastic_host] if cfg.elastic_host else None,
        basic_auth=(cfg.elastic_user, cfg.elastic_pass) if cfg.elastic_user and cfg.elastic_pass else None,
    )
    query = {
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": k,
            "num_candidates": 256
        },
        "_source": ["text", "doc", "page", "unit", "section", "title"]
    }
    res = es.search(index=cfg.elastic_index, knn=query["knn"], _source=query["_source"])
    return [h["_source"] for h in res["hits"]["hits"]]

# ---- Prompting & formatting --------------------------------------------------

def _assemble_prompt(chunks: List[Dict], question: str) -> str:
    """Context with bracketed citations [1], [2] … and plain text answer request."""
    ctx_lines = []
    for i, c in enumerate(chunks, start=1):
        label = c.get("doc") or c.get("title") or f"Unit {c.get('unit','?')}"
        page  = c.get("page", "?")
        text  = (c.get("text") or c.get("content") or "").strip()
        ctx_lines.append(f"[{i}] {label} • p.{page}\n{text}")
    ctx = "\n\n".join(ctx_lines)

    return (
        "You are PhysBot, a careful physics tutor. Use ONLY the context to answer. "
        "Cite with bracketed numbers like [1], [2] that correspond to the context items. "
        "If unsure, say so briefly. For equations, use LaTeX math delimiters ($...$ or $$...$$) without backticks.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{ctx}\n\nANSWER:"
    )

def _append_citations(answer: str, chunks: List[Dict]) -> str:
    # Extract used [n] citations and render a small reference list
    used = sorted({m.group(0) for m in re.finditer(r"\[(\d+)\]", answer)})
    if not used:
        return answer
    refs = ["\n\n**References**"]
    for tag in used:
        idx = int(tag.strip("[]")) - 1
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            label = c.get("doc") or c.get("title") or f"Unit {c.get('unit','?')}"
            page  = c.get("page", "?")
            refs.append(f"{tag} {label} • p.{page}")
    return answer + "\n" + "\n".join(refs)

# ---- Public API --------------------------------------------------------------

def generate_rag_response(query: str, cfg: AppSettings, k: int = 5, temperature: float = 0.2) -> Tuple[str, List[Dict]]:
    """
    FAISS-first RAG (no external ES). If cfg.deploy_mode == 'elastic' and ES is available,
    use ES instead.
    Returns: (answer_markdown, source_chunks)
    """
    client = _get_openai(cfg.openai_api_key)
    q_emb = _embed(client, query)

    if cfg.deploy_mode.lower() == "elastic" and HAS_ES:
        chunks = _es_search(q_emb, cfg, k=k)
    else:
        idx, meta = _load_faiss(cfg.faiss_index_path, cfg.faiss_meta_path)
        chunks = _faiss_search(q_emb, idx, meta, k=k)

    prompt = _assemble_prompt(chunks, query)
    chat = client.chat.completions.create(
        model=os.getenv("PHYSBOT_CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": "You are a helpful, citation-focused physics tutor."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    answer = chat.choices[0].message.content.strip()
    return _append_citations(answer, chunks), chunks

# ---- Utilities ---------------------------------------------------------------

def autoconvert_to_latex(text: str) -> str:
    pattern = r'(?<!\\)(?<!\$)(?<!`)(?<!\w)([A-Za-z][A-Za-z0-9\s^*/+\-=()]+=[^=][A-Za-z0-9\s^*/+\-()^]+)(?!\w)(?!\$)'
    def repl(m):
        expr = m.group(1).strip()
        return f"\\({expr}\\)"
    return re.sub(pattern, repl, text)
