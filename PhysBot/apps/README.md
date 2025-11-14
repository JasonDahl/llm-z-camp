# PhysBot Streamlit App Deployment Guide

This document describes how to deploy the **PhysBot** Retrieval-Augmented Generation (RAG) application using **Streamlit**. The app supports fully local FAISS-based retrieval as well as optional Elasticsearch mode. It is designed for zero‑infrastructure deployment on Streamlit Cloud.

---

## 🚀 Overview

The PhysBot Streamlit app provides:

- A clean user interface for physics question answering  
- RAG-based responses with contextual citations  
- Inline + block LaTeX rendering  
- Configurable backend (FAISS or ElasticSearch)  
- Automatic environment loading from `.env` or Streamlit Secrets

---

## 📁 Project Structure (App Submodule)

```
PhysBot/
└── apps/
    └── physbot_app/
        ├── app.py               # Streamlit UI
        ├── requirements.txt     # App-only environment
        ├── logos/               # Rotating logos for sidebar
        └── README_streamlit.md  # (this file)
```

The app intentionally **does not** import any ingestion code.  
All runtime logic is isolated inside **physbot_core/**.

---

## ⚙️ Runtime Architecture

```
User Query
     ↓
Streamlit App (app.py)
     ↓
physbot_core.rag_utils.generate_rag_response()
     ↓
• Embedding via OpenAI
• Retrieve top‑k context (FAISS or ES)
• Assemble RAG prompt
• Generate model answer
     ↓
App renders:
• Answer
• Inline + block LaTeX
• Source metadata
```

---

## 🔧 Configuration

### Environment Variables

PhysBot will load config from:

1. `.env` (local development)  
2. **Streamlit Cloud Secrets** (deployment)  

Supported keys:

```
OPENAI_API_KEY="sk-..."
DEPLOY_MODE="faiss"      # or "elastic"

# FAISS paths (repo-relative for deployment)
FAISS_INDEX_PATH="PhysBot/artifacts/phys_demo/store.faiss"
FAISS_META_PATH="PhysBot/artifacts/phys_demo/store.pkl"

# Optional ElasticSearch mode
ELASTICSEARCH_HOST="https://...."
ELASTIC_USER="elastic"
ELASTIC_PASS="..."
```

---

## 💻 Local Development

### 1. Install dependencies

```
cd PhysBot/apps/physbot_app
pip install -r requirements.txt
```

### 2. Ensure FAISS artifacts exist

The repository contains a small demo index:

```
PhysBot/artifacts/phys_demo/store.faiss
PhysBot/artifacts/phys_demo/store.pkl
```

### 3. Run the UI

```
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

### Step 1 — App File

Set your app’s entrypoint to:

```
PhysBot/apps/physbot_app/app.py
```

### Step 2 — Python Version

Streamlit Cloud supports FAISS on:

```
python_version = "3.11"
```

Configure this in:

`~/.streamlit/config.toml` (local)  
**OR** Streamlit Cloud → App Settings → Python Version

### Step 3 — Secrets

Add to Streamlit Cloud:

```
OPENAI_API_KEY="sk-..."
DEPLOY_MODE="faiss"
FAISS_INDEX_PATH="PhysBot/artifacts/phys_demo/store.faiss"
FAISS_META_PATH="PhysBot/artifacts/phys_demo/store.pkl"
```

### Step 4 — Requirements

Use:

```
streamlit==1.39.0
openai>=1.0.0
tiktoken
pydantic
pydantic-settings
numpy<2
faiss-cpu==1.7.4
python-dotenv
pillow
```

Avoid pinning NumPy 2.x — FAISS wheels require NumPy 1.x.

---

## 🎨 UI Features

### Sidebar
- Random rotating logo
- Collapsible source metadata
- Unit + section + excerpt preview
- Equation rendering

### Main Panel
- Text area for question input
- RAG-generated answer
- Inline LaTeX: `\( F = ma \)`
- Block LaTeX: `$$ F = ma $$`

---

## 🧪 Testing

### Test embedding
```
python - <<'PY'
from openai import OpenAI
c = OpenAI()
print(len(c.embeddings.create(model="text-embedding-3-small", input="test").data[0].embedding))
PY
```

### Test FAISS index
```
python - <<'PY'
import faiss, pickle
idx = faiss.read_index("PhysBot/artifacts/phys_demo/store.faiss")
with open("PhysBot/artifacts/phys_demo/store.pkl","rb") as f: meta = pickle.load(f)
print(idx.ntotal, len(meta))
PY
```

---

## 🔍 Troubleshooting

### ❌ FAISS import error on Streamlit Cloud
Fix: add `numpy<2` to requirements.txt.

### ❌ FAISS file not found
Fix: use **repo-relative paths** in secrets:
```
FAISS_INDEX_PATH="PhysBot/artifacts/phys_demo/store.faiss"
```

### ❌ OpenAI key not loading
Fix: Unset exported keys in WSL:
```
unset OPENAI_API_KEY
```
and rely on `.env`.

---

## 🏁 Conclusion

This Streamlit deployment is lightweight, portable, and scalable.  
Future upgrades will support:

- Per-bot configuration from TOML  
- Inline figure reinsertion  
- Local LLM inference mode  

---