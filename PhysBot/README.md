# PhysBots Monorepo  
*A modular Retrieval-Augmented Generation (RAG) system for physics and geology education.*

This repository hosts a **multi-bot RAG platform** built around physics and geology content.  
It is structured like a production system: ingestion pipelines, shared inference core, and separate deployed apps.

Currently included:

- **PhysBot** — physics tutoring bot trained on the USCGA Physics I curriculum  
- **OreBot** (coming soon) — ore-deposit geology Q&A  
- **physbot_core** — shared RAG inference engine (FAISS + OpenAI)  
- **services/** — ingestion pipelines (multi-pass PDF → JSON → embeddings → FAISS/ES)  
- **apps/** — Streamlit Cloud apps for each bot  
- **artifacts/** — versioned FAISS snapshots for zero-infrastructure deployment  

---

## 🧠 Motivation

Educational RAG systems usually fall into one of two categories:

1. **Toy demos** — minimal parsing, no provenance, shallow answers  
2. **Full production stacks** — ElasticSearch vector DBs, multi-stage ingestion, complex ops

This project aims for a **middle path**:

- Serious **document processing** (figures, sections, math), multi-pass chunking, semantic structure  
- Clean **provenance** and contextual citations  
- App deployments with **no infrastructure dependency** (FAISS-only mode)  
- Architecture that scales: swap FAISS for ElasticSearch without touching app code  

The result is something between a **research platform**, a **teaching tool**, and a **lightweight RAG product**.

---

## 🏗 Repository Structure
```
llm-z-camp/
│
├── PhysBot/
│   ├── apps/
│   │   └── physbot_app/        # Streamlit UI
│   ├── physbot_core/           # Shared RAG logic (FAISS, prompts, settings)
│   ├── artifacts/
│   │   └── phys_demo/          # Small FAISS index published with the repo
│   ├── services/
│   │   └── ingest_phys/        # Multi-pass parsing & embedding pipeline
│   ├── datasets/               # Raw/interim/processed per-bot corpora
│   └── ...
│
├── OreBot/                     # (future parallel structure)
└── README.md
```

---

## 🚀 Quickstart (PhysBot)

### **Run locally**
```bash
cd PhysBot
pip install -r apps/physbot_app/requirements.txt
streamlit run apps/physbot_app/app.py
```

### **Deploy on Streamlit Cloud**

App file: `PhysBot/apps/physbot_app/app.py`

Secrets:
```bash
OPENAI_API_KEY="sk-...."
FAISS_INDEX_PATH="PhysBot/artifacts/phys_demo/store.faiss"
FAISS_META_PATH="PhysBot/artifacts/phys_demo/store.pkl"
DEPLOY_MODE="faiss"
```

---

## 📚 Core Features

### ✔ Multi-pass ingestion
- PDF → markdown → structured JSON  
- Figure extraction + equation reintegration  
- Clean semantic chunking for embeddings  
- Standardized metadata: unit, section, chunk_index, source, etc.

### ✔ Swappable vector backend
- FAISS by default (fast, zero infrastructure)  
- Optional ElasticSearch mode  

### ✔ Strong provenance & citations
- Context formatting ensures models cite sources via bracketed references  

### ✔ App/runtime clarity
- Apps never import ingestion code  
- Clean separation of responsibilities

---

## 🧭 Roadmap

- Add full OreBot ingestion  
- Add config TOML for each bot  
- CI: ensure apps don’t import ingestion modules  
- Optional local LLM mode for inference  