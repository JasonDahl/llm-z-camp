# llm-z-camp  
*A sandbox and development workspace for LLM and RAG projects.*

This repo contains a variety of experimental and production-grade AI projects created during my long-term LLM/RAG learning journey.  
The flagship project currently hosted here is **PhysBot** — a physics question-answering system built with a full ingestion pipeline and FAISS-based RAG architecture.

---

## 🔭 Featured Project: **PhysBot**

**PhysBot** is a retrieval-augmented physics tutor trained on the USCGA Physics I curriculum.  
It uses:

- Multi-pass PDF → Markdown → structured JSON ingestion  
- OpenAI embeddings  
- FAISS for fast vector search  
- Streamlit UI for interactive Q&A  
- Full provenance and bracketed citations  

🚀 **Try the live app:**  
https://physbot-demo.streamlit.app/

📂 **Project code:**  
[`PhysBot/`](PhysBot/)

---

## 📁 Repository Layout (High-Level)

llm-z-camp/
│
├── PhysBot/ # Production-ready physics RAG system
│ ├── apps/ # Streamlit apps (PhysBot live app)
│ ├── physbot_core/ # Shared RAG engine (FAISS + OpenAI)
│ ├── services/ # Multi-pass ingestion pipelines
│ ├── datasets/ # Raw/interim/processed corpus
│ └── artifacts/ # Versioned FAISS snapshots
│
├── intro01/ # Basic RAG code from LLM Zoomcamp by Data Talks Club
├── README.md # You are here
└── requirements.txt  # Copy of Physbot requirements


---

## 🛠 What’s Coming Next

- **OreBot** — RAG for ore-deposit geology  
- Custom LLM fine-tuning and evaluation tools
- vision encoding and multi-modal RAG

---

## 📬 Contact

If you have questions or suggestions, feel free reach out.

---