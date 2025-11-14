# apps/physbot_app/app.py
import sys
import os
from pathlib import Path

# Ensure project root (the folder that contains physbot_core/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import re
import random
from PIL import Image
from dotenv import load_dotenv

from physbot_core.path_utils import get_project_root
from physbot_core.settings import AppSettings
from physbot_core.rag_utils import generate_rag_response, autoconvert_to_latex

# ---------------------- Page & env ----------------------
st.set_page_config(page_title="PhysBot – Ask a Physics Question", page_icon="⚛️", layout="wide")

# Load local .env if present (Streamlit Cloud will use Secrets instead)
env_path = PROJECT_ROOT.parent / ".env"
load_dotenv(env_path , override=True)


# App config (env-first, with FAISS defaults for demo)
cfg = AppSettings(
    app_name="PhysBot",
    deploy_mode=os.getenv("DEPLOY_MODE", "faiss"),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    faiss_index_path=os.getenv("FAISS_INDEX_PATH", "PhysBot/artifacts/phys_demo/store.faiss"),
    faiss_meta_path=os.getenv("FAISS_META_PATH", "PhysBot/artifacts/phys_demo/store.pkl"),
    elastic_index=os.getenv("ELASTIC_INDEX", "physbot_units"),
    elastic_host=os.getenv("ELASTICSEARCH_HOST"),
    elastic_user=os.getenv("ELASTIC_USER"),
    elastic_pass=os.getenv("ELASTIC_PASS"),
)

# ---------------------- Branding ----------------------
# logos live next to this file: apps/physbot_app/logos/*.webp
logo_dir = Path(__file__).parent / "logos"
webp_files = sorted(logo_dir.glob("*.webp")) if logo_dir.exists() else []

if webp_files:
    logo_path = random.choice(webp_files)
    st.sidebar.image(
        Image.open(logo_path),
        caption="PhysBot: Your Helpful Physics Chatbot",
        use_container_width=True,
    )

st.sidebar.title("📘 PhysBot")
st.sidebar.markdown("**Ask your physics question** and get an answer with bracketed citations.")

# ---------------------- Rendering helpers ----------------------
def render_answer(answer: str):
    st.markdown("## 🤖 Answer")
    # Split on $$...$$ (block math). Render inline \( ... \) via markdown.
    parts = re.split(r"(\$\$.*?\$\$)", answer, flags=re.DOTALL)
    for part in parts:
        if part.startswith("$$") and part.endswith(("$$",)):
            # strip the outer $$ and render as block LaTeX
            st.latex(part[2:-2].strip())
        else:
            # Keep inline LaTeX: \( ... \)
            html_with_inline = re.sub(r"\\\((.*?)\\\)", r"\\(\1\\)", part)
            st.markdown(html_with_inline.strip(), unsafe_allow_html=True)

# ---------------------- Main UI ----------------------
st.title("Physics Question Answering with RAG")
st.markdown("Enter a question related to the USCGA Physics I curriculum (citations like [1], [2] will appear).")

query = st.text_area(
    "Enter your physics question.  Hit Ctrl+Enter to submit.",
    height=150,
    max_chars=500,
    placeholder="e.g., Explain the work–energy theorem.",
)

if query:
    with st.spinner("Thinking..."):
        answer, top_chunks = generate_rag_response(query, cfg)

    # Answer
    render_answer(answer)
    st.markdown("---")

    # Provenance (sidebar expander pattern)
    with st.sidebar:
        st.markdown("## 📚 Sources & Sections")
        for i, chunk in enumerate(top_chunks, 1):
            label = chunk.get("doc") or chunk.get("title") or f"Unit {chunk.get('unit','?')}"
            page = chunk.get("page", "?")
            with st.expander(f"[{i}] {label} • p.{page}"):
                # Show common fields across FAISS/ES paths
                st.write(f"**Unit:** {chunk.get('unit', '—')}")
                st.write(f"**Section:** {chunk.get('section', '—')}")
                st.write(f"**Doc:** {label}")
                excerpt = (chunk.get("text") or chunk.get("content") or "").strip()
                if excerpt:
                    st.markdown(f"> {excerpt[:500]}{'…' if len(excerpt) > 500 else ''}")

# Footer
st.markdown("---")
st.markdown("<sub>Powered by OpenAI and FAISS. ES optional.</sub>", unsafe_allow_html=True)
