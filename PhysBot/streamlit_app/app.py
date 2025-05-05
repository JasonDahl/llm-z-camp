# app.py
import streamlit as st
import sys
from pathlib import Path
import re
import random
from PIL import Image
from streamlit import markdown, latex


# Add the project root to sys.path so you can import physbot.*
sys.path.append(str(Path(__file__).resolve().parents[1]))
from physbot.path_utils import get_project_root
from dotenv import load_dotenv
load_dotenv(get_project_root() / ".env")

from physbot.rag_utils import generate_rag_response, autoconvert_to_latex

def render_answer(answer: str):
    st.markdown("## 🤖 Answer")

    # Split answer into parts based on $$...$$ for block LaTeX
    parts = re.split(r"(\$\$.*?\$\$)", answer, flags=re.DOTALL)

    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            # Strip surrounding $$ and render as block LaTeX
            st.latex(part.strip("$"))
        else:
            # Support inline LaTeX: \( ... \)
            # NOTE: Streamlit renders \( ... \) as LaTeX with unsafe_allow_html
            html_with_inline = re.sub(r"\\\((.*?)\\\)", r"\\(\1\\)", part)
            st.markdown(html_with_inline.strip(), unsafe_allow_html=True)

st.set_page_config(page_title="PhysBot – Ask a Physics Question", layout="wide")

# Get project root and resolve the logos folder
project_root = get_project_root()
logo_dir = project_root / "streamlit_app" / "logos"
webp_files = list(logo_dir.glob("*.webp"))

   
# Sidebar
# Randomly select and display logo in sidebar
if webp_files:
    logo_path = random.choice(webp_files)
    logo = Image.open(logo_path)
    st.sidebar.image(logo, caption="PhysBot: Your Helpful Physics Chatbot", use_container_width=True)
st.sidebar.title("📘 PhysBot v0.5")
st.sidebar.markdown(
    """
    **Ask your physics question** and receive an answer with cited textbook sections.
    """
)

# Main UI
st.title("Physics Question Answering with RAG")
st.markdown("Enter a question related to any unit in the USCGA Physics 1 curriculum:")

query = st.text_area(
    "Enter your physics question.  Hit ctrl-Enter to submit.",
    height=150,             # Height in pixels
    max_chars=500,          # Max character input
    placeholder="e.g., What is Newton's second law?",
)

if query:
    with st.spinner("Thinking..."):
        answer, top_chunks = generate_rag_response(query)

        # Render answer
        render_answer(answer)

        st.markdown("---")

        # Sidebar: Collapsible source list
        with st.sidebar:
            st.markdown("## 📚 Sources & Sections")
            for i, chunk in enumerate(top_chunks, 1):
                with st.expander(f"Source {i}"):
                    st.write(f"**Unit:** {chunk.get('unit', 'Unknown')}")
                    st.write(f"**Section:** {chunk.get('section', 'Unknown')}")
                    st.write(f"**Source file:** {chunk.get('source', 'Unknown')}")

                    excerpt = chunk.get("content", "")[:300].strip()
                    st.markdown(f"> {excerpt}...")

                    # Attempt to extract and render LaTeX equations if present
                    equations = chunk.get("equations", [])
                    if equations:
                        st.markdown("**Equations:**")
                        for eq in equations:
                            st.latex(eq)
# Footer
st.markdown("---")
st.markdown(
    "<sub>Powered by OpenAI, Elasticsearch, and Streamlit.</sub>",
    unsafe_allow_html=True
)
