from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import sys
import re
import streamlit as st

# Add parent of 'physbot' to the path
sys.path.append(os.path.abspath(".."))

from physbot.path_utils import get_project_root

# Load environment variables from the project root
# For Jupyter: assume this notebook is in /PhysBot/notebooks
notebook_dir = Path.cwd()
project_root = get_project_root()
load_dotenv(dotenv_path=project_root / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
es = Elasticsearch("http://localhost:9200")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Settings
ES_INDEX = "physbot_units"
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 5

def get_query_embedding(query: str):
    """Embed the user query using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding

def semantic_search(query_embedding, top_k=TOP_K):
    """Use Elasticsearch to retrieve top-k similar chunks."""
    search_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }
    response = es.search(index=ES_INDEX, query=search_query["query"])
    return [hit["_source"] for hit in response["hits"]["hits"]]

def assemble_prompt(chunks, question):
    """Construct an improved prompt using context and metadata, guiding LaTeX formatting."""
    
    context_blocks = []
    for i, chunk in enumerate(chunks):
        unit = chunk.get("unit", "Unknown")
        section = chunk.get("section", "Unknown")
        content = chunk.get("content", "").strip()
        source_tag = f"[Source {i+1}: Unit {unit}, Section {section}]"
        context_blocks.append(f"{source_tag}\n{content}")
    
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful physics tutor assistant.

    Use the context passages below to answer the student's question. 
    - Cite sources like [Source 1], [Source 2], etc.
    - Format any and all equations using block LaTeX like $$F = ma$$.
    - Do not use Markdown or backticks for equations.
    - Return the response as clean, plain text.

    Context:
    {context_text}

    Question: {question}

    Answer:"""

    return prompt


def append_citation_details(answer_text, chunks):
    """
    Replace [Source X] references in the answer with actual source info,
    and append a readable reference block at the end.
    """
    source_map = {}
    for i, chunk in enumerate(chunks):
        source_map[f"Source {i+1}"] = f"Unit {chunk.get('unit', 'Unknown')} – {chunk.get('section', 'Unknown')}"

    # Append a references section
    used_sources = sorted(set(re.findall(r"\[Source \d+\]", answer_text)))
    references = "\n\n**References:**\n"
    for src in used_sources:
        ref = source_map.get(src.replace("[", "").replace("]", ""), "Unknown source")
        references += f"{src}: {ref}\n"

    return answer_text + references

def generate_rag_response(query: str):
    """Perform semantic search and return a response from OpenAI."""
    embedding = get_query_embedding(query)
    top_chunks = semantic_search(embedding)
    prompt = assemble_prompt(top_chunks, query)
    
    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    response = completion.choices[0].message.content.strip()
    
    answer_with_citations = append_citation_details(response, top_chunks)
    return answer_with_citations, top_chunks

def autoconvert_to_latex(text: str) -> str:
    # Match basic equations not already in LaTeX or code
    pattern = r'(?<!\\)(?<!\$)(?<!`)(?<!\w)([A-Za-z][A-Za-z0-9\s^*/+\-=()]+=[^=][A-Za-z0-9\s^*/+\-()^]+)(?!\w)(?!\$)'
    def repl(match):
        expr = match.group(1).strip()
        return f"\\({expr}\\)"
    return re.sub(pattern, repl, text)