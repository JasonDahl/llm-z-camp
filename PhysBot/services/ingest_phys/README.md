# services/ingest_phys — PhysBot Ingestion Pipeline
*Multi-pass PDF → Structured JSON → Embeddings → FAISS/ES Index*

This directory contains the complete offline ingestion workflow that transforms raw curricular PDFs into the structured data used by PhysBot and future bots.

## Overview
The ingestion pipeline produces:
- Markdown (initial pass)
- Structured JSON (unit metadata, sections, text chunks, equations, figures)
- Semantic chunks for embedding
- Embeddings for ElasticSearch or FAISS

## Architecture
```
raw PDFs →
  multipass_ingest.py
    ↓
  interim markdown/figures
    ↓
  processed JSON
    ↓
  export_to_faiss.py / ES exporter
```

## Components
### multipass_ingest.py
- Converts PDFs to markdown
- Extracts metadata, equations, figures
- Creates semantic chunks
- Outputs processed JSON

### batch_ingest.py
- Orchestrates ingestion across units
- Offers flags: --units, --skip-existing, --overwrite

### cli_batch_driver.py
- CLI wrapper
- Manages logging and directory setup

### constants.py
Defines:
```
datasets/physbot/{raw, interim, processed, metadata}
```

### utils.py
- Logging
- OpenAI client management
- Chunk and equation helpers

## Output Schema
Each chunk:
```json
{
  "unit": "105",
  "section": "Work-Energy Theorem",
  "content": "...",
  "equations": ["W = ΔK"],
  "source": "Unit_105.pdf",
  "chunk_index": 8
}
```

## Embedding Exporters
### FAISS (`export_to_faiss.py`)
Produces:
```
artifacts/phys_demo/store.faiss
artifacts/phys_demo/store.pkl
```

### ElasticSearch
Optional full-scale search backend.

## Roadmap
- Schema validation
- Dataset manifests
- Config TOML per bot
- Multimodal integration

