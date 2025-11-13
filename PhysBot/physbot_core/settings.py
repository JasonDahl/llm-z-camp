# physbot_core/settings.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppSettings:
    # High-level
    app_name: str = "PhysBot"
    deploy_mode: str = "faiss"  # "faiss" | "elastic"

    # OpenAI
    openai_api_key: str = ""

    # FAISS (default demo paths)
    faiss_index_path: str = "artifacts/phys_demo/store.faiss"
    faiss_meta_path: str = "artifacts/phys_demo/store.pkl"

    # Optional Elasticsearch config (for future ES mode)
    elastic_index: str = "physbot_units"
    elastic_host: Optional[str] = None
    elastic_user: Optional[str] = None
    elastic_pass: Optional[str] = None
