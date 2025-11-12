# physbot_core/settings.py
from pydantic import BaseSettings


class AppSettings(BaseSettings):
    app_name: str = "PhysBot"
    deploy_mode: str = "faiss" # "faiss" | "elastic"
    openai_api_key: str = ""
    # FAISS
    faiss_index_path: str = "artifacts/phys_demo/store.faiss"
    faiss_meta_path: str = "artifacts/phys_demo/store.pkl"
    # Elasticsearch (optional)
    elastic_index: str = "physbot_units"
    elastic_host: str | None = None
    elastic_user: str | None = None
    elastic_pass: str | None = None


    class Config:
        env_prefix = ""
        case_sensitive = False