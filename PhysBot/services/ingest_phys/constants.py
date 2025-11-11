from pathlib import Path
from physbot_core.path_utils import get_project_root  # already in your project

PROJECT_ROOT = get_project_root()  # resolves to .../llm-z-camp/PhysBot

# Dataset roots
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "physbot"
RAW_DIR       = DATASETS_ROOT / "raw"
INTERIM_DIR   = DATASETS_ROOT / "interim"
PROCESSED_DIR = DATASETS_ROOT / "processed"
META_DIR      = DATASETS_ROOT / "metadata"

# Pipeline IO
# raw inputs
CHAPTER_DIR   = RAW_DIR / "chapters"

# intermediate artifacts
MARKDOWN_DIR  = INTERIM_DIR / "markdown"
FIGURE_DIR    = INTERIM_DIR / "figures"

# processed outputs
JSON_DIR      = PROCESSED_DIR / "json"
INDEX_DIR     = PROCESSED_DIR / "index"

# logs
LOG_DIR       = META_DIR / "logs"

def ensure_directories():
    for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, META_DIR,
              CHAPTER_DIR, MARKDOWN_DIR, FIGURE_DIR, JSON_DIR, INDEX_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)
