from pathlib import Path

# Define project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define all standard directories
CHAPTER_DIR = PROJECT_ROOT / "data" / "chapters" / "units"
JSON_DIR = PROJECT_ROOT / "data" / "json"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
FIGURE_DIR = PROJECT_ROOT / "data" / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

def ensure_directories():
    for path in [CHAPTER_DIR, JSON_DIR, MARKDOWN_DIR, FIGURE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)