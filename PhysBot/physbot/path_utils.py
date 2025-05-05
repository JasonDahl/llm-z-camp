from pathlib import Path

def get_project_root():
    """
    Returns the project root path based on the current context.
    - In scripts: uses __file__
    - In Jupyter notebooks: falls back to current working directory's parent.
    """
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd().resolve().parent