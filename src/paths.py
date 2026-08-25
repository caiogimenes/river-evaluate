from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root (not the process CWD)."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()
