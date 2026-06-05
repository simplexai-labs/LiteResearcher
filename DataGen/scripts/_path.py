"""Helpers for running scripts directly from the repository checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def add_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    for path in (str(repo_root), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root
