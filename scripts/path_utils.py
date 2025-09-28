"""Utilities for locating project paths regardless of current working directory."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Return the repository root directory as an absolute Path."""

    return _PROJECT_ROOT


def resolve_repo_path(*parts: str) -> Path:
    """Build an absolute path inside the repository from the given path parts."""

    return _PROJECT_ROOT.joinpath(*parts)
