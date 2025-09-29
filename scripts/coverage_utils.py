"""Helpers for working with the coverage JSON artifact across the repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

try:  # allow execution both as ``scripts.coverage_utils`` and plain module
    from scripts.path_utils import resolve_repo_path
except ModuleNotFoundError:  # pragma: no cover - fallback when running from scripts/
    from path_utils import resolve_repo_path

_COVERAGE_LATEST_PATH: Path = resolve_repo_path("coverage-latest.json")


def coverage_latest_path() -> Path:
    """Return the absolute path to ``coverage-latest.json`` inside the repo."""

    return _COVERAGE_LATEST_PATH


def load_coverage_latest() -> Dict[str, Any]:
    """Load and return the parsed coverage JSON data.

    Raises:
        FileNotFoundError: If the coverage artifact is missing.
        json.JSONDecodeError: If the file contents are not valid JSON.
    """

    try:
        with coverage_latest_path().open(encoding="utf-8") as handle:
            data = json.load(handle)
            return cast(Dict[str, Any], data)
    except FileNotFoundError as exc:  # pragma: no cover - exercised in tests via explicit assertion
        msg = f"coverage-latest.json not found at {coverage_latest_path()}"
        raise FileNotFoundError(msg) from exc


if __name__ == "__main__":  # pragma: no cover - manual utility
    data = load_coverage_latest()
    totals = data.get("totals", {})
    percent = totals.get("percent_covered_display")
    covered = totals.get("covered_lines")
    statements = totals.get("num_statements")
    print("Coverage totals:")
    print(f"  Percent covered: {percent if percent is not None else 'unknown'}")
    if covered is not None and statements is not None:
        print(f"  Lines covered: {covered}/{statements}")
