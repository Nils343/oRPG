from __future__ import annotations

import json

import scripts.coverage_utils as coverage_utils


def test_load_coverage_latest_ignores_cwd(monkeypatch, tmp_path):
    sample_dir = tmp_path / "artifacts"
    sample_dir.mkdir()
    sample_file = sample_dir / "coverage-latest.json"
    sample_payload = {"totals": {"percent_covered_display": "99"}}
    sample_file.write_text(json.dumps(sample_payload), encoding="utf-8")

    monkeypatch.setattr(coverage_utils, "_COVERAGE_LATEST_PATH", sample_file)

    working_dir = tmp_path / "some" / "deep" / "dir"
    working_dir.mkdir(parents=True)
    monkeypatch.chdir(working_dir)

    data = coverage_utils.load_coverage_latest()

    assert data == sample_payload
    assert coverage_utils.coverage_latest_path() == sample_file
    assert coverage_utils.coverage_latest_path().is_file()
