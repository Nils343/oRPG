import io
import subprocess
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import analyze_slow_tests, run_tests_with_timeout


def _write_sample_test(tmp_path: Path, name: str = "test_sample.py") -> Path:
    source = textwrap.dedent(
        """
        import asyncio


        async def test_async_case():
            await asyncio.sleep(0)
            for _ in range(5):
                await asyncio.sleep(0)
        """
    ).strip()
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_analyze_file_extracts_metrics(tmp_path: Path) -> None:
    test_file = _write_sample_test(tmp_path)
    metrics = analyze_slow_tests.analyze_file(test_file)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "test_async_case"
    assert metric.await_count == 2
    assert metric.loop_count == 1
    assert metric.score > 0


def test_analyze_main_prints_summary(tmp_path: Path) -> None:
    test_file = _write_sample_test(tmp_path)
    args = SimpleNamespace(paths=[test_file], top=1)
    with (
        mock.patch.object(analyze_slow_tests.argparse.ArgumentParser, "parse_args", return_value=args),
        redirect_stdout(io.StringIO()) as buffer,
    ):
        analyze_slow_tests.main()
    output = buffer.getvalue()
    assert "Analyzed 1 tests" in output
    assert str(test_file) in output


def test_collect_node_ids_parses_pytest_stdout() -> None:
    fake_output = "tests/test_example.py::test_one\n<Package 'tests'>\n"
    completed = mock.Mock(stdout=fake_output, returncode=0)
    with mock.patch("scripts.run_tests_with_timeout.subprocess.run", return_value=completed) as run_mock:
        nodes = run_tests_with_timeout.collect_node_ids(["-k", "sample"])
    run_mock.assert_called_once()
    assert nodes == ["tests/test_example.py::test_one"]


def test_run_node_handles_success_and_failure() -> None:
    success = mock.Mock(returncode=0, stdout="ok", stderr="")
    with mock.patch("scripts.run_tests_with_timeout.subprocess.run", return_value=success):
        result = run_tests_with_timeout.run_node("tests/test_example.py::test_one", timeout=0.1)
    assert result["status"] == "passed"
    assert result["returncode"] == 0

    timeout_error = subprocess.TimeoutExpired(cmd=["pytest"], timeout=0.1)
    with mock.patch("scripts.run_tests_with_timeout.subprocess.run", side_effect=timeout_error):
        result = run_tests_with_timeout.run_node("tests/test_example.py::test_two", timeout=0.1)
    assert result["status"] == "timeout"
    assert result["returncode"] is None


def test_run_tests_main_reports_summary(tmp_path: Path) -> None:
    args = SimpleNamespace(pytest_args=["tests"], timeout=0.5, json=None)
    nodes = ["tests/test_example.py::test_one", "tests/test_example.py::test_two"]
    results = [
        {"nodeid": nodes[0], "status": "passed", "duration_s": 0.01, "returncode": 0, "stdout": "", "stderr": ""},
        {"nodeid": nodes[1], "status": "timeout", "duration_s": 1.2, "returncode": None, "stdout": "", "stderr": ""},
    ]

    with (
        mock.patch.object(run_tests_with_timeout.argparse.ArgumentParser, "parse_args", return_value=args),
        mock.patch("scripts.run_tests_with_timeout.collect_node_ids", return_value=nodes),
        mock.patch("scripts.run_tests_with_timeout.run_node", side_effect=results),
        redirect_stdout(io.StringIO()) as buffer,
    ):
        run_tests_with_timeout.main()

    output = buffer.getvalue()
    assert "Collected 2 test(s)" in output
    assert "Timed out" in output
    assert nodes[1] in output
