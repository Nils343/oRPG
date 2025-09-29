#!/usr/bin/env python3
"""Run each pytest test node individually with a timeout and report slow cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

DEFAULT_TIMEOUT = 3.0
COLLECT_CMD = ["pytest", "--collect-only", "-q"]


def collect_node_ids(pytest_args: List[str]) -> List[str]:
    cmd = COLLECT_CMD + pytest_args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    nodes: List[str] = []
    for line in proc.stdout.splitlines():
        candidate = line.strip()
        if candidate and "::" in candidate and not candidate.startswith("<"):
            nodes.append(candidate)
    return nodes


def run_node(nodeid: str, timeout: float) -> Dict[str, object]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["pytest", "-q", nodeid],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        duration = time.perf_counter() - start
        status = "passed" if completed.returncode == 0 else "failed"
        return {
            "nodeid": nodeid,
            "status": status,
            "duration_s": round(duration, 3),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = exc.output or ""
        stderr = exc.stderr or ""
        return {
            "nodeid": nodeid,
            "status": "timeout",
            "duration_s": round(duration, 3),
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytest_args", nargs="*", help="Additional pytest arguments (e.g. specific files)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Timeout per test in seconds")
    parser.add_argument("--json", type=Path, help="Optional path to write detailed JSON results")
    args = parser.parse_args()

    nodes = collect_node_ids(args.pytest_args)
    if not nodes:
        print("No tests collected.", file=sys.stderr)
        sys.exit(1)

    print(f"Collected {len(nodes)} test(s). Running each with a {args.timeout:.1f}s timeout...", flush=True)

    results: List[Dict[str, object]] = []
    slow: List[Dict[str, object]] = []

    for idx, nodeid in enumerate(nodes, start=1):
        print(f"[{idx}/{len(nodes)}] {nodeid}", flush=True)
        result = run_node(nodeid, args.timeout)
        results.append(result)
        duration = result.get("duration_s")
        timed_out = result["status"] == "timeout"
        if timed_out or (isinstance(duration, (int, float)) and duration > args.timeout):
            slow.append(result)

    print("\nSummary:")
    print(f"  Total tests run: {len(results)}")
    timeouts = [r for r in results if r["status"] == "timeout"]
    failures = [r for r in results if r["status"] == "failed"]
    print(f"  Passed: {len(results) - len(timeouts) - len(failures)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Timed out (> {args.timeout:.1f}s): {len(timeouts)}")

    if slow:
        print("\nTests exceeding timeout:")
        for entry in slow:
            print(f"  {entry['nodeid']}  status={entry['status']} duration={entry['duration_s']}s")

    if args.json:
        payload = {
            "timeout": args.timeout,
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nDetailed results written to {args.json}")


if __name__ == "__main__":
    main()
