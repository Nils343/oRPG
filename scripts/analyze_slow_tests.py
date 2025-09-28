#!/usr/bin/env python3
"""Heuristic analyzer to flag potentially slow tests without executing them."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Heuristic weights tuned for this project. Adjust as needed after inspecting output.
HEAVY_CALL_WEIGHTS: Dict[str, float] = {
    "rpg.resolve_turn": 6.0,
    "rpg.save_settings": 4.0,
    "rpg.load_settings": 4.0,
    "rpg.reset_session_progress": 4.5,
    "rpg.create_portrait": 4.5,
    "rpg.create_image": 5.0,
    "rpg.animate_scene": 5.0,
    "rpg.schedule_auto_scene_image": 4.0,
    "rpg.schedule_auto_scene_video": 4.0,
    "rpg.schedule_auto_tts": 4.0,
    "rpg.request_turn_payload": 6.0,
    "rpg.request_summary_payload": 5.0,
    "rpg.gemini_generate_image": 6.0,
    "rpg.generate_scene_video": 6.0,
    "rpg.gemini_list_models": 4.0,
    "TestClient": 5.0,
    "fastapi.testclient.TestClient": 5.0,
    "asyncio.sleep": 1.5,
    "time.sleep": 2.0,
    "pathlib.Path.write_text": 2.0,
    "pathlib.Path.read_text": 2.0,
    "json.dump": 2.0,
    "json.loads": 1.0,
    "tempfile.TemporaryDirectory": 3.0,
    "tempfile.NamedTemporaryFile": 3.0,
    "requests.get": 5.0,
    "requests.post": 5.0,
    "httpx.AsyncClient": 5.0,
    "httpx.Client": 4.0,
}

PATCH_TARGET_WEIGHTS: Dict[str, float] = {
    name: weight * 0.6 for name, weight in HEAVY_CALL_WEIGHTS.items()
}

LOOP_ITERATION_THRESHOLD = 50
LARGE_LITERAL_THRESHOLD = 1000


@dataclass
class TestMetrics:
    name: str
    path: Path
    lineno: int
    end_lineno: int
    async_def: bool
    sloc: int = 0
    await_count: int = 0
    loop_count: int = 0
    large_iterations: int = 0
    heavy_calls: Dict[str, int] = field(default_factory=dict)
    patch_hits: Dict[str, int] = field(default_factory=dict)
    io_calls: Dict[str, int] = field(default_factory=dict)
    large_literals: int = 0
    score: float = 0.0

    def compute_score(self) -> None:
        score = 0.0
        score += 0.05 * self.sloc
        if self.async_def:
            score += 2.5
        score += 1.5 * self.await_count
        score += 1.5 * self.loop_count
        score += 0.01 * self.large_iterations
        score += 0.5 * self.large_literals
        for name, count in self.heavy_calls.items():
            weight = HEAVY_CALL_WEIGHTS.get(name, 2.0)
            score += weight * count
        for name, count in self.patch_hits.items():
            weight = PATCH_TARGET_WEIGHTS.get(name, 1.0)
            score += weight * count
        for _name, count in self.io_calls.items():
            score += 1.0 * count
        self.score = round(score, 2)


class FunctionFeatureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.await_count = 0
        self.loop_count = 0
        self.large_iterations = 0
        self.heavy_calls: Dict[str, int] = {}
        self.io_calls: Dict[str, int] = {}
        self.patch_hits: Dict[str, int] = {}
        self.large_literals = 0

    def visit_Await(self, node: ast.Await) -> None:
        self.await_count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_count += 1
        self.large_iterations += _estimate_loop_iterations(node.iter)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loop_count += 1
        self.large_iterations += LOOP_ITERATION_THRESHOLD
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loop_count += 1
        self.large_iterations += _estimate_loop_iterations(node.iter)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _extract_call_name(node.func)
        if name:
            if name in HEAVY_CALL_WEIGHTS:
                self.heavy_calls[name] = self.heavy_calls.get(name, 0) + 1
            if name in {"pathlib.Path.write_text", "pathlib.Path.read_text", "open"}:
                self.io_calls[name] = self.io_calls.get(name, 0) + 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._handle_with_items(node.items)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._handle_with_items(node.items)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)) and abs(node.value) >= LARGE_LITERAL_THRESHOLD:
            self.large_literals += 1
        self.generic_visit(node)

    def _handle_with_items(self, items: Iterable[ast.withitem]) -> None:
        for item in items:
            call_name, target_name = _extract_patch_target(item.context_expr)
            if call_name == "mock.patch" and target_name:
                weight_key = target_name if target_name in PATCH_TARGET_WEIGHTS else None
                if weight_key:
                    self.patch_hits[weight_key] = self.patch_hits.get(weight_key, 0) + 1


def _extract_call_name(func: ast.AST) -> Optional[str]:
    parts: List[str] = []
    current = func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif isinstance(current, ast.Call):
        # Something like foo().bar(), skip
        return None
    else:
        return None
    return ".".join(reversed(parts))


def _extract_patch_target(expr: ast.AST) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(expr, ast.Call):
        return (None, None)
    call_name = _extract_call_name(expr.func)
    target_name: Optional[str] = None
    if expr.args:
        first = expr.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            target_name = first.value
    return (call_name, target_name)


def _estimate_loop_iterations(iter_node: ast.AST) -> int:
    if isinstance(iter_node, ast.Call):
        callee = _extract_call_name(iter_node.func)
        if callee == "range" and iter_node.args:
            try:
                args = [ast.literal_eval(arg) for arg in iter_node.args]
            except Exception:
                return LOOP_ITERATION_THRESHOLD
            if len(args) == 1 and isinstance(args[0], int):
                return args[0]
            if len(args) >= 2 and all(isinstance(val, int) for val in args[:2]):
                start, stop = args[0], args[1]
                step = args[2] if len(args) >= 3 and isinstance(args[2], int) else 1
                if step == 0:
                    return LOOP_ITERATION_THRESHOLD
                return abs((stop - start) // step)
        return LOOP_ITERATION_THRESHOLD
    return 0


def _is_test_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = _extract_call_name(base)
        if not base_name and isinstance(base, ast.Name):
            base_name = base.id
        if base_name and ("TestCase" in base_name or base_name.startswith("Test")):
            return True
    return False


def _iter_tests(module: ast.Module) -> Iterable[Tuple[Optional[str], ast.AST]]:
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            yield (None, node)
        elif isinstance(node, ast.ClassDef) and _is_test_class(node):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    yield (node.name, child)


def analyze_file(path: Path) -> List[TestMetrics]:
    source = path.read_text()
    module = ast.parse(source)
    metrics: List[TestMetrics] = []
    for class_name, func in _iter_tests(module):
        assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
        name = func.name if not class_name else f"{class_name}.{func.name}"
        visitor = FunctionFeatureVisitor()
        visitor.visit(func)
        sloc = (func.end_lineno or func.lineno) - func.lineno + 1
        metrics.append(
            TestMetrics(
                name=name,
                path=path,
                lineno=func.lineno,
                end_lineno=func.end_lineno or func.lineno,
                async_def=isinstance(func, ast.AsyncFunctionDef),
                sloc=sloc,
                await_count=visitor.await_count,
                loop_count=visitor.loop_count,
                large_iterations=visitor.large_iterations,
                heavy_calls=visitor.heavy_calls,
                patch_hits=visitor.patch_hits,
                io_calls=visitor.io_calls,
                large_literals=visitor.large_literals,
            )
        )
    for metric in metrics:
        metric.compute_score()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Test file(s) to analyze")
    parser.add_argument("--top", type=int, default=15, help="Number of highest scoring tests to display")
    args = parser.parse_args()

    all_metrics: List[TestMetrics] = []
    for path in args.paths:
        all_metrics.extend(analyze_file(path))

    all_metrics.sort(key=lambda m: m.score, reverse=True)
    top_n = args.top if args.top > 0 else len(all_metrics)

    print(f"Analyzed {len(all_metrics)} tests across {len(args.paths)} file(s).\n")
    print(f"Top {top_n} potentially long-running tests (higher score => likely slower):")
    for metric in all_metrics[:top_n]:
        heavy_summary = ", ".join(f"{name}:{count}" for name, count in sorted(metric.heavy_calls.items())) or "-"
        patch_summary = ", ".join(f"{name}:{count}" for name, count in sorted(metric.patch_hits.items())) or "-"
        io_summary = ", ".join(f"{name}:{count}" for name, count in sorted(metric.io_calls.items())) or "-"
        print(
            f"  {metric.score:5.2f}  {metric.name}  ({metric.path}:{metric.lineno})\n"
            f"          sloc={metric.sloc}, awaits={metric.await_count}, loops={metric.loop_count}, "
            f"large_iters={metric.large_iterations}, literals>={LARGE_LITERAL_THRESHOLD}={metric.large_literals}\n"
            f"          heavy_calls={heavy_summary}\n"
            f"          patch_hits={patch_summary}\n"
            f"          io_calls={io_summary}\n"
        )


if __name__ == "__main__":
    main()
