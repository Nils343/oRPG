import os
import sys
import pathlib
import importlib.util


def load_module_fresh(module_name: str, env: dict):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    spec = importlib.util.spec_from_file_location(
        module_name, pathlib.Path(__file__).resolve().parents[1] / "oRPG.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allow_anyone_env_respects_zero_and_one(monkeypatch):
    m0 = load_module_fresh("oRPG_env0", {"ALLOW_ANYONE_TO_RESOLVE": "0"})
    assert m0.ALLOW_ANYONE_TO_RESOLVE is False
    m1 = load_module_fresh("oRPG_env1", {"ALLOW_ANYONE_TO_RESOLVE": "1"})
    assert m1.ALLOW_ANYONE_TO_RESOLVE is True


def test_join_code_is_stripped_on_import(monkeypatch):
    m = load_module_fresh("oRPG_env_joincode", {"JOIN_CODE": "  open-sesame  "})
    assert m.JOIN_CODE == "open-sesame"

