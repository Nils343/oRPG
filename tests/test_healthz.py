import sys, pathlib, importlib.util
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_health_endpoint_reports_status(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {player.id: player}
    g.turn_number = 2
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["players"] == 1
    assert data["turn"] == 2


def test_server_starts_with_default_env_vars(monkeypatch):
    for var in ("OLLAMA_HOST", "OLLAMA_MODEL", "BIND_HOST", "BIND_PORT"):
        monkeypatch.delenv(var, raising=False)

    spec = importlib.util.spec_from_file_location(
        "oRPG_default", pathlib.Path(__file__).resolve().parents[1] / "oRPG.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.OLLAMA_HOST == "http://127.0.0.1:11434"
    assert module.OLLAMA_MODEL == "gpt-oss:20b"
    assert module.BIND_HOST == "0.0.0.0"
    assert module.BIND_PORT == 8000

    client = TestClient(module.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
