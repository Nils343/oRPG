import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_resolve_returns_500_and_clears_resolving_on_exception(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)

    async def boom():
        raise RuntimeError("LLM failed")

    monkeypatch.setattr(oRPG, "do_resolution", boom)

    client = TestClient(oRPG.app, raise_server_exceptions=False)
    r = client.post("/resolve", json={"player_id": host.id})
    assert r.status_code == 500
    assert g.resolving is False


def test_join_propagates_initial_scene_errors_as_500(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    async def boom():
        raise RuntimeError("LLM down")

    # Ensure we attempt initial scene
    monkeypatch.setattr(oRPG, "ensure_initial_scene", boom)
    async def fake_choose_class(background: str, max_attempts: int = 5) -> str:
        return "Shadow Blade"
    monkeypatch.setattr(oRPG, "choose_class_with_ollama", fake_choose_class)

    client = TestClient(oRPG.app, raise_server_exceptions=False)
    r = client.post("/join", json={"name": "Alice", "background": "bg"})
    assert r.status_code == 500

