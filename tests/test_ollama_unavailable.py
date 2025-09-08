import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_join_returns_503_with_clear_message_when_choose_class_unavailable(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    async def boom(background: str, max_attempts: int = 5):
        raise oRPG.OllamaUnavailable("cannot connect to service")

    monkeypatch.setattr(oRPG, "choose_class_with_ollama", boom)

    client = TestClient(oRPG.app, raise_server_exceptions=False)
    r = client.post("/join", json={"name": "Alice", "background": "bg"})
    assert r.status_code == 503
    body = r.text
    assert "AI service unavailable" in body
    assert "Host:" in body
    assert "Hint:" in body
    # Ensure no player was created
    assert g.players == {}
    assert g.host_id is None


def test_join_rolls_back_player_if_initial_scene_fails(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    async def fake_choose(background: str, max_attempts: int = 5) -> str:
        return "Shadow Blade"

    async def fake_sheet(bg: str, cls: str):
        return ["Sneak", "Backstab"], ["Dagger"]

    async def fail_initial():
        raise oRPG.OllamaUnavailable("down")

    monkeypatch.setattr(oRPG, "choose_class_with_ollama", fake_choose)
    monkeypatch.setattr(oRPG, "generate_sheet_with_ollama", fake_sheet)
    monkeypatch.setattr(oRPG, "ensure_initial_scene", fail_initial)

    client = TestClient(oRPG.app, raise_server_exceptions=False)
    r = client.post("/join", json={"name": "Alice", "background": "bg"})
    assert r.status_code == 503
    # Player should be rolled back (no ghost player or host)
    assert g.players == {}
    assert g.host_id is None
    assert g.turn_number == 0
    assert g.current_scenario is None


def test_resolve_returns_503_and_clears_resolving_on_ollama_unavailable(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", [])
    host.token = "tok-host"
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"
    monkeypatch.setattr(oRPG, "GAME", g)

    async def boom():
        raise oRPG.OllamaUnavailable("LLM not reachable")

    monkeypatch.setattr(oRPG, "do_resolution", boom)

    client = TestClient(oRPG.app, raise_server_exceptions=False)
    r = client.post("/resolve", json={"player_id": host.id, "player_token": host.token})
    assert r.status_code == 503
    assert g.resolving is False
    assert "AI service unavailable" in r.text

