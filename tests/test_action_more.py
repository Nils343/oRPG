import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_post_action_missing_player_id_returns_400(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    client = TestClient(oRPG.app)
    resp = client.post("/action", json={"text": "hi"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid player."


def test_post_action_none_text_clears_action(monkeypatch):
    g = oRPG.Game()
    p = oRPG.Player("Alice", "bg", 1.0, [])
    g.players = {p.id: p}
    g.current_actions[p.id] = "something"
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.post("/action", json={"player_id": p.id, "text": None})
    assert resp.status_code == 200
    assert p.id not in g.current_actions

