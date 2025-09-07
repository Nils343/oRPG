import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_state_with_unknown_player_id_returns_defaults_and_no_update(monkeypatch):
    g = oRPG.Game()
    real = oRPG.Player("Real", "bg", 1.0, [])
    real.last_seen = 123.0
    g.players = {real.id: real}
    g.host_id = real.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "secret")

    client = TestClient(oRPG.app)
    resp = client.get("/state", params={"player_id": "unknown-id"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["your_action"] == ""
    assert data["is_host"] is False
    assert data["join_code_required"] is True
    # ensure no other player's last_seen was updated
    assert g.players[real.id].last_seen == 123.0

