import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_state_includes_flags_and_updates_last_seen(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"
    g.last_summary = "summary"
    g.current_actions = {host.id: "look", other.id: "hide"}

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "secret")
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)

    client = TestClient(oRPG.app)

    host.last_seen = 0
    resp = client.get("/state", params={"player_id": host.id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["your_action"] == "look"
    assert data["join_code_required"] is True
    assert host.last_seen > 0

    other.last_seen = 0
    resp2 = client.get("/state", params={"player_id": other.id})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["your_action"] == "hide"
    assert data2["join_code_required"] is True
    assert other.last_seen > 0

def test_state_can_resolve_when_anyone_allowed(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", True)

    client = TestClient(oRPG.app)
    resp = client.get("/state", params={"player_id": other.id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["can_resolve"] is True
