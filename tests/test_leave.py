import sys, pathlib, time
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_leave_removes_player(monkeypatch):
    g = oRPG.Game()
    p1 = oRPG.Player("Alice", "warrior", 1.0, [])
    p2 = oRPG.Player("Bob", "rogue", 1.0, [])
    now = time.time()
    p1.last_seen = now
    p2.last_seen = now
    g.players = {p1.id: p1, p2.id: p2}
    g.host_id = p1.id
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.post("/leave", json={"player_id": p2.id})
    assert resp.status_code == 200
    assert p2.id not in g.players
    assert g.host_id == p1.id


def test_leave_reassigns_host(monkeypatch):
    g = oRPG.Game()
    p1 = oRPG.Player("Alice", "warrior", 1.0, [])
    p2 = oRPG.Player("Bob", "rogue", 1.0, [])
    now = time.time()
    p1.last_seen = now
    p2.last_seen = now
    g.players = {p1.id: p1, p2.id: p2}
    g.host_id = p1.id
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.post("/leave", json={"player_id": p1.id})
    assert resp.status_code == 200
    assert p1.id not in g.players
    assert g.host_id == p2.id


def test_leave_last_player_clears_host(monkeypatch):
    g = oRPG.Game()
    p1 = oRPG.Player("Alice", "warrior", 1.0, [])
    p1.last_seen = time.time()
    g.players = {p1.id: p1}
    g.host_id = p1.id
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.post("/leave", json={"player_id": p1.id})
    assert resp.status_code == 200
    assert g.players == {}
    assert g.host_id is None
