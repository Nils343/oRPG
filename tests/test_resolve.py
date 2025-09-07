import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient
import asyncio


def test_resolve_requires_host(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)

    called = {"flag": False}

    async def fake_do_resolution():
        called["flag"] = True

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)

    # non-host cannot resolve
    resp = client.post("/resolve", json={"player_id": other.id})
    assert resp.status_code == 403
    assert called["flag"] is False

    # host can resolve
    resp2 = client.post("/resolve", json={"player_id": host.id})
    assert resp2.status_code == 200
    assert resp2.json()["ok"] is True
    assert called["flag"] is True


def test_resolve_allows_anyone_when_enabled(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", True)

    called = {"flag": False}

    async def fake_do_resolution():
        called["flag"] = True

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)

    # non-host can resolve when allowed
    resp = client.post("/resolve", json={"player_id": other.id})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert called["flag"] is True


def test_resolve_rejects_invalid_player(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)

    called = {"flag": False}

    async def fake_do_resolution():
        called["flag"] = True

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)
    resp = client.post("/resolve", json={"player_id": "not-real"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid player."
    assert called["flag"] is False

def test_resolve_rejects_when_already_resolving(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"
    g.resolving = True

    monkeypatch.setattr(oRPG, "GAME", g)

    called = {"flag": False}

    async def fake_do_resolution():
        called["flag"] = True

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)

    resp = client.post("/resolve", json={"player_id": host.id})
    assert resp.status_code == 200
    assert resp.json()["status"] == "already resolving"
    assert called["flag"] is False


def test_resolve_updates_last_seen(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    host.last_seen = 0
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)

    async def fake_do_resolution():
        pass

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)
    resp = client.post("/resolve", json={"player_id": host.id})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert host.last_seen > 0


def test_resolve_sets_and_clears_resolving(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)

    states = []

    async def fake_do_resolution():
        states.append(oRPG.GAME.resolving)
        await asyncio.sleep(0)
        states.append(oRPG.GAME.resolving)

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)

    assert oRPG.GAME.resolving is False
    resp = client.post("/resolve", json={"player_id": host.id})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert states == [True, True]
    assert oRPG.GAME.resolving is False


def test_resolve_missing_player_id(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)

    called = {"flag": False}

    async def fake_do_resolution():
        called["flag"] = True

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)
    resp = client.post("/resolve", json={})
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid player."
    assert called["flag"] is False
