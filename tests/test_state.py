import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import time

import oRPG
from fastapi.testclient import TestClient
from tests.conftest import assert_last_seen_updates


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

    resp = assert_last_seen_updates(
        client, host, "get", "/state", params={"player_id": host.id}
    )
    data = resp.json()
    assert data["your_action"] == "look"
    assert data["join_code_required"] is True
    resp2 = assert_last_seen_updates(
        client, other, "get", "/state", params={"player_id": other.id}
    )
    data2 = resp2.json()
    assert data2["your_action"] == "hide"
    assert data2["join_code_required"] is True
    

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

    resp_host = client.get("/state", params={"player_id": host.id})
    assert resp_host.status_code == 200
    assert resp_host.json()["can_resolve"] is True

    resp_other = client.get("/state", params={"player_id": other.id})
    assert resp_other.status_code == 200
    assert resp_other.json()["can_resolve"] is True


def test_state_only_host_can_resolve_when_anyone_disallowed(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)

    client = TestClient(oRPG.app)

    resp_host = client.get("/state", params={"player_id": host.id})
    assert resp_host.status_code == 200
    assert resp_host.json()["can_resolve"] is True

    resp_other = client.get("/state", params={"player_id": other.id})
    assert resp_other.status_code == 200
    assert resp_other.json()["can_resolve"] is False


def test_state_marks_host_flag(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id

    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)

    resp_host = assert_last_seen_updates(
        client, host, "get", "/state", params={"player_id": host.id}
    )
    assert resp_host.json()["is_host"] is True

    resp_other = assert_last_seen_updates(
        client, other, "get", "/state", params={"player_id": other.id}
    )
    assert resp_other.json()["is_host"] is False


def test_state_without_player_id_returns_public_info(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"
    g.last_summary = "summary"
    g.current_actions = {host.id: "look", other.id: "hide"}
    g.resolving = True

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "secret")
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", True)

    client = TestClient(oRPG.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()

    assert data["turn"] == 1
    assert data["scenario"] == "scene"
    assert data["summary"] == "summary"
    party = data["party"]
    expected_party = [
        {
            "id": host.id,
            "name": host.name,
            "power": host.power,
            "archetype": oRPG.archetype_for_background(host.background),
        },
        {
            "id": other.id,
            "name": other.name,
            "power": other.power,
            "archetype": oRPG.archetype_for_background(other.background),
        },
    ]
    assert party == expected_party
    assert data["actions_submitted"] == 2
    assert data["resolving"] is True
    assert data["join_code_required"] is True
    assert data["can_resolve"] is True


def test_state_can_resolve_reflects_runtime_toggle(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1

    monkeypatch.setattr(oRPG, "GAME", g)
    client = TestClient(oRPG.app)

    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", True)
    resp = client.get("/state", params={"player_id": other.id})
    assert resp.status_code == 200
    assert resp.json()["can_resolve"] is True

    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)
    resp2 = client.get("/state", params={"player_id": other.id})
    assert resp2.status_code == 200
    assert resp2.json()["can_resolve"] is False

    
def test_state_party_excludes_stale_players(monkeypatch):
    g = oRPG.Game()
    recent = oRPG.Player("Recent", "active", 1.0, [])
    stale = oRPG.Player("Stale", "inactive", 1.0, [])
    g.players = {recent.id: recent, stale.id: stale}

    now = time.time()
    recent.last_seen = now
    stale.last_seen = now - 601

    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.get("/state", params={"player_id": recent.id})
    assert resp.status_code == 200
    data = resp.json()
    party = data["party"]
    ids = [p["id"] for p in party]
    assert recent.id in ids
    assert stale.id not in ids
    assert data["is_host"] is False


def test_state_last_seen_monotonic(monkeypatch):
    import itertools

    g = oRPG.Game()
    player = oRPG.Player("Alice", "adventurer", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    player.last_seen = 0
    times = itertools.count(100.0, 1.0)
    monkeypatch.setattr(oRPG.time, "time", lambda: next(times))

    client = TestClient(oRPG.app)
    resp1 = client.get("/state", params={"player_id": player.id})
    assert resp1.status_code == 200
    first_seen = player.last_seen
    resp2 = client.get("/state", params={"player_id": player.id})
    assert resp2.status_code == 200
    assert player.last_seen > first_seen
