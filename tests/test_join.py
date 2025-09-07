import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
import time
import pytest
from fastapi.testclient import TestClient


def test_join_requires_code_and_sets_host(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "secret")

    called = {"flag": False}

    async def fake_initial_scene():
        called["flag"] = True
        g.turn_number = 1
        g.current_scenario = "intro"

    monkeypatch.setattr(oRPG, "ensure_initial_scene", fake_initial_scene)

    client = TestClient(oRPG.app)

    # invalid join code should be rejected
    resp = client.post("/join", json={"name": "Alice", "background": "brave warrior", "code": "wrong"})
    assert resp.status_code == 403
    assert called["flag"] is False
    assert g.host_id is None

    # valid join code allows joining and sets host
    resp = client.post("/join", json={"name": "Alice", "background": "brave warrior", "code": "secret"})
    assert resp.status_code == 200
    data = resp.json()
    assert g.host_id == data["player_id"]
    assert called["flag"] is True

    # subsequent joins do not change host or call initial scene again
    called["flag"] = False
    resp2 = client.post("/join", json={"name": "Bob", "background": "sneaky rogue", "code": "secret"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert g.host_id != data2["player_id"]
    assert called["flag"] is False


def test_join_requires_name_and_background(monkeypatch):
    g = oRPG.Game()
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    client = TestClient(oRPG.app)

    # missing name
    resp = client.post("/join", json={"name": "", "background": "hero"})
    assert resp.status_code == 400
    assert g.players == {}
    assert g.host_id is None

    # missing background
    resp2 = client.post("/join", json={"name": "Alice", "background": ""})
    assert resp2.status_code == 400
    assert g.players == {}
    assert g.host_id is None


def test_join_power_averages_active_players(monkeypatch):
    g = oRPG.Game()
    now = time.time()
    p1 = oRPG.Player("Alice", "warrior", 1.2, [])
    p2 = oRPG.Player("Bob", "rogue", 0.8, [])
    stale = oRPG.Player("Carol", "wizard", 10.0, [])
    p1.last_seen = now
    p2.last_seen = now
    stale.last_seen = now - 1000
    g.players = {p1.id: p1, p2.id: p2, stale.id: stale}
    g.host_id = p1.id
    g.turn_number = 1
    g.current_scenario = "scene"
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    client = TestClient(oRPG.app)

    resp = client.post("/join", json={"name": "Dana", "background": "brave hero"})
    assert resp.status_code == 200
    new_id = resp.json()["player_id"]
    new_player = g.players[new_id]
    assert new_player.power == 1.0
    assert new_player.power != (1.2 + 0.8 + 10.0) / 3


@pytest.mark.parametrize(
    "turn,scenario,called",
    [
        (0, None, True),
        (1, None, False),
        (0, "scene", False),
    ],
)
def test_join_triggers_initial_scene_only_when_initial(monkeypatch, turn, scenario, called):
    g = oRPG.Game()
    g.turn_number = turn
    g.current_scenario = scenario
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    flag = {"called": False}

    async def fake_initial_scene():
        flag["called"] = True

    monkeypatch.setattr(oRPG, "ensure_initial_scene", fake_initial_scene)

    client = TestClient(oRPG.app)

    resp = client.post("/join", json={"name": "Alice", "background": "brave warrior"})
    assert resp.status_code == 200
    assert flag["called"] is called
