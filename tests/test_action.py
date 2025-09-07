import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient
from tests.conftest import assert_last_seen_updates


def test_submit_action_stores_truncates_and_clears(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "brave hero", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)

    long_text = "x" * 600
    resp = client.post("/action", json={"player_id": player.id, "text": long_text})
    assert resp.status_code == 200
    assert len(g.current_actions[player.id]) == 500

    resp2 = client.post("/action", json={"player_id": player.id, "text": "   "})
    assert resp2.status_code == 200
    assert player.id not in g.current_actions


def test_submit_action_requires_valid_player():
    g = oRPG.Game()
    oRPG.GAME = g
    client = TestClient(oRPG.app)

    resp = client.post("/action", json={"player_id": "bogus", "text": "hi"})
    assert resp.status_code == 400


def test_submit_action_updates_last_seen(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    assert_last_seen_updates(
        client, player, "post", "/action", json={"player_id": player.id, "text": "attack"}
    )


def test_submit_action_trims_and_updates_last_seen(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    long_text = "  " + "x" * 600 + "  "
    assert_last_seen_updates(
        client, player, "post", "/action", json={"player_id": player.id, "text": long_text}
    )
    assert g.current_actions[player.id] == "x" * 500
