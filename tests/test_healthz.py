import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_health_endpoint_reports_status(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {player.id: player}
    g.turn_number = 2
    monkeypatch.setattr(oRPG, "GAME", g)

    client = TestClient(oRPG.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["players"] == 1
    assert data["turn"] == 2
