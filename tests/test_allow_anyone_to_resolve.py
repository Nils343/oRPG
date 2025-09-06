import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "endpoint,method,expected_key,expect_called",
    [
        ("/resolve", "post", "ok", True),
        ("/state", "get", "can_resolve", False),
    ],
)
def test_anyone_can_resolve_or_view_state(
    monkeypatch, endpoint, method, expected_key, expect_called
):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    other = oRPG.Player("Other", "member", 1.0, [])
    g.players = {host.id: host, other.id: other}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"

    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", True)

    if expect_called:
        called = {"flag": False}

        async def fake_do_resolution():
            called["flag"] = True

        monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)
    else:
        called = None

    client = TestClient(oRPG.app)

    payload = {"player_id": other.id}
    if method == "post":
        resp = client.post(endpoint, json=payload)
    else:
        resp = client.get(endpoint, params=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data[expected_key] is True

    if expect_called:
        assert called["flag"] is True

