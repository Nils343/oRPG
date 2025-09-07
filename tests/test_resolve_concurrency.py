import sys, pathlib, threading, time, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_resolve_while_in_progress_returns_already_resolving(monkeypatch):
    g = oRPG.Game()
    host = oRPG.Player("Host", "leader", 1.0, [])
    g.players = {host.id: host}
    g.host_id = host.id
    g.turn_number = 1
    g.current_scenario = "scene"
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "ALLOW_ANYONE_TO_RESOLVE", False)

    called = {"count": 0}

    async def fake_do_resolution():
        called["count"] += 1
        # hold the resolving state long enough for the second request
        await asyncio.sleep(0.1)

    monkeypatch.setattr(oRPG, "do_resolution", fake_do_resolution)

    client = TestClient(oRPG.app)

    result_one = {}

    def do_first():
        r = client.post("/resolve", json={"player_id": host.id})
        result_one["resp"] = r

    t = threading.Thread(target=do_first)
    t.start()

    # ensure first request starts and sets resolving
    time.sleep(0.02)

    r2 = client.post("/resolve", json={"player_id": host.id})
    t.join()

    resp1 = result_one["resp"]
    assert resp1.status_code == 200
    assert resp1.json().get("ok") is True
    assert r2.status_code == 200
    assert r2.json().get("status") == "already resolving"
    assert called["count"] == 1

