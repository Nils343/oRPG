import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_join_trims_and_truncates_long_inputs(monkeypatch):
    g = oRPG.Game()
    # Make it not the very first join to avoid initial scene call
    g.turn_number = 1
    g.current_scenario = "scene"
    monkeypatch.setattr(oRPG, "GAME", g)
    monkeypatch.setattr(oRPG, "JOIN_CODE", "")

    async def fake_choose_class(background: str, max_attempts: int = 5) -> str:
        return "Shadow Blade"

    monkeypatch.setattr(oRPG, "choose_class_with_ollama", fake_choose_class)

    client = TestClient(oRPG.app)

    long_name = "  " + ("A" * 100) + "  "
    long_bg = "  " + ("B" * 1000) + "  "

    resp = client.post("/join", json={"name": long_name, "background": long_bg})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["name"]) == 40  # truncated
    assert data["name"] == "A" * 40  # trimmed and truncated

    pid = data["player_id"]
    p = g.players[pid]
    assert p.background == "B" * 400
    assert p.char_class == "Shadow Blade"

