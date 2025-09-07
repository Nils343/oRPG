import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_do_resolution_history_timestamp_includes_timezone(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "bg", 1.0, [])
    g.players = {player.id: player}
    g.turn_number = 1
    g.current_scenario = "Scene."
    g.current_actions = {}
    g.last_summary = "Summary."
    monkeypatch.setattr(oRPG, "GAME", g)

    # two responses: narration then summary
    responses = iter([
        "Narration - What do you do?",
        "Updated summary"
    ])

    async def fake_chat(messages, options=None):
        return next(responses)

    monkeypatch.setattr(oRPG, "ollama_chat", fake_chat)

    asyncio.run(oRPG.do_resolution())

    assert len(g.history) == 1
    ts = g.history[0]["ts"]
    assert ts.endswith("+00:00")
    # also confirm actions were empty and stored as such
    assert g.history[0]["actions"] == {}

