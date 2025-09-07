import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_do_resolution_updates_game_state(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "brave hero", 1.0, [])
    g.players = {player.id: player}
    g.turn_number = 1
    g.current_scenario = "A dragon blocks the path."
    g.current_actions = {player.id: "attack"}
    g.last_summary = "They set out from town."
    monkeypatch.setattr(oRPG, "GAME", g)

    async def fake_ollama_chat(messages, options=None):
        if messages and messages[0].get("content") == oRPG.GM_SYSTEM_PROMPT:
            return "The dragon falls. — What do you do?"
        return "The party defeated a dragon."

    monkeypatch.setattr(oRPG, "ollama_chat", fake_ollama_chat)

    asyncio.run(oRPG.do_resolution())

    assert g.turn_number == 2
    assert g.current_scenario == "The dragon falls. — What do you do?"
    assert g.last_summary == "The party defeated a dragon."
    assert g.current_actions == {}
    assert len(g.history) == 1
    hist = g.history[0]
    assert hist["turn"] == 1
    assert hist["scenario"] == "A dragon blocks the path."
    assert hist["actions"] == {player.id: "attack"}
    assert "The dragon falls" in hist["narration"]


def test_do_resolution_creates_initial_scene_if_none(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "brave hero", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    async def fake_ollama_chat(messages, options=None):
        if messages and messages[0].get("content") == oRPG.GM_SYSTEM_PROMPT:
            return "A forest clearing. — What do you do?"
        return "They reached a forest."

      
def test_do_resolution_history_and_state(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "brave hero", 1.0, [])
    g.players = {player.id: player}
    g.turn_number = 1
    g.current_scenario = "A dragon blocks the path."
    g.current_actions = {player.id: "attack"}
    g.last_summary = "They set out from town."
    monkeypatch.setattr(oRPG, "GAME", g)

    responses = iter([
        "The dragon falls. — What do you do?",
        "The party defeated a dragon.",
    ])

    call_count = 0

    async def fake_ollama_chat(messages, options=None):
        nonlocal call_count
        call_count += 1
        return next(responses)

    monkeypatch.setattr(oRPG, "ollama_chat", fake_ollama_chat)

    asyncio.run(oRPG.do_resolution())

    assert g.turn_number == 1
    assert g.current_scenario == "A forest clearing. — What do you do?"
    assert g.last_summary == "They reached a forest."
    assert g.current_actions == {}
    assert g.history == []
    assert call_count == 2
    assert g.turn_number == 2
    assert g.current_scenario == "The dragon falls. — What do you do?"
    assert g.last_summary == "The party defeated a dragon."
    assert g.current_actions == {}
    assert len(g.history) == 1
    hist = g.history[0]
    assert set(hist.keys()) == {"turn", "scenario", "actions", "narration", "ts"}
    assert hist["turn"] == 1
    assert hist["scenario"] == "A dragon blocks the path."
    assert hist["actions"] == {player.id: "attack"}
    assert hist["narration"] == "The dragon falls. — What do you do?"
    assert isinstance(hist["ts"], str)