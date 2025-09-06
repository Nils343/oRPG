import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_ensure_initial_scene_sets_initial_state(monkeypatch):
    g = oRPG.Game()
    player = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {player.id: player}
    monkeypatch.setattr(oRPG, "GAME", g)

    async def fake_ollama_chat(messages, options=None):
        # first call returns scene, second call returns summary
        if messages and messages[0].get("content") == oRPG.GM_SYSTEM_PROMPT:
            return "A dark cave. — What do you do?"
        return "They entered a cave."

    monkeypatch.setattr(oRPG, "ollama_chat", fake_ollama_chat)

    asyncio.run(oRPG.ensure_initial_scene())

    assert g.turn_number == 1
    assert g.current_scenario == "A dark cave. — What do you do?"
    assert g.current_actions == {}
    assert g.last_summary == "They entered a cave."
