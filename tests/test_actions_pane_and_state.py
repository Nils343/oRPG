import asyncio
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_html_sidebar_replaced_with_actions():
    # Ensure the client UI shows the new Submitted Actions pane
    html = oRPG.HTML_PAGE
    assert "Submitted Actions" in html
    assert '<h4>Summary of Facts</h4>' not in html
    assert 'id="actions"' in html
    assert 'id="summary"' not in html


def test_state_includes_actions_with_names():
    # Use a fresh game state to avoid cross-test interference
    oRPG.GAME = oRPG.Game()

    # Create a known player and an unknown action to test both name and fallback
    p = oRPG.Player(name="Alice", background="Scout from the north.", power=1.0, abilities=["A", "B", "C"], char_class="Scout")
    oRPG.GAME.players[p.id] = p

    unknown_id = "abc1234567890"
    oRPG.GAME.current_actions[p.id] = "Move north quietly"
    oRPG.GAME.current_actions[unknown_id] = "A strange whisper echoes"

    state = asyncio.run(oRPG.get_state(player_id=p.id))

    assert "actions" in state
    actions = state["actions"]
    assert isinstance(actions, list)
    assert len(actions) == 2
    assert state.get("actions_submitted") == 2

    by_id = {a["id"]: a for a in actions}
    assert by_id[p.id]["name"] == p.name
    assert by_id[p.id]["text"] == "Move north quietly"

    # Unknown player id falls back to id prefix
    assert by_id[unknown_id]["name"] == unknown_id[:8]
    assert by_id[unknown_id]["text"] == "A strange whisper echoes"

