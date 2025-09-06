import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_actions_snapshot_formats_and_handles_unknown_player(monkeypatch):
    g = oRPG.Game()
    p1 = oRPG.Player("Alice", "hero", 1.0, [])
    g.players = {p1.id: p1}
    monkeypatch.setattr(oRPG, "GAME", g)

    actions = {
        p1.id: " attack ",
        "1234567890abcdef": " defend "
    }
    result = oRPG.actions_snapshot(actions)
    assert result == f"- Alice: attack\n- 12345678: defend"


def test_actions_snapshot_empty():
    assert oRPG.actions_snapshot({}) == "(no actions submitted)"
