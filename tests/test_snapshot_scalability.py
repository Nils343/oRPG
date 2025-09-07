import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_snapshots_handle_many_players(monkeypatch):
    g = oRPG.Game()
    players = []
    actions = {}
    for i in range(150):
        p = oRPG.Player(f"Player{i}", "background", 1.0, [f"Ability{i}"])
        g.players[p.id] = p
        players.append(p)
        actions[p.id] = f"action {i}"
    monkeypatch.setattr(oRPG, "GAME", g)

    party_out = oRPG.party_snapshot(players)
    assert len(party_out.splitlines()) == len(players)

    actions_out = oRPG.actions_snapshot(actions)
    assert len(actions_out.splitlines()) == len(actions)
