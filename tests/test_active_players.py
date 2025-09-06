import sys, pathlib, time
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from oRPG import Game, Player


def test_active_players_filters_stale_entries():
    g = Game()
    recent = Player("Alice", "brave warrior", 1.0, [])
    stale = Player("Bob", "old rogue", 1.0, [])
    g.players = {recent.id: recent, stale.id: stale}

    now = time.time()
    recent.last_seen = now
    stale.last_seen = now - 100

    active = g.active_players(stale_seconds=60)
    assert recent in active
    assert stale not in active
