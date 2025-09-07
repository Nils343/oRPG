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

def test_active_players_uses_default_window():
    g = Game()
    inside = Player("Carol", "mighty mage", 1.0, [])
    boundary = Player("Dave", "old ranger", 1.0, [])
    outside = Player("Eve", "ancient cleric", 1.0, [])
    g.players = {inside.id: inside, boundary.id: boundary, outside.id: outside}

    now = time.time()
    inside.last_seen = now - 100  # within default window
    boundary.last_seen = now - 600  # exactly at boundary; should be excluded
    outside.last_seen = now - 601  # outside default window

    active = g.active_players()
    assert inside in active
    assert boundary not in active
    assert outside not in active
