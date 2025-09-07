import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from oRPG import Player


def test_player_initialization_truncates_and_rounds():
    long_name = "   " + "A" * 50 + "   "
    long_bg = "   " + "B" * 450 + "   "
    abilities = [f"ab{i}" for i in range(10)]
    p = Player(long_name, long_bg, 1.236, abilities)
    assert p.name == "A" * 40
    assert p.background == "B" * 400
    assert p.power == 1.24
    assert len(p.abilities) == 5


def test_player_char_class_truncation_and_strip():
    long_class = "  " + ("C" * 45) + "  "
    p = Player("Alice", "bg", 1.0, [], char_class=long_class)
    assert p.char_class == "C" * 40


def test_player_time_fields_are_ordered():
    p = Player("Alice", "bg", 1.0, [])
    import time as _time
    now = _time.time()
    assert p.joined_at <= p.last_seen <= now
