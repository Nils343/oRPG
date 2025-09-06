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
