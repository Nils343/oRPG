import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from oRPG import archetype_for_background


@pytest.mark.parametrize("bg,expected", [
    ("Mystic sorceress exploring ruins", "Mage"),
    ("Silent THIEF from the city", "Rogue"),
    ("An elven archer guarding the forest", "Ranger"),
    ("Devout priest spreading light", "Cleric"),
    ("BARBARIAN warrior of the north", "Warrior"),
    ("A simple farmer with no special training", "Adventurer"),
    ("", "Adventurer"),
])
def test_archetype_for_background(bg, expected):
    assert archetype_for_background(bg) == expected
