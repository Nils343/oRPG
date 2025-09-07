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
])
def test_archetype_for_background(bg, expected):
    assert archetype_for_background(bg) == expected


@pytest.mark.parametrize(
    "bg",
    [
        "A wandering MAGE seeking knowledge",
        "The wise wizard of the north",
        "An enigmatic SoRcErEr's apprentice",
        "Escaped WARLOCK with a secret",
    ],
)
def test_spellcaster_keywords_map_to_mage(bg):
    assert archetype_for_background(bg) == "Mage"
