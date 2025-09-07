import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from oRPG import abilities_for_class


def test_abilities_tier_and_signature_truncation():
    bg = "A wandering mage seeking knowledge." + "x" * 80
    abilities = abilities_for_class("Mage", 0.5, bg)
    assert abilities[0].startswith("Novice"), abilities
    assert abilities[-1] == f"Signature: {bg.splitlines()[0][:60]}"
    assert len(abilities) == 4

    abilities_mid = abilities_for_class("Mage", 1.0, bg)
    assert abilities_mid[0].startswith("Seasoned"), abilities_mid

    abilities_high = abilities_for_class("Mage", 1.1, bg)
    assert abilities_high[0].startswith("Expert"), abilities_high


def test_abilities_order_before_signature():
    bg = "An intrepid adventurer."
    for c in ["Mage", "Rogue", "Ranger", "Cleric", "Warrior", "Adventurer"]:
        abilities = abilities_for_class(c, 1.0, bg)
        # last item is Signature, first is tiered, total 4
        assert len(abilities) == 4
        assert abilities[0].startswith("Seasoned")
        assert abilities[-1].startswith("Signature:")

        

def test_signature_truncates_at_newline_and_60_chars():
    first_line = "A wandering mage seeking knowledge." + "x" * 80
    bg = first_line + "\nSecond line should be ignored"
    abilities = abilities_for_class("Mage", 0.5, bg)
    signature = abilities[-1]
    assert signature == f"Signature: {first_line[:60]}"
    assert "Second line should be ignored" not in signature
    assert len(signature) == len("Signature: ") + 60

    

def test_abilities_tier_boundaries():
    bg = "Any background"
    assert abilities_for_class("Mage", 0.94, bg)[0].startswith("Novice")
    assert abilities_for_class("Mage", 0.95, bg)[0].startswith("Seasoned")
    assert abilities_for_class("Mage", 1.04, bg)[0].startswith("Seasoned")
    assert abilities_for_class("Mage", 1.05, bg)[0].startswith("Expert")
    

def test_abilities_signature_first_line_and_length():
    bg = "First line of background.\nSecond line should be ignored."
    abilities = abilities_for_class("Rogue", 0.9, bg)
    assert len(abilities) == 4
    assert abilities[-1] == f"Signature: {bg.splitlines()[0][:60]}"


def test_abilities_includes_base_traits_in_order():
    abilities = abilities_for_class("Anything", 1.0, "Background text")
    assert abilities[1] == "Resourcefulness"
    assert abilities[2] == "Adaptability"


def test_abilities_class_token_truncation_and_first_line():
    # 25-char class token with newline should be truncated to first 20 chars, first line only
    char_class = ("X" * 25) + "\nSecond Line"
    abilities = abilities_for_class(char_class, 1.0, "Some background")
    assert len(abilities) == 4
    first = abilities[0]
    # Expect tier prefix and exactly 20 X's in the token
    assert first.startswith("Seasoned ")
    assert "X" * 20 in first
    assert first.endswith(" Techniques")
