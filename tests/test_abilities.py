import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from oRPG import abilities_for_archetype


def test_abilities_tier_and_signature_truncation():
    bg = "A wandering mage seeking knowledge." + "x" * 80
    abilities = abilities_for_archetype("Mage", 0.5, bg)
    assert abilities[0].startswith("Novice"), abilities
    assert abilities[-1] == f"Signature: {bg.splitlines()[0][:60]}"
    assert len(abilities) == 4

    abilities_mid = abilities_for_archetype("Mage", 1.0, bg)
    assert abilities_mid[0].startswith("Seasoned"), abilities_mid

    abilities_high = abilities_for_archetype("Mage", 1.1, bg)
    assert abilities_high[0].startswith("Expert"), abilities_high


def test_signature_truncates_at_newline_and_60_chars():
    first_line = "A wandering mage seeking knowledge." + "x" * 80
    bg = first_line + "\nSecond line should be ignored"
    abilities = abilities_for_archetype("Mage", 0.5, bg)
    signature = abilities[-1]
    assert signature == f"Signature: {first_line[:60]}"
    assert "Second line should be ignored" not in signature
    assert len(signature) == len("Signature: ") + 60
