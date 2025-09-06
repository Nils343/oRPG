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
