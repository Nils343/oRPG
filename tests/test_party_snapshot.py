import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_party_snapshot_formats_and_handles_empty():
    # empty party
    assert oRPG.party_snapshot([]) == "(no one yet)"

    # populated party formatting
    p1 = oRPG.Player("Alice", "brave warrior", 1.0, ["Slash", "Shield Bash"])
    p2 = oRPG.Player("Bob", "stealthy rogue", 0.9, ["Sneak", "Backstab"])
    result = oRPG.party_snapshot([p1, p2])
    expected = (
        f"- {p1.name} ({p1.power}x power) – Warrior; Abilities: Slash, Shield Bash\n"
        f"- {p2.name} ({p2.power}x power) – Rogue; Abilities: Sneak, Backstab"
    )
    assert result == expected


def test_party_snapshot_includes_all_player_details():
    players = [
        oRPG.Player("Cara", "pious cleric", 1.2, ["Heal", "Bless"]),
        oRPG.Player("Dan", "keen ranger", 0.8, ["Track", "Ambush"]),
    ]
    lines = oRPG.party_snapshot(players).splitlines()
    for line, p in zip(lines, players):
        assert p.name in line
        assert f"{p.power}x power" in line
        assert oRPG.archetype_for_background(p.background) in line
        assert ", ".join(p.abilities) in line
