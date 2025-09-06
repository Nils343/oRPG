import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


def test_initial_scene_user_message_includes_inputs_and_defaults():
    msg = oRPG.initial_scene_user_message("", "- Alice")
    # ensures default summary placeholder and party roster are included
    assert "START A NEW SCENE." in msg
    assert "(none)" in msg
    assert "- Alice" in msg
    assert "— What do you do?" in msg


def test_resolution_user_message_includes_all_sections():
    msg = oRPG.resolution_user_message("facts", "- Bob", "a scenario", "- Bob: acts")
    assert "RESOLVE THE PARTY'S ACTIONS" in msg
    assert "facts" in msg
    assert "a scenario" in msg
    assert "- Bob" in msg
    assert "- Bob: acts" in msg
    assert "— What do you do?" in msg
