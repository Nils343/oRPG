import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_root_contains_key_elements_and_script():
    client = TestClient(oRPG.app)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Key containers
    assert 'id="join"' in html
    assert 'id="game"' in html
    # Inputs and controls
    assert 'id="name"' in html
    assert 'id="background"' in html
    assert 'id="action"' in html
    assert 'id="resolveBtn"' in html
    assert 'id="leaveBtn"' in html
    # Render targets
    assert 'id="scenario"' in html
    assert 'id="summary"' in html
    assert 'id="party"' in html

    # Script bootstraps
    assert "function load()" in html
    assert "load();" in html

