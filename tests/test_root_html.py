import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_root_returns_single_page_html():
    client = TestClient(oRPG.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in resp.text.lower()
    # Brand text should be present, but avoid coupling to decorative prefixes
    assert "Ollama Fantasy Party" in resp.text

