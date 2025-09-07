import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG
from fastapi.testclient import TestClient


def test_cors_allows_any_origin_method_and_header():
    client = TestClient(oRPG.app)

    resp = client.options(
        "/",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "X-Custom-Header",
        },
    )

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    assert "PUT" in resp.headers["access-control-allow-methods"]
    assert "X-Custom-Header" in resp.headers["access-control-allow-headers"]
