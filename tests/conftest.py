import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import oRPG


def assert_last_seen_updates(client: TestClient, player: oRPG.Player, method: str, url: str, **kwargs):
    """Hit an endpoint and assert the player's last_seen is updated."""
    player.last_seen = 0
    response = client.request(method, url, **kwargs)
    assert response.status_code == 200
    assert player.last_seen > 0
    return response
