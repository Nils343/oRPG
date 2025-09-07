import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        # echo content in a shape similar to Ollama
        return {"message": {"content": self._payload.get("echo", "ok")}}


class FakeAsyncClient:
    last_url = None
    last_json = None
    return_payload = {"echo": "ok"}

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        FakeAsyncClient.last_url = url
        FakeAsyncClient.last_json = json
        return FakeResponse(FakeAsyncClient.return_payload)


class FakeHttpxModule:
    AsyncClient = FakeAsyncClient


def test_ollama_chat_builds_payload_without_options(monkeypatch):
    monkeypatch.setattr(oRPG, "httpx", FakeHttpxModule)
    monkeypatch.setattr(oRPG, "OLLAMA_HOST", "http://example-host")
    monkeypatch.setattr(oRPG, "OLLAMA_MODEL", "test-model")

    FakeAsyncClient.return_payload = {"echo": "  hello\n"}

    result = asyncio.run(oRPG.ollama_chat([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]))
    assert result == "hello"

    assert FakeAsyncClient.last_url == "http://example-host/api/chat"
    payload = FakeAsyncClient.last_json
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["stream"] is False
    assert "options" not in payload


def test_ollama_chat_includes_options(monkeypatch):
    monkeypatch.setattr(oRPG, "httpx", FakeHttpxModule)
    monkeypatch.setattr(oRPG, "OLLAMA_HOST", "http://h")
    monkeypatch.setattr(oRPG, "OLLAMA_MODEL", "m")

    opts = {"temperature": 0.7, "num_ctx": 99}
    FakeAsyncClient.return_payload = {"echo": "ok"}

    _ = asyncio.run(oRPG.ollama_chat([
        {"role": "user", "content": "u"},
    ], options=opts))

    payload = FakeAsyncClient.last_json
    assert payload["options"] == opts

