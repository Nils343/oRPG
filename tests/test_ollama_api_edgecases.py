import sys, pathlib, asyncio
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import oRPG


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self._payload


class FakeAsyncClient:
    return_payload = {}
    def __init__(self, timeout=None):
        self.timeout = timeout
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False
    async def post(self, url, json):
        return FakeResponse(FakeAsyncClient.return_payload)


class FakeHttpx:
    AsyncClient = FakeAsyncClient


def test_ollama_chat_handles_missing_message(monkeypatch):
    monkeypatch.setattr(oRPG, "httpx", FakeHttpx)
    FakeAsyncClient.return_payload = {}
    out = asyncio.run(oRPG.ollama_chat([{ "role": "user", "content": "hi" }]))
    assert out == ""


def test_ollama_chat_handles_none_message(monkeypatch):
    monkeypatch.setattr(oRPG, "httpx", FakeHttpx)
    FakeAsyncClient.return_payload = {"message": None}
    out = asyncio.run(oRPG.ollama_chat([{ "role": "user", "content": "hi" }]))
    assert out == ""


def test_ollama_chat_handles_nonstring_content(monkeypatch):
    monkeypatch.setattr(oRPG, "httpx", FakeHttpx)
    FakeAsyncClient.return_payload = {"message": {"content": 12345}}
    out = asyncio.run(oRPG.ollama_chat([{ "role": "user", "content": "hi" }]))
    assert out == ""

