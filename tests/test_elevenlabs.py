import asyncio
import base64
import json
import sys
import types
from typing import Any, Dict, List
from unittest import mock

import pytest

import rpg
from tests.test_rpg import reset_state


@pytest.fixture(autouse=True)
def clean_rpg_state() -> None:
    reset_state()
    rpg.reset_session_progress()
    rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False
    rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False
    rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False
    yield
    reset_state()
    rpg.reset_session_progress()
    rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False
    rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False
    rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False


class DummyAsyncResponse:
    def __init__(self, payload: Any, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = rpg.httpx.Request("GET", "https://api.elevenlabs.io/v1/models")
            raise rpg.httpx.HTTPStatusError(
                "error",
                request=request,
                response=rpg.httpx.Response(self.status_code, request=request),
            )


class DummyAsyncClient:
    def __init__(self, response: DummyAsyncResponse):
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, headers: Dict[str, str]) -> DummyAsyncResponse:
        self.calls.append({"url": url, "headers": headers})
        return self._response


def run_async(coro):
    return asyncio.run(coro)


def test_elevenlabs_list_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "modelId": "model-1",
            "display_name": "Model One",
            "languages": [
                {"name": "English"},
                {"language_id": "FR"},
            ],
        },
        {
            "id": "model-2",
            "name": "Second",
            "supported_languages": "German",
        },
    ]
    client = DummyAsyncClient(DummyAsyncResponse(payload))
    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models(" api-key "))

    assert len(models) == 2
    first = models[0]
    assert first["id"] == "model-1"
    assert first["name"] == "Model One"
    assert first["languages"] == ["English", "FR"]
    assert first["language_codes"] == ["en"]
    second = models[1]
    assert second["language_codes"] == ["de"]
    assert client.calls[0]["headers"]["xi-api-key"] == "api-key"


def test_elevenlabs_list_models_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyAsyncClient(DummyAsyncResponse({}, status_code=401))
    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models("bad-key"))
    assert models == []


def test_elevenlabs_list_models_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = DummyAsyncResponse({}, status_code=200)

    def bad_json() -> None:
        raise json.JSONDecodeError("msg", "doc", 0)

    response.json = bad_json  # type: ignore[assignment]
    client = DummyAsyncClient(response)
    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models("key"))
    assert models == []


def test_elevenlabs_list_models_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def passthrough(provider: str, key: str, loader):
        raise AssertionError("loader should not run")

    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)
    models = run_async(rpg.elevenlabs_list_models("  "))
    assert models == []


def _install_elevenlabs_stubs(
    chunks: List[bytes],
    headers: Dict[str, str],
    *,
    subscription_error: Exception | None = None,
    header_raises: bool = False,
    convert_exception: Exception | None = None,
):
    class DummyVoiceSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyResponse:
        def __init__(self):
            self.data = chunks

        @property
        def headers(self):
            if header_raises:
                raise RuntimeError("headers boom")
            return headers

    class DummyConvertContext:
        def __enter__(self):
            if convert_exception is not None:
                raise convert_exception
            return DummyResponse()

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummySubscription:
        character_limit = 1000
        character_count = 100
        next_character_count_reset_unix = 4242

    def subscription_get():
        if subscription_error is not None:
            raise subscription_error
        return DummySubscription()

    class DummyElevenLabs:
        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            self.user = types.SimpleNamespace(subscription=types.SimpleNamespace(get=subscription_get))
            self.text_to_speech = types.SimpleNamespace(
                with_raw_response=types.SimpleNamespace(convert=lambda *args, **kwargs: DummyConvertContext())
            )

    client_module = types.ModuleType("elevenlabs.client")
    client_module.ElevenLabs = DummyElevenLabs
    types_module = types.ModuleType("elevenlabs.types")
    types_module.VoiceSettings = DummyVoiceSettings
    package = types.ModuleType("elevenlabs")
    package.client = client_module
    package.types = types_module

    return {
        "elevenlabs": package,
        "elevenlabs.client": client_module,
        "elevenlabs.types": types_module,
    }


def test_elevenlabs_convert_to_base64_success(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [b"audio-bytes"]
    headers = {
        "X-Characters-Used": "50; detail=primary",
        "X-Request-Id": "req-123",
    }
    stubs = _install_elevenlabs_stubs(chunks, headers)

    with mock.patch.dict(sys.modules, stubs):
        monkeypatch.setattr(
            rpg,
            "_lookup_model_pricing",
            lambda data, model_id: {"usd_per_million": 20.0, "credits_per_character": 0.1},
        )
        result = rpg._elevenlabs_convert_to_base64("Hello there", "secret-key", model_id="model-1")

    audio_b64 = base64.b64encode(b"audio-bytes").decode("ascii")
    assert result["audio_base64"] == audio_b64
    metadata = result["metadata"]
    assert metadata["model_id"] == "model-1"
    assert metadata["characters_reported"] == 50
    assert metadata["characters_final"] == 50
    assert metadata["estimated_cost_usd"] == pytest.approx(0.001)
    assert metadata["estimated_credits"] == pytest.approx(5.0)
    assert metadata["subscription_remaining_credits"] == 900
    assert metadata["headers"]["x-request-id"] == "req-123"


def test_elevenlabs_convert_to_base64_empty_audio_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = _install_elevenlabs_stubs([], {})
    with mock.patch.dict(sys.modules, stubs):
        with pytest.raises(rpg.ElevenLabsNarrationError):
            rpg._elevenlabs_convert_to_base64("Hello", "secret")


def test_elevenlabs_convert_to_base64_handles_subscription_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = _install_elevenlabs_stubs(
        [b"data"],
        {},
        subscription_error=RuntimeError("sub fail"),
        header_raises=True,
    )
    with mock.patch.dict(sys.modules, stubs):
        monkeypatch.setattr(rpg, "_lookup_model_pricing", lambda *args, **kwargs: {})
        result = rpg._elevenlabs_convert_to_base64("Hi", "secret-key")

    metadata = result["metadata"]
    assert metadata["character_source"] == "text_length"
    assert metadata["characters_final"] == len("Hi")
    assert metadata["subscription_total_credits"] is None
    assert metadata["subscription_remaining_credits"] is None
    assert metadata["headers"] == {}


def test_elevenlabs_convert_to_base64_conversion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = _install_elevenlabs_stubs(
        [],
        {},
        convert_exception=RuntimeError("boom"),
    )
    with mock.patch.dict(sys.modules, stubs):
        with pytest.raises(rpg.ElevenLabsNarrationError):
            rpg._elevenlabs_convert_to_base64("Hi", "secret")


def test_elevenlabs_convert_to_base64_empty_inputs_short_circuit() -> None:
    assert rpg._elevenlabs_convert_to_base64("", "key") == {"audio_base64": None, "metadata": {}}
    assert rpg._elevenlabs_convert_to_base64("Hello", "") == {"audio_base64": None, "metadata": {}}


def test_elevenlabs_list_models_server_error_logs(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    response = DummyAsyncResponse({}, status_code=500)
    client = DummyAsyncClient(response)

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models("key"))
    assert models == []
    captured = capsys.readouterr()
    assert "HTTP 500" in captured.err


def test_elevenlabs_list_models_network_exception(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class ExplodingClient(DummyAsyncClient):
        async def get(self, url: str, headers: Dict[str, str]) -> DummyAsyncResponse:
            raise RuntimeError("boom")

    client = ExplodingClient(DummyAsyncResponse({}, status_code=200))

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models("key"))
    assert models == []
    captured = capsys.readouterr()
    assert "Failed to fetch ElevenLabs models" in captured.err


def test_elevenlabs_list_models_handles_models_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "models": [
            {
                "modelId": "model-3",
                "description": "Sample",
                "supported_languages": [
                    {"language_name": "Spanish"},
                    {"id": "EN"},
                    "German",
                ],
            }
        ]
    }
    client = DummyAsyncClient(DummyAsyncResponse(payload))

    async def passthrough(provider: str, key: str, loader):
        return await loader()

    monkeypatch.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(rpg, "_get_cached_models", passthrough)

    models = run_async(rpg.elevenlabs_list_models(" key "))
    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "model-3"
    assert sorted(entry["language_codes"]) == ["de", "en"]


def test_schedule_auto_tts_success_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rpg.game_state.auto_tts_enabled = True
    rpg.game_state.settings["elevenlabs_api_key"] = "key"
    metadata = {
        "model_id": "model-1",
        "character_source": "x-characters",
        "characters_final": 12,
        "estimated_cost_usd": 0.002,
        "estimated_credits": 1.5,
        "headers": {"x-request-id": "req-1"},
        "subscription_total_credits": 100,
        "subscription_remaining_credits": 88,
        "subscription_next_reset_unix": 9999,
    }
    result = {"audio_base64": "QUJD", "metadata": metadata}

    monkeypatch.setattr(rpg, "_elevenlabs_library_available", lambda: True)
    monkeypatch.setattr(rpg.asyncio, "to_thread", mock.AsyncMock(return_value=result))
    send_mock = mock.AsyncMock()
    error_mock = mock.AsyncMock()
    monkeypatch.setattr(rpg, "_send_json_to_sockets", send_mock)
    monkeypatch.setattr(rpg, "_broadcast_tts_error", error_mock)

    created_tasks = []
    original_create = asyncio.create_task

    def capture_task(coro):
        task = original_create(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(rpg.asyncio, "create_task", capture_task)

    async def runner() -> None:
        await rpg.schedule_auto_tts("Narration text", turn_index=3)
        if created_tasks:
            await asyncio.gather(*created_tasks)

    run_async(runner())

    assert rpg.game_state.last_tts_model == "model-1"
    assert rpg.game_state.session_tts_requests == 1
    assert rpg.game_state.last_tts_cost_usd == 0.002
    assert send_mock.await_count == 1
    sockets_arg, payload = send_mock.await_args.args
    assert sockets_arg is rpg.game_state.global_sockets
    assert payload["event"] == "tts_audio"
    assert payload["data"]["audio_base64"] == "QUJD"
    assert error_mock.await_count == 0


def test_schedule_auto_tts_handles_conversion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    rpg.game_state.auto_tts_enabled = True
    rpg.game_state.settings["elevenlabs_api_key"] = "key"

    monkeypatch.setattr(rpg, "_elevenlabs_library_available", lambda: True)
    monkeypatch.setattr(
        rpg.asyncio,
        "to_thread",
        mock.AsyncMock(side_effect=rpg.ElevenLabsNarrationError("fail")),
    )
    send_mock = mock.AsyncMock()
    error_mock = mock.AsyncMock()
    monkeypatch.setattr(rpg, "_send_json_to_sockets", send_mock)
    monkeypatch.setattr(rpg, "_broadcast_tts_error", error_mock)

    created_tasks = []
    original_create = asyncio.create_task

    def capture_task(coro):
        task = original_create(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(rpg.asyncio, "create_task", capture_task)

    async def runner() -> None:
        await rpg.schedule_auto_tts("Narration text", turn_index=5)
        if created_tasks:
            await asyncio.gather(*created_tasks)

    run_async(runner())

    assert send_mock.await_count == 0
    assert error_mock.await_count == 1
    args = error_mock.await_args.args
    assert "fail" in args[0]
    assert args[1] == 5
