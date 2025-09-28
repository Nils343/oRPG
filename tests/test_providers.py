import json
import unittest
from typing import Any, Dict
from unittest import mock

import httpx
import rpg
from fastapi import HTTPException

from tests.test_rpg import reset_state


class _DummyResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: Dict[str, str] | None = None,
        text: str | None = None,
        raise_for_status_exc: Exception | None = None,
        json_exception: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload)
        self._raise_for_status_exc = raise_for_status_exc
        self._json_exception = json_exception

    def json(self) -> Any:
        if self._json_exception is not None:
            raise self._json_exception
        return self._payload

    def raise_for_status(self) -> None:
        if self._raise_for_status_exc is not None:
            raise self._raise_for_status_exc


class _DummyAsyncClient:
    def __init__(self, response: _DummyResponse):
        self._response = response
        self.calls: list[Dict[str, Any]] = []

    async def __aenter__(self) -> "_DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any]) -> _DummyResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._response


class _DummyGetClient:
    def __init__(self, responses: list[_DummyResponse]):
        self._responses = list(responses)
        self.calls: list[Dict[str, Any]] = []

    async def __aenter__(self) -> "_DummyGetClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(
        self,
        url: str,
        headers: Dict[str, str] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> _DummyResponse:
        self.calls.append({
            "url": url,
            "headers": headers or {},
            "params": params or {},
        })
        if not self._responses:
            raise AssertionError("No responses configured")
        return self._responses.pop(0)


class ElevenLabsListModelsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_returns_empty_when_unauthorized(self) -> None:
        url = f"{rpg.ELEVENLABS_BASE_URL.rstrip('/')}/v1/models"
        request = httpx.Request("GET", url)
        response_obj = httpx.Response(401, request=request)
        exc = httpx.HTTPStatusError("unauthorized", request=request, response=response_obj)
        response = _DummyResponse({}, status_code=401, raise_for_status_exc=exc)
        client = _DummyGetClient([response])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
            mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
            mock.patch("rpg._get_cached_models", side_effect=passthrough),
        ):
            result = await rpg.elevenlabs_list_models("  test-key  ")

        self.assertEqual(result, [])

    async def test_handles_malformed_json_response(self) -> None:
        json_error = json.JSONDecodeError("bad json", "{}", 0)
        response = _DummyResponse({}, json_exception=json_error)
        client = _DummyGetClient([response])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
            mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
            mock.patch("rpg._get_cached_models", side_effect=passthrough),
        ):
            result = await rpg.elevenlabs_list_models("key")

        self.assertEqual(result, [])

    async def test_parses_languages_and_codes(self) -> None:
        payload = [
            {
                "model_id": "eleven_flash_v2_5",
                "name": "Flash",
                "languages": [
                    {"name": "English"},
                    {"language_id": "de-DE"},
                    "en",
                ],
            },
            {
                "modelId": "eleven_multilingual_v2",
                "description": "Multilingual",
                "languages": "Deutsch",
            },
        ]

        response = _DummyResponse(payload)
        client = _DummyGetClient([response])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
            mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
            mock.patch("rpg._get_cached_models", side_effect=passthrough),
        ):
            result = await rpg.elevenlabs_list_models("key")

        self.assertEqual(len(result), 2)
        first = result[0]
        self.assertEqual(first["id"], "eleven_flash_v2_5")
        self.assertIn("en", first["language_codes"])
        self.assertIn("de", first["language_codes"])
        second = result[1]
        self.assertEqual(second["id"], "eleven_multilingual_v2")
        self.assertEqual(second["language_codes"], ["de"])


class GrokGenerateStructuredTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.settings["thinking_mode"] = "balanced"

    async def test_success_records_usage_and_cost(self) -> None:
        schema = {"type": "OBJECT", "properties": {"nar": {"type": "STRING"}}}
        payload = {"players": []}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": json.dumps({"nar": "Storytime"})}
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": "17",
                "completion_tokens": 42,
                "completion_tokens_details": {"reasoning_tokens": "5"},
            },
        }
        response = _DummyResponse(response_payload, headers={"x-request-id": "req-123"})
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="api-key"), \
                mock.patch("rpg.calculate_turn_cost") as cost_mock, \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            cost_mock.return_value = {"total_usd": 0.5}
            result = await rpg._grok_generate_structured(
                model="grok-beta",
                system_prompt="You are the narrator",
                user_payload=payload,
                schema=schema,
            )

        self.assertEqual(result, {"nar": "Storytime"})
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer api-key")
        response_format = client.calls[0]["json"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["schema"]["type"].lower(), "object")
        self.assertEqual(rpg.game_state.last_token_usage, {"input": 17, "output": 42, "thinking": 5})
        self.assertEqual(rpg.game_state.session_token_usage["input"], 17)
        self.assertEqual(rpg.game_state.session_token_usage["thinking"], 5)
        self.assertAlmostEqual(rpg.game_state.session_cost_usd, 0.5)
        cost_mock.assert_called_once_with("grok-beta", 17, 47)

    async def test_http_error_raises(self) -> None:
        response = _DummyResponse({}, status_code=500, text="Internal error")
        client = _DummyAsyncClient(response)
        with mock.patch("rpg.require_text_api_key", return_value="api-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="System",
                    user_payload={},
                    schema={},
                )
        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertEqual(excinfo.exception.detail, "Internal error")

    async def test_missing_choices_raises(self) -> None:
        response = _DummyResponse({"choices": []}, status_code=200)
        client = _DummyAsyncClient(response)
        with mock.patch("rpg.require_text_api_key", return_value="api-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException):
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="System",
                    user_payload={},
                    schema={},
                )


class OpenAIGenerateStructuredTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.settings["thinking_mode"] = "deep"

    async def test_success_records_usage_and_reasoning(self) -> None:
        schema = {"type": "OBJECT", "properties": {"nar": {"type": "STRING"}}}
        payload = {"turn": 1}
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": json.dumps({"nar": "Resolved"})}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 30,
                "reasoning_tokens": 4,
            },
        }
        response = _DummyResponse(response_payload, headers={"x-openai-request-id": "req-456"})
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.calculate_turn_cost") as cost_mock, \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            cost_mock.return_value = {"total_usd": 0.25}
            result = await rpg._openai_generate_structured(
                model="openai/o1-mini",
                system_prompt="Narrate",
                user_payload=payload,
                schema=schema,
            )

        self.assertEqual(result, {"nar": "Resolved"})
        req = client.calls[0]
        self.assertEqual(req["headers"].get("OpenAI-Beta"), "reasoning")
        self.assertEqual(
            req["json"]["response_format"]["json_schema"]["schema"]["type"].lower(),
            "object",
        )
        self.assertEqual(req["json"]["reasoning"], {"effort": "high"})
        self.assertEqual(rpg.game_state.last_token_usage, {"input": 12, "output": 30, "thinking": 4})
        self.assertEqual(rpg.game_state.session_token_usage["output"], 30)
        self.assertAlmostEqual(rpg.game_state.session_cost_usd, 0.25)
        cost_mock.assert_called_once_with("openai/o1-mini", 12, 34)

    async def test_http_error_raises(self) -> None:
        response = _DummyResponse({}, status_code=500, text="OpenAI boom")
        client = _DummyAsyncClient(response)
        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._openai_generate_structured(
                    model="openai/o1-mini",
                    system_prompt="Narrate",
                    user_payload={},
                    schema={},
                )
        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertEqual(excinfo.exception.detail, "OpenAI boom")

    async def test_malformed_output_raises(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "not json"}
                    ],
                }
            ],
            "usage": {},
        }
        response = _DummyResponse(response_payload)
        client = _DummyAsyncClient(response)
        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException):
                await rpg._openai_generate_structured(
                    model="openai/o1-mini",
                    system_prompt="Narrate",
                    user_payload={},
                    schema={},
                )



class StructuredGenerationParsingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_grok_code_block_parsing_and_snapshots(self) -> None:
        rpg.game_state.settings["thinking_mode"] = "mystery"  # exercise mode fallback

        schema = {"type": "OBJECT"}
        payload = {"turn": 2}
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "```json\n{\"nar\": \"Story\"}\n```"}
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": "5",
                "completion_tokens": "9",
                "completion_tokens_details": {"reasoning": "3"},
            },
        }
        response = _DummyResponse(response_payload, text=lambda: "callable-text")
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="grok-key"), \
                mock.patch("rpg.calculate_turn_cost", return_value=None), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            result = await rpg._grok_generate_structured(
                model="grok-beta",
                system_prompt="System",
                user_payload=payload,
                schema=schema,
                dev_snapshot="scenario",
            )

        self.assertEqual(result, {"nar": "Story"})
        self.assertEqual(rpg.game_state.last_cost_usd, None)
        self.assertEqual(rpg.game_state.session_token_usage["input"], 5)
        self.assertEqual(rpg.game_state.session_token_usage["output"], 9)
        self.assertEqual(rpg.game_state.session_token_usage["thinking"], 3)
        self.assertIsNotNone(rpg.game_state.last_turn_request)
        self.assertIsNotNone(rpg.game_state.last_turn_response)

    async def test_openai_code_block_parsing_and_snapshots(self) -> None:
        rpg.game_state.settings["thinking_mode"] = "bogus"

        payload = {"players": []}
        response_payload = {
            "output": [],
            "output_text": "```json\n{\"nar\": \"Ok\"}\n```",
            "usage": {
                "input_tokens": 4,
                "output_tokens": 6,
                "output_tokens_details": {"thinking_tokens": 2},
            },
        }
        response = _DummyResponse(response_payload, headers={"x-test": "1"}, text=lambda: "callable")
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.calculate_turn_cost", return_value=None), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            result = await rpg._openai_generate_structured(
                model="openai/o1-preview",
                system_prompt="Narrate",
                user_payload=payload,
                schema={},
                dev_snapshot="scenario",
            )

        self.assertEqual(result, {"nar": "Ok"})
        self.assertEqual(rpg.game_state.session_token_usage["thinking"], 2)
        self.assertEqual(rpg.game_state.last_cost_usd, None)
        self.assertIsNotNone(rpg.game_state.last_turn_request)
        self.assertIsNotNone(rpg.game_state.last_turn_response)


class StructuredGenerationErrorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_grok_rejects_non_dict_response_payload(self) -> None:
        schema = {"type": "OBJECT"}
        payload = {"turn": 1}

        def explosive_text() -> None:
            raise RuntimeError("boom")

        response = _DummyResponse([], status_code=200, text=explosive_text)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="grok-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="System",
                    user_payload=payload,
                    schema=schema,
                )

        self.assertEqual(excinfo.exception.status_code, 502)

    async def test_grok_rejects_choice_without_dict(self) -> None:
        response_payload = {
            "choices": ["oops"],
            "usage": {},
        }
        response = _DummyResponse(response_payload)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="grok-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="System",
                    user_payload={},
                    schema={"type": "OBJECT"},
                )

        self.assertEqual(excinfo.exception.status_code, 502)

    async def test_grok_rejects_empty_content_after_strip(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": ["   ", {"value": "\t"}],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": True,
                "completion_tokens": 0,
            },
        }
        response = _DummyResponse(response_payload)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="grok-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="System",
                    user_payload={},
                    schema={"type": "OBJECT"},
                )

        self.assertEqual(excinfo.exception.status_code, 502)

    async def test_gemini_structured_surfaces_http_failure(self) -> None:
        response = _DummyResponse({}, status_code=500, text="backend failure")
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_gemini_api_key", return_value="gemini-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._gemini_generate_structured(
                    model="models/gemini-1.5-flash",
                    system_prompt="System",
                    user_payload={},
                    schema={"type": "OBJECT"},
                )

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("backend failure", excinfo.exception.detail)

    async def test_gemini_structured_rejects_non_dict_json(self) -> None:
        response = _DummyResponse([], status_code=200)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_gemini_api_key", return_value="gemini-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._gemini_generate_structured(
                    model="models/gemini-1.5-flash",
                    system_prompt="System",
                    user_payload={},
                    schema={"type": "OBJECT"},
                )

        self.assertEqual(excinfo.exception.status_code, 502)

    async def test_openai_structured_normalizes_schema_defaults(self) -> None:
        schema = {"type": "OBJECT", "properties": {"nar": {"type": "STRING"}}}
        payload = {"turn": 7}
        response_payload = {
            "output": [],
            "output_text": "{\"nar\": \"All good\"}",
            "usage": {},
        }
        response = _DummyResponse(response_payload)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.calculate_turn_cost", return_value=None), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            result = await rpg._openai_generate_structured(
                model="openai/gpt-4o-mini",
                system_prompt="System",
                user_payload=payload,
                schema=schema,
                schema_name="summary",
            )

        self.assertEqual(result, {"nar": "All good"})
        request_body = client.calls[0]["json"]
        json_schema_cfg = request_body["response_format"]["json_schema"]
        self.assertEqual(json_schema_cfg["name"], "summary")
        self.assertEqual(json_schema_cfg["schema"]["type"].lower(), "object")

    async def test_openai_structured_raises_on_empty_output(self) -> None:
        def bad_text() -> None:
            raise RuntimeError("no text")

        response_payload = {
            "output": [{"type": "message", "content": []}],
            "output_text": "  ",
            "usage": {},
        }
        response = _DummyResponse(response_payload, text=bad_text)
        client = _DummyAsyncClient(response)

        with mock.patch("rpg.require_text_api_key", return_value="openai-key"), \
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg._openai_generate_structured(
                    model="openai/gpt-4o-mini",
                    system_prompt="System",
                    user_payload={},
                    schema={},
                )

        self.assertEqual(excinfo.exception.status_code, 502)


class GeminiListModelsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_success_with_pagination_and_normalization(self) -> None:
        page_one = _DummyResponse(
            {
                "models": [
                    {
                        "name": "models/gemini-1.5-pro",
                        "displayName": "Gemini 1.5 Pro",
                        "supportedGenerationMethods": ["generateContent", "RESPONSES"],
                    },
                    {
                        "name": "model-custom",
                        "display_name": "Custom",
                        "supported_actions": {"predictLongRunning": True},
                    },
                ],
                "nextPageToken": "page-2",
            }
        )
        page_two = _DummyResponse(
            {
                "models": [
                    {
                        "name": "models/veo-1",
                        "displayName": "Veo",
                        "supportedGenerationMethods": ["PredictLongRunning"],
                    },
                    {
                        "name": "models/other",
                        "description": "Other",
                    },
                ]
            }
        )
        client = _DummyGetClient([page_one, page_two])

        cache_calls: list[tuple[str, str]] = []

        async def passthrough(provider: str, key: str, loader):
            cache_calls.append((provider, key))
            return await loader()

        with (
                mock.patch("rpg.require_gemini_api_key", return_value="api-key"),
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            models = await rpg.gemini_list_models()

        self.assertEqual(cache_calls, [(rpg.TEXT_PROVIDER_GEMINI, "api-key")])
        self.assertEqual(len(models), 4)
        first = models[0]
        self.assertEqual(first["name"], "models/gemini-1.5-pro")
        self.assertEqual(first["modelId"], "gemini-1.5-pro")
        self.assertEqual(first["category"], "text")
        video_entry = next(item for item in models if item["name"] == "models/veo-1")
        self.assertEqual(video_entry["category"], "video")
        self.assertEqual(video_entry["family"], "veo")
        custom_entry = next(item for item in models if item["name"] == "model-custom")
        self.assertIn("predictlongrunning", [s.lower() for s in custom_entry["supported"]])
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("pageToken", client.calls[0]["params"])
        self.assertEqual(client.calls[1]["params"].get("pageToken"), "page-2")

    async def test_http_error_raises(self) -> None:
        failure = _DummyResponse({}, status_code=500, text="boom")
        client = _DummyGetClient([failure])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.require_gemini_api_key", return_value="api-key"),
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.gemini_list_models()

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("boom", excinfo.exception.detail)

    async def test_malformed_json_propagates(self) -> None:
        broken = _DummyResponse({}, status_code=200)

        def raise_json_error() -> None:
            raise json.JSONDecodeError("bad", "doc", 0)

        broken.json = raise_json_error  # type: ignore[assignment]
        client = _DummyGetClient([broken])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.require_gemini_api_key", return_value="api-key"),
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(json.JSONDecodeError):
                await rpg.gemini_list_models()

    async def test_cached_result_skips_loader(self) -> None:
        payload = _DummyResponse({
            "models": [
                {
                    "name": "models/gemini-flash",
                    "displayName": "Flash",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        })
        client = _DummyGetClient([payload])

        class CacheStub:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []
                self.load_count = 0
                self.cached: list[Dict[str, Any]] | None = None

            async def __call__(self, provider: str, key: str, loader):
                self.calls.append((provider, key))
                if self.cached is not None:
                    return self.cached
                self.load_count += 1
                self.cached = await loader()
                return self.cached

        cache_stub = CacheStub()

        with (
                mock.patch("rpg.require_gemini_api_key", return_value="api-key"),
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", cache_stub),
            ):
            result_one = await rpg.gemini_list_models()
            result_two = await rpg.gemini_list_models()

        self.assertIs(result_one, result_two)
        self.assertEqual(cache_stub.load_count, 1)
        self.assertEqual(len(client.calls), 1)


class GrokListModelsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_success_normalizes_supported_values(self) -> None:
        response = _DummyResponse(
            {
                "data": [
                    {
                        "id": "grok-1",
                        "display_name": "Grok One",
                        "capabilities": ["chat.completions"],
                    },
                    {
                        "model": "grok-2",
                        "description": "Two",
                        "modalities": {"text": True},
                    },
                    {
                        "slug": "no-chat",
                        "interfaces": ["embeddings"],
                    },
                ]
            }
        )
        client = _DummyGetClient([response])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            models = await rpg.grok_list_models(" secret ")

        self.assertEqual(len(models), 3)
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer secret")
        appended = next(item for item in models if item["name"] == "no-chat")
        self.assertIn("chat.completions", [s.lower() for s in appended["supported"]])

    async def test_http_error_raises(self) -> None:
        failure = _DummyResponse({}, status_code=403, text="denied")
        client = _DummyGetClient([failure])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.grok_list_models("token")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("denied", excinfo.exception.detail)

    async def test_malformed_json_raises_http_exception(self) -> None:
        broken = _DummyResponse({}, status_code=200)

        def raise_json_error() -> None:
            raise ValueError("bad json")

        broken.json = raise_json_error  # type: ignore[assignment]
        client = _DummyGetClient([broken])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.grok_list_models("token")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("Malformed Grok", excinfo.exception.detail)

    async def test_cached_result_skips_loader(self) -> None:
        response = _DummyResponse({"data": [{"id": "grok-1", "capabilities": ["chat.completions"]}]})
        client = _DummyGetClient([response])

        class CacheStub:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []
                self.load_count = 0
                self.cached: list[Dict[str, Any]] | None = None

            async def __call__(self, provider: str, key: str, loader):
                self.calls.append((provider, key))
                if self.cached is not None:
                    return self.cached
                self.load_count += 1
                self.cached = await loader()
                return self.cached

        cache_stub = CacheStub()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", cache_stub),
            ):
            one = await rpg.grok_list_models("token")
            two = await rpg.grok_list_models("token")

        self.assertIs(one, two)
        self.assertEqual(cache_stub.load_count, 1)
        self.assertEqual(len(client.calls), 1)

    async def test_empty_key_short_circuits(self) -> None:
        with mock.patch("rpg._get_cached_models") as cache_mock:
            result = await rpg.grok_list_models("   ")

        self.assertEqual(result, [])
        cache_mock.assert_not_called()


class OpenAIListModelsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_success_normalizes_entries(self) -> None:
        response = _DummyResponse(
            {
                "data": [
                    {
                        "id": "gpt-4o",
                        "display_name": "GPT-4o",
                        "capabilities": ["responses"],
                    },
                    {
                        "model": "gpt-mini",
                        "description": "Mini",
                        "modalities": {"responses": True},
                    },
                    {
                        "slug": "fallback",
                        "interfaces": [],
                    },
                ]
            }
        )
        client = _DummyGetClient([response])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            models = await rpg.openai_list_models(" openai-key ")

        self.assertEqual(len(models), 3)
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer openai-key")
        fallback = next(item for item in models if item["name"] == "fallback")
        self.assertIn("responses", [s.lower() for s in fallback["supported"]])

    async def test_http_error_raises(self) -> None:
        failure = _DummyResponse({}, status_code=500, text="openai down")
        client = _DummyGetClient([failure])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.openai_list_models("key")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("openai down", excinfo.exception.detail)

    async def test_malformed_json_raises_http_exception(self) -> None:
        broken = _DummyResponse({}, status_code=200)

        def raise_json_error() -> None:
            raise ValueError("bad json")

        broken.json = raise_json_error  # type: ignore[assignment]
        client = _DummyGetClient([broken])

        async def passthrough(provider: str, key: str, loader):
            return await loader()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", passthrough),
            ):
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.openai_list_models("key")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("Malformed OpenAI", excinfo.exception.detail)

    async def test_cached_result_skips_loader(self) -> None:
        response = _DummyResponse({"data": [{"id": "gpt-4o", "capabilities": ["responses"]}]})
        client = _DummyGetClient([response])

        class CacheStub:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []
                self.load_count = 0
                self.cached: list[Dict[str, Any]] | None = None

            async def __call__(self, provider: str, key: str, loader):
                self.calls.append((provider, key))
                if self.cached is not None:
                    return self.cached
                self.load_count += 1
                self.cached = await loader()
                return self.cached

        cache_stub = CacheStub()

        with (
                mock.patch("rpg.httpx.AsyncClient", lambda *args, **kwargs: client),
                mock.patch("rpg._get_cached_models", cache_stub),
            ):
            one = await rpg.openai_list_models("key")
            two = await rpg.openai_list_models("key")

        self.assertIs(one, two)
        self.assertEqual(cache_stub.load_count, 1)
        self.assertEqual(len(client.calls), 1)

    async def test_empty_key_short_circuits(self) -> None:
        with mock.patch("rpg._get_cached_models") as cache_mock:
            result = await rpg.openai_list_models("   ")

        self.assertEqual(result, [])
        cache_mock.assert_not_called()
