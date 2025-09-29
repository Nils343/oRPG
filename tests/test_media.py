import asyncio
import sys
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import rpg
from tests.test_rpg import reset_state


class _FakeResponse:
    def __init__(self, headers: dict[str, str], data: list[bytes]):
        self.headers = headers
        self.data = data


class _FakeConvertContext:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self) -> _FakeResponse:
        return self._response

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_elevenlabs_modules(response: _FakeResponse, subscription: object | None = None) -> dict[str, object]:
    if subscription is None:
        subscription = types.SimpleNamespace(
            character_limit=1000,
            character_count=400,
            next_character_count_reset_unix=123456,
        )

    fake_root = types.ModuleType("elevenlabs")
    fake_client = types.ModuleType("elevenlabs.client")
    fake_types = types.ModuleType("elevenlabs.types")

    class FakeElevenLabs:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.user = types.SimpleNamespace(
                subscription=types.SimpleNamespace(get=lambda: subscription)
            )
            self.text_to_speech = types.SimpleNamespace(
                with_raw_response=types.SimpleNamespace(
                    convert=lambda *args, **kwargs: _FakeConvertContext(response)
                )
            )

    class FakeVoiceSettings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_client.ElevenLabs = FakeElevenLabs
    fake_types.VoiceSettings = FakeVoiceSettings
    fake_root.client = fake_client
    fake_root.types = fake_types

    return {
        "elevenlabs": fake_root,
        "elevenlabs.client": fake_client,
        "elevenlabs.types": fake_types,
    }


class ElevenLabsConvertToBase64Tests(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False

    def tearDown(self) -> None:
        rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False
        reset_state()

    def test_returns_audio_and_metadata(self) -> None:
        headers = {
            "X-Characters-Used": "123;foo=bar",
            "X-Request-Id": "req-42",
            "Other": "Value",
        }
        response = _FakeResponse(headers, [b"voice bytes"])
        modules = _make_elevenlabs_modules(response)

        with mock.patch.dict(sys.modules, modules, clear=False):
            result = rpg._elevenlabs_convert_to_base64(
                "Speak kindly",
                api_key="key123",
                model_id="eleven_flash_v2_5",
            )

        self.assertEqual(result["audio_base64"], "dm9pY2UgYnl0ZXM=")
        metadata = result["metadata"]
        self.assertEqual(metadata["characters_reported"], 123)
        self.assertEqual(metadata["character_source"], "x-characters-used")
        self.assertAlmostEqual(metadata["estimated_credits"], 61.5)
        self.assertAlmostEqual(metadata["estimated_cost_usd"], 0.00615)
        self.assertEqual(metadata["subscription_total_credits"], 1000)
        self.assertEqual(metadata["subscription_used_credits"], 400)
        self.assertEqual(metadata["subscription_remaining_credits"], 600)
        self.assertEqual(metadata["request_id"], "req-42")
        self.assertEqual(metadata["headers"]["other"], "Value")

    def test_raises_when_audio_missing(self) -> None:
        headers = {"X-Characters": "7"}
        response = _FakeResponse(headers, [])
        modules = _make_elevenlabs_modules(response)

        with mock.patch.dict(sys.modules, modules, clear=False):
            with self.assertRaises(rpg.ElevenLabsNarrationError) as excinfo:
                rpg._elevenlabs_convert_to_base64("Silence", api_key="key123")

        self.assertIn("no audio", str(excinfo.exception).lower())

    def test_extracts_generic_character_header_and_ignores_non_str_counts(self) -> None:
        headers = {
            "X-Characters-Used": "abc",  # causes the preferred extraction to fail
            "X-Other-Character-Count": "900;foo=bar",
            "X-Request-Id": "req-99",
        }
        response = _FakeResponse(headers, [b"bytes"])
        modules = _make_elevenlabs_modules(response)

        with mock.patch.dict(sys.modules, modules, clear=False):
            result = rpg._elevenlabs_convert_to_base64("Hi", api_key="key123")

        metadata = result["metadata"]
        # Preferred header is skipped because value was non-numeric in the payload
        # The fallback loop should pick the secondary character header.
        self.assertEqual(metadata["character_source"], "x-other-character-count")
        self.assertEqual(metadata["characters_reported"], 900)
        self.assertEqual(metadata["request_id"], "req-99")


class MaybeQueueSceneImageWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.last_image_prompt = None
        rpg.game_state.last_scene_image_turn_index = None

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_worker_retries_until_lock_clears_and_generates(self) -> None:
        captured_tasks: list[asyncio.Future] = []
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            if rpg.game_state.lock.active:
                rpg.game_state.lock = rpg.LockState(active=False, reason="")

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured_tasks.append(coro)  # type: ignore[arg-type]
            return mock.Mock()

        data_url = "data:image/png;base64,ok"
        generate_mock = mock.AsyncMock(return_value=data_url)
        broadcast_mock = mock.AsyncMock()
        announce_mock = mock.AsyncMock()
        clear_mock = mock.Mock()

        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
            mock.patch("rpg.gemini_generate_image", new=generate_mock),
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
            mock.patch("rpg.announce", new=announce_mock),
            mock.patch("rpg._clear_scene_video", new=clear_mock),
        ):
            await rpg.schedule_auto_scene_image(" Mystical forest ", 3)
            self.assertTrue(captured_tasks, "worker coroutine should have been scheduled")
            await captured_tasks[0]

        self.assertGreaterEqual(len(sleeps), 1)
        self.assertEqual(sleeps[0], 0.5)
        generate_mock.assert_awaited_once()
        args, kwargs = generate_mock.await_args
        self.assertEqual(args[0], rpg.game_state.settings.get("image_model"))
        self.assertEqual(args[1], "Mystical forest")
        self.assertEqual(kwargs["purpose"], "scene")
        self.assertEqual(kwargs["turn_index"], 3)
        clear_mock.assert_called_once()
        announce_mock.assert_awaited_once()
        self.assertTrue(broadcast_mock.await_count >= 2)
        self.assertEqual(rpg.game_state.last_image_data_url, data_url)
        self.assertEqual(rpg.game_state.last_image_prompt, "Mystical forest")
        self.assertIsNone(rpg.game_state.last_manual_scene_image_turn_index)
        self.assertFalse(rpg.game_state.lock.active)

    async def test_worker_aborts_after_many_lock_retries(self) -> None:
        captured_tasks: list[asyncio.Future] = []
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured_tasks.append(coro)  # type: ignore[arg-type]
            return mock.Mock()

        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
            mock.patch("rpg.print") as print_mock,
        ):
            await rpg.schedule_auto_scene_image("Forest", 7)
            self.assertTrue(captured_tasks)
            await captured_tasks[0]

        # The worker should give up after repeated retries without clearing the lock.
        self.assertGreaterEqual(len(sleeps), 20)
        print_mock.assert_called()
        self.assertTrue(rpg.game_state.lock.active)


class MaybeQueueSceneVideoWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.last_video_prompt = None
        rpg.game_state.last_video_negative_prompt = None
        rpg.game_state.last_image_prompt = "Still frame"

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_worker_waits_for_lock_and_updates_state(self) -> None:
        captured_tasks: list[asyncio.Future] = []
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            if rpg.game_state.lock.active:
                rpg.game_state.lock = rpg.LockState(active=False, reason="")

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured_tasks.append(coro)  # type: ignore[arg-type]
            return mock.Mock()

        new_video = rpg.SceneVideo(
            url="/generated_media/video.mp4",
            prompt="Still frame",
            negative_prompt="No shadows",
            model="veo-demo",
            updated_at=123.45,
            file_path=str(rpg.GENERATED_MEDIA_DIR / "video.mp4"),
        )

        generate_mock = mock.AsyncMock(return_value=new_video)
        broadcast_mock = mock.AsyncMock()
        announce_mock = mock.AsyncMock()
        clear_mock = mock.Mock()

        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
            mock.patch("rpg.generate_scene_video", new=generate_mock),
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
            mock.patch("rpg.announce", new=announce_mock),
            mock.patch("rpg._clear_scene_video", new=clear_mock),
        ):
            await rpg.schedule_auto_scene_video(None, 4, negative_prompt="No shadows")
            self.assertTrue(captured_tasks, "worker coroutine should have been scheduled")
            await captured_tasks[0]

        self.assertGreaterEqual(len(sleeps), 1)
        self.assertEqual(sleeps[0], 0.5)
        generate_mock.assert_awaited_once()
        args, kwargs = generate_mock.await_args
        self.assertEqual(args[0], "Still frame")
        self.assertEqual(kwargs["negative_prompt"], "No shadows")
        self.assertEqual(kwargs["turn_index"], 4)
        clear_mock.assert_called_once()
        announce_mock.assert_awaited_once()
        self.assertTrue(broadcast_mock.await_count >= 2)
        self.assertIs(rpg.game_state.scene_video, new_video)
        self.assertEqual(rpg.game_state.last_video_prompt, "Still frame")
        self.assertEqual(rpg.game_state.last_video_negative_prompt, "No shadows")
        self.assertEqual(rpg.game_state.last_scene_video_turn_index, 4)
        self.assertFalse(rpg.game_state.lock.active)

    async def test_worker_bails_when_lock_never_clears(self) -> None:
        captured_tasks: list[asyncio.Future] = []
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured_tasks.append(coro)  # type: ignore[arg-type]
            return mock.Mock()

        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
            mock.patch("rpg.print") as print_mock,
        ):
            await rpg.schedule_auto_scene_video("Scenery", 6, negative_prompt="foggy")
            self.assertTrue(captured_tasks)
            await captured_tasks[0]

        self.assertGreaterEqual(len(sleeps), 20)
        print_mock.assert_called()
        self.assertTrue(rpg.game_state.lock.active)


class MaybeQueueSceneImageGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_returns_immediately_when_disabled(self) -> None:
        rpg.game_state.auto_image_enabled = False
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_image("Prompt", 1)
        create_task.assert_not_called()

    async def test_skips_when_prompt_is_blank(self) -> None:
        rpg.game_state.auto_image_enabled = True
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_image("   ", 2)
        create_task.assert_not_called()

    async def test_skips_when_turn_already_handled(self) -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.last_scene_image_turn_index = 5
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_image("Forest", 5)
        create_task.assert_not_called()


class MaybeQueueSceneVideoGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_returns_immediately_when_disabled(self) -> None:
        rpg.game_state.auto_video_enabled = False
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_video("Prompt", 1)
        create_task.assert_not_called()

    async def test_skips_when_prompt_missing(self) -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.last_video_prompt = None
        rpg.game_state.last_image_prompt = None
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_video("  ", 2)
        create_task.assert_not_called()

    async def test_skips_when_turn_already_generated(self) -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.last_scene_video_turn_index = 8
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_scene_video("Prompt", 8)
        create_task.assert_not_called()


class MediaToggleRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self.player = rpg.Player(id="player-1", name="Adventurer", background="Scholar", token="secret")
        rpg.game_state.players[self.player.id] = self.player
        rpg.game_state.turn_index = 5

    def tearDown(self) -> None:
        reset_state()

    def test_toggle_scene_image_enables_and_triggers_queue(self) -> None:
        rpg.game_state.last_image_prompt = "A castle"
        rpg.game_state.auto_video_enabled = True

        queue_mock = mock.AsyncMock()
        broadcast_mock = mock.AsyncMock()

        with (
            mock.patch("rpg.schedule_auto_scene_image", new=queue_mock),
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
        ):
            with TestClient(rpg.app) as client:
                response = client.post(
                    "/api/image_toggle",
                    json={"player_id": self.player.id, "token": "secret", "enabled": True},
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["auto_image_enabled"])
        self.assertFalse(response.json()["auto_video_enabled"])
        broadcast_mock.assert_awaited()
        queue_mock.assert_awaited_once()
        q_args, q_kwargs = queue_mock.await_args
        self.assertEqual(q_args[0], "A castle")
        self.assertEqual(q_args[1], 5)
        self.assertTrue(q_kwargs["force"])
        self.assertTrue(rpg.game_state.auto_image_enabled)
        self.assertFalse(rpg.game_state.auto_video_enabled)

    def test_toggle_scene_video_enables_and_triggers_queue(self) -> None:
        rpg.game_state.last_video_prompt = "Enchanted panorama"
        rpg.game_state.auto_image_enabled = True

        queue_mock = mock.AsyncMock()
        broadcast_mock = mock.AsyncMock()

        with (
            mock.patch("rpg.schedule_auto_scene_video", new=queue_mock),
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
        ):
            with TestClient(rpg.app) as client:
                response = client.post(
                    "/api/video_toggle",
                    json={"player_id": self.player.id, "token": "secret", "enabled": True},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["auto_video_enabled"])
        self.assertFalse(payload["auto_image_enabled"])
        broadcast_mock.assert_awaited()
        queue_mock.assert_awaited_once()
        v_args, v_kwargs = queue_mock.await_args
        self.assertEqual(v_args[0], "Enchanted panorama")
        self.assertEqual(v_args[1], 5)
        self.assertTrue(v_kwargs["force"])
        self.assertTrue(rpg.game_state.auto_video_enabled)
        self.assertFalse(rpg.game_state.auto_image_enabled)


class MaybeQueueTTSWorkerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.auto_tts_enabled = True
        rpg.game_state.settings["elevenlabs_api_key"] = "key123"
        rpg.game_state.settings["narration_model"] = "eleven_turbo_v2_5"
        rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False

    async def asyncTearDown(self) -> None:
        reset_state()
        rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False

    async def test_worker_reports_error_on_exception(self) -> None:
        captured: list[asyncio.Future] = []

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured.append(coro)
            return mock.Mock()

        with (
            mock.patch("rpg._elevenlabs_library_available", return_value=True),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch(
                "rpg.asyncio.to_thread",
                new=mock.AsyncMock(side_effect=rpg.ElevenLabsNarrationError("boom")),
            ),
            mock.patch("rpg._broadcast_tts_error", new=mock.AsyncMock()) as broadcast_mock,
        ):
            await rpg.schedule_auto_tts("Narrate this", 9)
            self.assertTrue(captured, "worker coroutine should be scheduled")
            await captured[0]

        broadcast_mock.assert_awaited_once()
        args = broadcast_mock.await_args.args
        self.assertEqual(args[0], "boom")
        self.assertEqual(args[1], 9)

    async def test_worker_reports_when_audio_missing(self) -> None:
        captured: list[asyncio.Future] = []

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured.append(coro)
            return mock.Mock()

        result_payload = {"audio_base64": None, "metadata": {"headers": {}}}

        with (
            mock.patch("rpg._elevenlabs_library_available", return_value=True),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.to_thread", new=mock.AsyncMock(return_value=result_payload)),
            mock.patch("rpg._broadcast_tts_error", new=mock.AsyncMock()) as broadcast_mock,
        ):
            await rpg.schedule_auto_tts("Narrate this", 5)
            self.assertTrue(captured, "worker coroutine should be scheduled")
            await captured[0]

        broadcast_mock.assert_awaited_once()
        message, turn_index = broadcast_mock.await_args.args
        self.assertIn("no audio", message.lower())
        self.assertEqual(turn_index, 5)
        self.assertIsNone(rpg.game_state.last_tts_turn_index)


class MaybeQueueTTSGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False
        rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False
        rpg.game_state.auto_tts_enabled = True

    async def asyncTearDown(self) -> None:
        reset_state()
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False
        rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False

    async def test_returns_immediately_for_blank_text(self) -> None:
        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_tts("   ", 2)
        create_task.assert_not_called()

    async def test_warns_once_when_api_key_missing(self) -> None:
        with mock.patch("rpg.print") as print_mock:
            await rpg.schedule_auto_tts("Narrate", 3)
            await rpg.schedule_auto_tts("Again", 4)
        # Only the first call should log the warning.
        print_mock.assert_called_once()
        self.assertTrue(rpg._ELEVENLABS_API_KEY_WARNING_LOGGED)

    async def test_warns_when_library_missing(self) -> None:
        rpg.game_state.settings["elevenlabs_api_key"] = "key123"
        with mock.patch("rpg._elevenlabs_library_available", return_value=False), \
                mock.patch("rpg.print") as print_mock:
            await rpg.schedule_auto_tts("Narrate", 5)
        print_mock.assert_called_once()
        self.assertTrue(rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED)

    async def test_worker_records_metadata_with_string_numbers(self) -> None:
        rpg.game_state.settings["elevenlabs_api_key"] = "key123"

        captured: list[asyncio.Future] = []

        def capture_task(coro: asyncio.Future) -> mock.Mock:
            captured.append(coro)
            return mock.Mock()

        metadata = {
            "model_id": "eleven_flash_v2_5",
            "characters_final": 321,
            "estimated_credits": 1.5,
            "estimated_cost_usd": 0.0025,
            "subscription_total_credits": "750",
            "subscription_remaining_credits": "349",
            "subscription_next_reset_unix": 1700,
            "request_id": "req-123",
            "headers": {"X-Character-Usage": "321"},
            "usd_per_million": 11.0,
        }
        result_payload = {"audio_base64": "dm9pY2U=", "metadata": metadata}

        with (
            mock.patch("rpg._elevenlabs_library_available", return_value=True),
            mock.patch("rpg.asyncio.create_task", side_effect=capture_task),
            mock.patch("rpg.asyncio.to_thread", new=mock.AsyncMock(return_value=result_payload)),
            mock.patch("rpg._send_json_to_sockets", new=mock.AsyncMock()),
        ):
            await rpg.schedule_auto_tts("Narrate this", 7)
            self.assertTrue(captured)
            await captured[0]

        self.assertEqual(rpg.game_state.last_tts_model, "eleven_flash_v2_5")
        self.assertEqual(rpg.game_state.last_tts_character_source, metadata.get("character_source"))
        self.assertEqual(rpg.game_state.last_tts_total_credits, 750)
        self.assertEqual(rpg.game_state.last_tts_remaining_credits, 349)
        self.assertEqual(rpg.game_state.last_tts_turn_index, 7)
