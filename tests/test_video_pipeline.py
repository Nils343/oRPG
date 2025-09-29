import asyncio
import io
import struct
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

from fastapi.testclient import TestClient

import rpg
from tests.test_rpg import reset_state


class ExtractMvhdDurationTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_state()

    def test_version_zero_duration(self) -> None:
        # version 0 mvhd atom with timescale 1000 and duration 5000 (5 seconds)
        body = bytes([0, 0, 0, 0]) + struct.pack(
            ">IIII",
            0,
            0,
            1000,
            5000,
        ) + b"\x00" * 16
        atom = struct.pack(">I4s", 8 + len(body), b"mvhd") + body
        stream = io.BytesIO(atom)
        duration = rpg._extract_mvhd_duration(stream)
        self.assertEqual(duration, 5.0)

    def test_version_one_duration(self) -> None:
        # version 1 mvhd atom with timescale 48000 and duration 96000 (2 seconds)
        body = bytes([1, 0, 0, 0])
        body += struct.pack(">QQ", 0, 0)
        body += b"\x00\x00\x00\x00"
        body += struct.pack(">I", 48000)
        body += struct.pack(">Q", 96000)
        body += b"\x00" * 16
        atom = struct.pack(">I4s", 8 + len(body), b"mvhd") + body
        stream = io.BytesIO(atom)
        duration = rpg._extract_mvhd_duration(stream)
        self.assertEqual(duration, 2.0)


class ClearSceneVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()

    def tearDown(self) -> None:
        reset_state()

    def test_clear_scene_video_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scene.mp4"
            path.write_bytes(b"video")
            rpg.game_state.scene_video = rpg.SceneVideo(
                url="/generated_media/scene.mp4",
                prompt="prompt",
                negative_prompt=None,
                model="model",
                updated_at=1.23,
                file_path=str(path),
            )
            rpg.game_state.last_scene_video_turn_index = 42

            rpg._clear_scene_video(remove_file=True)

            self.assertIsNone(rpg.game_state.scene_video)
            self.assertIsNone(rpg.game_state.last_scene_video_turn_index)
            self.assertFalse(path.exists())

    def test_clear_scene_video_preserves_file_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scene.mp4"
            path.write_bytes(b"video")
            rpg.game_state.scene_video = rpg.SceneVideo(
                url="/generated_media/scene.mp4",
                prompt="prompt",
                negative_prompt=None,
                model="model",
                updated_at=1.23,
                file_path=str(path),
            )
            rpg.game_state.last_scene_video_turn_index = 7

            rpg._clear_scene_video(remove_file=False)

            self.assertIsNone(rpg.game_state.scene_video)
            self.assertIsNone(rpg.game_state.last_scene_video_turn_index)
            self.assertTrue(path.exists())


class SessionResetSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.cancel_pending_reset_task()

    async def asyncTearDown(self) -> None:
        rpg.cancel_pending_reset_task()
        reset_state()

    async def test_schedule_replaces_existing_task_and_runs_reset(self) -> None:
        loop = asyncio.get_running_loop()
        real_tasks: list[asyncio.Task] = []
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        maybe_mock = mock.Mock(return_value=True)
        broadcast_mock = mock.AsyncMock()

        dummy_task = mock.Mock()
        dummy_task.done.return_value = False

        first_call = True

        def create_task_side_effect(coro):
            nonlocal first_call
            if first_call:
                first_call = False
                dummy_task.coro = coro
                return dummy_task
            task = loop.create_task(coro)
            real_tasks.append(task)
            return task

        with (
            mock.patch("rpg.reset_session_if_inactive", new=maybe_mock),
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
            mock.patch("rpg.asyncio.create_task", side_effect=create_task_side_effect),
        ):
            rpg.schedule_session_reset_check(delay=0.1)
            self.assertIs(rpg._RESET_CHECK_TASK, dummy_task)
            rpg.schedule_session_reset_check(delay=0.2)
            self.assertEqual(len(real_tasks), 1)
            dummy_task.cancel.assert_called_once()
            dummy_coro = getattr(dummy_task, "coro", None)
            if dummy_coro is not None:
                dummy_coro.close()
            await real_tasks[0]

        self.assertEqual(sleeps, [0.2])
        maybe_mock.assert_called()
        broadcast_mock.assert_awaited_once()
        self.assertIsNone(rpg._RESET_CHECK_TASK)

        await asyncio.gather(*real_tasks, return_exceptions=True)


class GetDevTextInspectTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self.request_snapshot = {"req": 1}
        self.response_snapshot = {"resp": 2}
        self.turn_request = {"scenario_req": 3}
        self.turn_response = {"scenario_resp": 4}

    def tearDown(self) -> None:
        reset_state()

    def test_returns_deep_copied_snapshots(self) -> None:
        with TestClient(rpg.app) as client:
            rpg.game_state.last_text_request = self.request_snapshot.copy()
            rpg.game_state.last_text_response = self.response_snapshot.copy()
            rpg.game_state.last_turn_request = self.turn_request.copy()
            rpg.game_state.last_turn_response = self.turn_response.copy()
            rpg.game_state.settings["history_mode"] = "SUMMARY"

            response = client.get("/api/dev/text_inspect")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request"], self.request_snapshot)
        self.assertEqual(payload["response"], self.response_snapshot)
        self.assertEqual(payload["turn_request"], self.turn_request)
        self.assertEqual(payload["turn_response"], self.turn_response)
        self.assertEqual(payload["history_mode"], "summary")
        # Mutating the response should not affect the underlying state
        payload["request"]["req"] = 99
        self.assertEqual(rpg.game_state.last_text_request, self.request_snapshot)


class GenerateSceneVideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_requires_genai_package(self) -> None:
        with (
            mock.patch.object(rpg, "genai", None),
            mock.patch.object(rpg, "genai_types", None),
        ):
            with self.assertRaises(rpg.HTTPException) as excinfo:
                await rpg.generate_scene_video("prompt")
        self.assertEqual(excinfo.exception.status_code, 500)

    async def test_requires_genai_types_module(self) -> None:
        with (
            mock.patch.object(rpg, "genai", types.SimpleNamespace()),
            mock.patch.object(rpg, "genai_types", None),
        ):
            with self.assertRaises(rpg.HTTPException) as excinfo:
                await rpg.generate_scene_video("prompt")
        self.assertEqual(excinfo.exception.status_code, 500)
        self.assertIn("type definitions", excinfo.exception.detail)

    async def test_rejects_blank_prompt(self) -> None:
        with (
            mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=mock.Mock())),
            mock.patch.object(rpg, "genai_types", types.SimpleNamespace()),
            mock.patch("rpg.require_gemini_api_key", return_value="key"),
        ):
            with self.assertRaises(rpg.HTTPException) as excinfo:
                await rpg.generate_scene_video("   ")
        self.assertEqual(excinfo.exception.status_code, 400)
        self.assertIn("No image", excinfo.exception.detail)

    async def test_generates_video_and_records_usage(self) -> None:
        class DummyVideoFile:
            def save(self, path: str) -> None:
                Path(path).write_bytes(b"mp4")

        class DummyOperation:
            def __init__(self) -> None:
                self.done = False
                self.error = None
                self.response = types.SimpleNamespace(
                    generated_videos=[types.SimpleNamespace(video=DummyVideoFile())]
                )

        class DummyClient:
            instance: Optional["DummyClient"] = None

            def __init__(self, api_key: str):
                self.api_key = api_key
                self._operation = DummyOperation()
                self.models = types.SimpleNamespace(generate_videos=self._generate_videos)
                self.operations = types.SimpleNamespace(get=self._get_operation)
                self.files = types.SimpleNamespace(download=lambda file: None)
                DummyClient.instance = self

            def _generate_videos(self, **kwargs):
                self.last_kwargs = kwargs
                return self._operation

            def _get_operation(self, operation):
                operation.done = True
                return operation

        class DummyImage:
            def __init__(self, image_bytes: bytes, mime_type: str) -> None:
                self.image_bytes = image_bytes
                self.mime_type = mime_type

        async def to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        def fake_sleep(_: float) -> None:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", Path(tmp_dir)),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=DummyImage)),
                mock.patch("rpg.require_gemini_api_key", return_value="gemini-key"),
                mock.patch("rpg.record_video_usage") as usage_mock,
                mock.patch("rpg._probe_mp4_duration_seconds", return_value=4.2),
                mock.patch("rpg.asyncio.to_thread", side_effect=to_thread),
                mock.patch("rpg.time.sleep", side_effect=fake_sleep),
            ):
                result = await rpg.generate_scene_video(
                    "Create a scene",
                    model="veo",
                    image_data_url="data:image/png;base64,QUJD",
                    negative_prompt="no rain",
                    turn_index=7,
                )
                self.assertIsInstance(result, rpg.SceneVideo)
                self.assertTrue(Path(result.file_path).exists())
                self.assertEqual(result.prompt, "Create a scene")
                self.assertEqual(result.negative_prompt, "no rain")
                self.assertEqual(result.model, "veo")
                usage_mock.assert_called_once_with("veo", seconds=4.2, turn_index=7)
        client = DummyClient.instance
        self.assertIsNotNone(client)
        if client is not None:
            kwargs = getattr(client, "last_kwargs", {})
            self.assertEqual(kwargs.get("model"), "veo")
            self.assertEqual(kwargs.get("prompt"), "Create a scene")
            self.assertEqual(kwargs.get("config"), {"negative_prompt": "no rain"})

    async def test_wraps_runtime_errors_from_generation(self) -> None:
        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = types.SimpleNamespace(message="server exploded")
                self.response = types.SimpleNamespace(generated_videos=[object()])

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_videos=lambda **kwargs: DummyOperation())
                self.operations = types.SimpleNamespace(get=lambda op: op)
                self.files = types.SimpleNamespace(download=lambda file: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir) / "scenes"
            scene_dir.mkdir()
            with (
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", scene_dir),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=lambda *args, **kwargs: None)),
                mock.patch("rpg.require_gemini_api_key", return_value="key"),
                mock.patch("rpg.asyncio.to_thread", side_effect=lambda func, *a, **kw: func()),
            ):
                with self.assertRaises(rpg.HTTPException) as excinfo:
                    await rpg.generate_scene_video("Prompt")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertEqual(excinfo.exception.detail, "server exploded")

    async def test_raises_when_no_video_returned(self) -> None:
        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = None
                self.response = types.SimpleNamespace(generated_videos=[])

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_videos=lambda **kwargs: DummyOperation())
                self.operations = types.SimpleNamespace(get=lambda op: op)
                self.files = types.SimpleNamespace(download=lambda file: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir) / "scenes"
            scene_dir.mkdir()
            with (
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", scene_dir),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=lambda *args, **kwargs: None)),
                mock.patch("rpg.require_gemini_api_key", return_value="key"),
                mock.patch("rpg.asyncio.to_thread", side_effect=lambda func, *a, **kw: func()),
            ):
                with self.assertRaises(rpg.HTTPException) as excinfo:
                    await rpg.generate_scene_video("Prompt")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("No video returned", excinfo.exception.detail)

    async def test_wraps_unexpected_exception(self) -> None:
        class DummyVideo:
            def save(self, path: str) -> None:
                Path(path).write_bytes(b"vid")

        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = None
                self.response = types.SimpleNamespace(
                    generated_videos=[types.SimpleNamespace(video=DummyVideo())]
                )

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_videos=lambda **kwargs: DummyOperation())
                self.operations = types.SimpleNamespace(get=lambda op: op)
                def _download_raise(*_args: Any, **_kwargs: Any) -> None:
                    raise ValueError("download failed")

                self.files = types.SimpleNamespace(download=_download_raise)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir) / "scenes"
            scene_dir.mkdir()
            with (
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", scene_dir),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=lambda *args, **kwargs: None)),
                mock.patch("rpg.require_gemini_api_key", return_value="key"),
                mock.patch("rpg.asyncio.to_thread", side_effect=lambda func, *a, **kw: func()),
            ):
                with self.assertRaises(rpg.HTTPException) as excinfo:
                    await rpg.generate_scene_video("Prompt")

        self.assertEqual(excinfo.exception.status_code, 502)
        self.assertIn("download failed", excinfo.exception.detail)

    async def test_url_path_uses_static_when_under_app_static(self) -> None:
        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = None
                class _Video:
                    @staticmethod
                    def save(path: str) -> None:
                        Path(path).write_bytes(b"vid")

                self.response = types.SimpleNamespace(
                    generated_videos=[types.SimpleNamespace(video=_Video())]
                )

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_videos=lambda **kwargs: DummyOperation())
                self.operations = types.SimpleNamespace(get=lambda op: op)
                self.files = types.SimpleNamespace(download=lambda file: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            app_dir = Path(tmp_dir) / "app"
            static_dir = app_dir / "static"
            scene_dir = static_dir / "videos"
            scene_dir.mkdir(parents=True)

            with (
                mock.patch.object(rpg, "APP_DIR", app_dir),
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", scene_dir),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=lambda *args, **kwargs: None)),
                mock.patch("rpg.require_gemini_api_key", return_value="key"),
                mock.patch("rpg.record_video_usage"),
                mock.patch("rpg._probe_mp4_duration_seconds", return_value=1.0),
                mock.patch("rpg.asyncio.to_thread", side_effect=lambda func, *a, **kw: func()),
                mock.patch("rpg.secrets.token_hex", return_value="abcd"),
                mock.patch("rpg.time.time", side_effect=[1000.0, 1001.0]),
            ):
                video = await rpg.generate_scene_video("Prompt")

        self.assertEqual(video.url, "/static/videos/scene_1000_abcd.mp4")

    async def test_url_path_falls_back_to_generated_media(self) -> None:
        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = None
                class _Video:
                    @staticmethod
                    def save(path: str) -> None:
                        Path(path).write_bytes(b"vid")

                self.response = types.SimpleNamespace(
                    generated_videos=[types.SimpleNamespace(video=_Video())]
                )

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_videos=lambda **kwargs: DummyOperation())
                self.operations = types.SimpleNamespace(get=lambda op: op)
                self.files = types.SimpleNamespace(download=lambda file: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            app_dir = Path(tmp_dir) / "app"
            static_dir = app_dir / "static"
            generated_dir = Path(tmp_dir) / "generated"
            scene_dir = Path(tmp_dir) / "other"
            static_dir.mkdir(parents=True)
            generated_dir.mkdir(parents=True)
            scene_dir.mkdir(parents=True)

            with (
                mock.patch.object(rpg, "APP_DIR", app_dir),
                mock.patch.object(rpg, "GENERATED_MEDIA_DIR", generated_dir),
                mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=DummyClient)),
                mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=lambda *args, **kwargs: None)),
                mock.patch("rpg.require_gemini_api_key", return_value="key"),
                mock.patch("rpg.record_video_usage"),
                mock.patch("rpg._probe_mp4_duration_seconds", return_value=1.0),
                mock.patch("rpg.asyncio.to_thread", side_effect=lambda func, *a, **kw: func()),
                mock.patch("rpg.secrets.token_hex", return_value="abcd"),
                mock.patch("rpg.time.time", side_effect=[1000.0, 1001.0]),
            ):
                video = await rpg.generate_scene_video("Prompt")

        self.assertTrue(video.url.startswith("/generated_media/scene_1000_abcd"))


class AnimateSceneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "Prompt"
        rpg.game_state.last_image_data_url = "data:image/png;base64,QUJD"
        player = rpg.Player(id="p1", name="Hero", background="", token="tok")
        rpg.game_state.players[player.id] = player

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_requires_available_prompt(self) -> None:
        rpg.game_state.last_image_prompt = ""
        with self.assertRaises(rpg.HTTPException) as excinfo:
            await rpg.animate_scene(rpg.AnimateSceneBody(player_id="p1", token="tok"))
        self.assertEqual(excinfo.exception.status_code, 400)

    async def test_generates_scene_video(self) -> None:
        video = rpg.SceneVideo(
            url="/generated_media/scene.mp4",
            prompt="Prompt",
            negative_prompt=None,
            model="veo",
            updated_at=1.0,
            file_path="/tmp/scene.mp4",
        )
        with (
            mock.patch("rpg.authenticate_player", side_effect=lambda pid, token: rpg.game_state.players[pid]),
            mock.patch("rpg.generate_scene_video", return_value=video, new_callable=mock.AsyncMock),
            mock.patch("rpg._clear_scene_video"),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
        ):
            await rpg.animate_scene(rpg.AnimateSceneBody(player_id="p1", token="tok"))
        self.assertEqual(rpg.game_state.scene_video, video)
        self.assertEqual(rpg.game_state.last_scene_video_turn_index, 0)


class MaybeQueueSceneVideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.last_image_prompt = "Image prompt"
        rpg.game_state.lock = rpg.LockState(active=False, reason="")

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_skips_when_auto_video_disabled(self) -> None:
        rpg.game_state.auto_video_enabled = False
        await rpg.schedule_auto_scene_video("Prompt", 1)
        self.assertIsNone(rpg.game_state.scene_video)

    async def test_runs_worker_and_updates_state(self) -> None:
        new_video = rpg.SceneVideo(
            url="/generated_media/v.mp4",
            prompt="Prompt",
            negative_prompt="neg",
            model="veo",
            updated_at=2.0,
            file_path="/tmp/video.mp4",
        )

        tasks: list[asyncio.Task] = []
        loop = asyncio.get_running_loop()

        def create_task_side_effect(coro):
            task = loop.create_task(coro)
            tasks.append(task)
            return task

        async def fake_sleep(_: float) -> None:
            return None

        rpg.game_state.last_image_prompt = None
        rpg.game_state.last_video_prompt = None

        with (
            mock.patch("rpg.generate_scene_video", new=mock.AsyncMock(return_value=new_video)) as gen_mock,
            mock.patch("rpg._clear_scene_video"),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
            mock.patch("rpg.asyncio.create_task", side_effect=create_task_side_effect),
            mock.patch("rpg.asyncio.sleep", side_effect=fake_sleep),
        ):
            await rpg.schedule_auto_scene_video("Prompt", 3, negative_prompt="neg")
            await asyncio.gather(*tasks)

        gen_mock.assert_awaited_once()
        self.assertEqual(rpg.game_state.scene_video, new_video)
        self.assertEqual(rpg.game_state.last_video_prompt, "Prompt")
        self.assertEqual(rpg.game_state.last_video_negative_prompt, "neg")
        self.assertEqual(rpg.game_state.last_scene_video_turn_index, 3)
