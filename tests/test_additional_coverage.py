"""Additional tests to cover critical branches that were previously untested."""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import struct
import types
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest
from fastapi import HTTPException

import rpg
from tests.test_rpg import reset_state


@pytest.fixture(autouse=True)
def _reset_state_between_tests() -> None:
    reset_state()
    rpg.reset_session_progress()
    yield
    reset_state()
    rpg.reset_session_progress()


# ---------------------------------------------------------------------------
# Media queue workers
# ---------------------------------------------------------------------------


def test_schedule_auto_scene_image_aborts_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")
        rpg.game_state.last_image_prompt = "Forest glade"
        rpg.game_state.last_scene_image_turn_index = None

        attempts: list[float] = []

        async def fake_sleep(delay: float) -> None:
            attempts.append(delay)

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        stderr_buffer = io.StringIO()

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "sleep", fake_sleep)
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg, "gemini_generate_image", mock.AsyncMock(side_effect=AssertionError("should not run")))
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())
            m.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

            await rpg.schedule_auto_scene_image("  Forest glade  ", turn_index=7)
            assert created, "worker task should be scheduled"
            await created[0]

        # The worker should retry the lock 20 times before aborting.
        assert attempts.count(0.5) == 20
        assert "Auto image generation aborted" in stderr_buffer.getvalue()
        # The lock stays busy as the attempt was abandoned without generating imagery.
        assert rpg.game_state.lock.active is True

    asyncio.run(scenario())


def test_schedule_auto_scene_image_http_error_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "Castle"

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        stderr_buffer = io.StringIO()

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            rate_limited_error = HTTPException(status_code=429, detail="rate limited")
            m.setattr(
                rpg,
                "gemini_generate_image",
                mock.AsyncMock(side_effect=rate_limited_error),
            )
            broadcast_mock = mock.AsyncMock()
            m.setattr(rpg, "broadcast_public", broadcast_mock)
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())
            m.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

            await rpg.schedule_auto_scene_image("Castle", turn_index=2)
            assert created
            await created[0]

        assert "Auto image generation failed" in stderr_buffer.getvalue()
        assert broadcast_mock.await_count >= 2
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_handles_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_video_prompt = "Stormy seas"
        rpg.game_state.last_image_prompt = "Storm image"

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        stderr_buffer = io.StringIO()

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            failure = RuntimeError("video boom")
            m.setattr(rpg, "generate_scene_video", mock.AsyncMock(side_effect=failure))
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())
            m.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

            await rpg.schedule_auto_scene_video("Stormy seas", turn_index=4)
            assert created
            await created[0]

        assert "Auto video generation error" in stderr_buffer.getvalue()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_updates_history_when_turn_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "Lighthouse"
        rpg.game_state.history = [rpg.TurnRecord(index=11, narrative="", image_prompt="", timestamp=1.0)]

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        new_video = rpg.SceneVideo(
            url="/generated_media/video.mp4",
            prompt="Lighthouse",
            negative_prompt=None,
            model="veo-test",
            updated_at=123.0,
            file_path=str(rpg.GENERATED_MEDIA_DIR / "video.mp4"),
        )

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            m.setattr(rpg, "generate_scene_video", mock.AsyncMock(return_value=new_video))
            broadcast_mock = mock.AsyncMock()
            announce_mock = mock.AsyncMock()
            m.setattr(rpg, "broadcast_public", broadcast_mock)
            m.setattr(rpg, "announce", announce_mock)
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_video("Lighthouse", turn_index="latest")
            assert created
            await created[0]

        assert rpg.game_state.scene_video == new_video
        assert rpg.game_state.last_scene_video_turn_index == 11
        assert broadcast_mock.await_count >= 2
        announce_mock.assert_awaited_once()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ElevenLabs TTS queueing – generic exception branch
# ---------------------------------------------------------------------------


def test_schedule_auto_scene_image_exits_when_disabled_mid_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")
        rpg.game_state.last_image_prompt = "Forest"

        attempts: List[float] = []

        async def fake_sleep(delay: float) -> None:
            attempts.append(delay)
            rpg.game_state.auto_image_enabled = False

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "sleep", fake_sleep)
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg, "gemini_generate_image", mock.AsyncMock(side_effect=AssertionError("should not run")))
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_image("Forest", turn_index=5)
            assert created
            await created[0]

        assert attempts == [0.5]
        assert rpg.game_state.lock.active is True
        assert rpg.game_state.auto_image_enabled is False

    asyncio.run(scenario())


def test_schedule_auto_scene_image_returns_when_prompt_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "   "

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            gem_mock = mock.AsyncMock()
            m.setattr(rpg, "gemini_generate_image", gem_mock)
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_image("Forest", turn_index=3)
            assert created
            await created[0]

        gem_mock.assert_not_awaited()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_image_skips_when_turn_already_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        target_turn = 9
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "Forest glade"
        rpg.game_state.last_scene_image_turn_index = None

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        class _ForceRepeatLock:
            async def __aenter__(self_inner) -> None:  # noqa: ANN001
                rpg.game_state.last_scene_image_turn_index = target_turn

            async def __aexit__(self_inner, exc_type, exc, tb) -> bool:  # noqa: ANN001
                return False

        with monkeypatch.context() as m:
            m.setattr(rpg, "STATE_LOCK", _ForceRepeatLock())
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            gem_mock = mock.AsyncMock()
            m.setattr(rpg, "gemini_generate_image", gem_mock)
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_image("Forest glade", turn_index=target_turn)
            assert created
            await created[0]

        gem_mock.assert_not_awaited()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_image_logs_generic_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_image_prompt = "Mountain"

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        stderr_buffer = io.StringIO()

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            m.setattr(rpg, "gemini_generate_image", mock.AsyncMock(side_effect=RuntimeError("boom")))
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())
            m.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

            await rpg.schedule_auto_scene_image("Mountain vista", turn_index=4)
            assert created
            await created[0]

        assert "Auto image generation error" in stderr_buffer.getvalue()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_http_error_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_video_prompt = "Storm"  # ensures branch prefers saved prompt

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        stderr_buffer = io.StringIO()

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            rate_limited_video_error = HTTPException(status_code=429, detail="limit")
            m.setattr(
                rpg,
                "generate_scene_video",
                mock.AsyncMock(side_effect=rate_limited_video_error),
            )
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())
            m.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

            await rpg.schedule_auto_scene_video("Storm", turn_index=6)
            assert created
            await created[0]

        assert "Auto video generation failed" in stderr_buffer.getvalue()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_returns_when_prompt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_video_prompt = "   "
        rpg.game_state.last_image_prompt = "   "

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        with (
            mock.patch.object(rpg, "STATE_LOCK", asyncio.Lock()),
            monkeypatch.context() as m,
        ):
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            gen_mock = mock.AsyncMock()
            m.setattr(rpg, "generate_scene_video", gen_mock)
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_video("Original", turn_index=2)
            assert created
            await created[0]

        gen_mock.assert_not_awaited()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_skips_when_turn_already_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        turn_index = 8
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_video_prompt = "River"
        rpg.game_state.last_scene_video_turn_index = None

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        class _SetRepeatLock:
            async def __aenter__(self_inner) -> None:  # noqa: ANN001
                rpg.game_state.last_scene_video_turn_index = turn_index

            async def __aexit__(self_inner, exc_type, exc, tb) -> bool:  # noqa: ANN001
                return False

        with monkeypatch.context() as m:
            m.setattr(rpg, "STATE_LOCK", _SetRepeatLock())
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg.asyncio, "sleep", mock.AsyncMock())
            gen_mock = mock.AsyncMock()
            m.setattr(rpg, "generate_scene_video", gen_mock)
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_video("River", turn_index=turn_index)
            assert created
            await created[0]

        gen_mock.assert_not_awaited()
        assert rpg.game_state.lock.active is False

    asyncio.run(scenario())


def test_schedule_auto_scene_video_retries_then_quits_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.lock = rpg.LockState(active=False, reason="")
        rpg.game_state.last_video_prompt = "Sky"

        attempts: List[float] = []

        class _BusyLock:
            async def __aenter__(self_inner) -> None:  # noqa: ANN001
                rpg.game_state.lock = rpg.LockState(active=True, reason="busy")

            async def __aexit__(self_inner, exc_type, exc, tb) -> bool:  # noqa: ANN001
                return False

        async def fake_sleep(delay: float) -> None:
            attempts.append(delay)
            rpg.game_state.lock = rpg.LockState(active=False, reason="")
            rpg.game_state.auto_video_enabled = False

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        with monkeypatch.context() as m:
            m.setattr(rpg, "STATE_LOCK", _BusyLock())
            m.setattr(rpg.asyncio, "sleep", fake_sleep)
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg, "generate_scene_video", mock.AsyncMock(side_effect=AssertionError("should not run")))
            m.setattr(rpg, "broadcast_public", mock.AsyncMock())
            m.setattr(rpg, "announce", mock.AsyncMock())
            m.setattr(rpg, "_clear_scene_video", mock.Mock())

            await rpg.schedule_auto_scene_video("Sky", turn_index=10)
            assert created
            await created[0]

        assert attempts == [0.5]
        assert rpg.game_state.lock.active is False
        assert rpg.game_state.auto_video_enabled is False

    asyncio.run(scenario())


def test_schedule_auto_tts_handles_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.auto_tts_enabled = True
        rpg.game_state.settings["elevenlabs_api_key"] = "key"

        async def raise_error(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("unexpected failure")

        created: List[asyncio.Task[Any]] = []
        original_create_task = asyncio.create_task

        def capture_task(coro: Any) -> asyncio.Task[Any]:
            task = original_create_task(coro)
            created.append(task)
            return task

        error_mock = mock.AsyncMock()

        with monkeypatch.context() as m:
            m.setattr(rpg, "_elevenlabs_library_available", lambda: True)
            m.setattr(rpg.asyncio, "to_thread", mock.AsyncMock(side_effect=raise_error))
            m.setattr(rpg.asyncio, "create_task", capture_task)
            m.setattr(rpg, "_broadcast_tts_error", error_mock)
            m.setattr(rpg, "_send_json_to_sockets", mock.AsyncMock())

            await rpg.schedule_auto_tts("Narration", turn_index=9)
            assert created
            await created[0]

        error_mock.assert_awaited_once()
        args, kwargs = error_mock.await_args
        assert args[1] == 9
        assert "Unexpected ElevenLabs narration failure" in args[0]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# WebSocket broadcasting helpers
# ---------------------------------------------------------------------------


def test_send_json_to_sockets_removes_dead_clients() -> None:
    class _StubWS:
        def __init__(self, should_fail: bool = False) -> None:
            self.should_fail = should_fail
            self.sent: List[Dict[str, Any]] = []

        async def send_json(self, payload: Dict[str, Any]) -> None:
            self.sent.append(payload)
            if self.should_fail:
                raise RuntimeError("disconnected")

    async def scenario() -> None:
        payload = {"event": "state"}
        live = _StubWS()
        dead = _StubWS(should_fail=True)
        sockets: set[Any] = {live, dead}

        await rpg._send_json_to_sockets(sockets, payload)

        assert payload in live.sent
        assert dead.sent == [payload]
        assert dead not in sockets

    asyncio.run(scenario())


def test_broadcast_tts_error_forwarded_to_global_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        captured: Dict[str, Any] = {}

        async def fake_send(sockets: Any, payload: Dict[str, Any]) -> None:
            captured["sockets"] = sockets
            captured["payload"] = payload

        sockets = {object()}
        rpg.game_state.global_sockets = sockets

        with monkeypatch.context() as m:
            m.setattr(rpg, "_send_json_to_sockets", fake_send)
            m.setattr(rpg.time, "time", lambda: 123.0)

            await rpg._broadcast_tts_error("Narration failed", turn_index=11)

        payload = captured["payload"]
        assert captured["sockets"] is sockets
        assert payload["event"] == "tts_error"
        assert payload["data"]["turn_index"] == 11
        assert payload["data"]["message"] == "Narration failed"
        assert payload["data"]["ts"] == 123.0

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# MP4 duration parsing helpers
# ---------------------------------------------------------------------------


def _build_mvhd_atom(*, version: int, timescale: int, duration: int) -> bytes:
    if version == 0:
        body = bytes([0])  # version
        body += b"\x00\x00\x00"  # flags
        body += b"\x00" * 8  # creation + modification time
        body += struct.pack(">I", timescale)
        body += struct.pack(">I", duration)
    else:
        body = bytes([1])
        body += b"\x00\x00\x00"
        body += b"\x00" * 16  # 64-bit times
        body += b"\x00" * 4  # reserved block before timescale
        body += struct.pack(">I", timescale)
        body += struct.pack(">Q", duration)
        body += b"\x00" * 12  # rate, volume, and reserved fields
    atom_size = 8 + len(body)
    return struct.pack(">I4s", atom_size, b"mvhd") + body


def _wrap_in_moov(payload: bytes) -> bytes:
    atom_size = 8 + len(payload)
    return struct.pack(">I4s", atom_size, b"moov") + payload


def test_read_mp4_duration_version0() -> None:
    mvhd = _build_mvhd_atom(version=0, timescale=1000, duration=4500)
    data = _wrap_in_moov(mvhd)
    stream = io.BytesIO(data)
    result = rpg._read_mp4_duration_from_stream(stream)
    assert result == pytest.approx(4.5)


def test_read_mp4_duration_version1() -> None:
    mvhd = _build_mvhd_atom(version=1, timescale=600, duration=1800)
    data = _wrap_in_moov(mvhd)
    stream = io.BytesIO(data)
    result = rpg._read_mp4_duration_from_stream(stream)
    assert result == pytest.approx(3.0)


def test_read_mp4_duration_truncated_returns_none() -> None:
    mvhd = struct.pack(">I4s", 12, b"mvhd") + b"\x00" * 4
    data = _wrap_in_moov(mvhd)
    stream = io.BytesIO(data)
    assert rpg._read_mp4_duration_from_stream(stream) is None


# ---------------------------------------------------------------------------
# OpenAI structured responses – response parsing fallbacks
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        payload: Optional[Dict[str, Any]] = None,
        text: Any = "",
        headers: Any = None,
    ) -> None:
        self.status_code = status
        self._payload = payload
        self._text = text
        self.headers = headers if headers is not None else {}

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    @property
    def text(self) -> Any:  # type: ignore[override]
        return self._text


@asynccontextmanager
async def _stub_async_client(response: _StubResponse):
    class _DummyClient:
        async def __aenter__(self_inner):  # noqa: ANN001
            return self_inner

        async def __aexit__(self_inner, exc_type, exc, tb):  # noqa: ANN001
            return False

        async def post(self_inner, url: str, headers: Dict[str, Any], json: Dict[str, Any]) -> _StubResponse:
            return response

    yield _DummyClient()


def test_openai_generate_structured_uses_data_field(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"

        payload = {
            "data": [
                {
                    "type": "output_text",
                    "text": json.dumps({"nar": "story", "img": "prompt", "pub": [], "upd": []}),
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            m.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: {"total_usd": 0.01})
            result = await rpg._openai_generate_structured(
                model="gpt-4.1-mini",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
                record_usage=True,
                include_thinking_budget=False,
            )

        assert result["nar"] == "story"
        assert rpg.game_state.session_request_count == 1
        assert rpg.game_state.last_cost_usd == pytest.approx(0.01)

    asyncio.run(scenario())


def test_openai_generate_structured_handles_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"

        response = _StubResponse(payload=ValueError("bad json"), text=lambda: "broken")

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._openai_generate_structured(
                    model="gpt-4.1-mini",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                    record_usage=False,
                    include_thinking_budget=False,
                )

        assert exc.value.status_code == 502
        assert "Malformed response" in str(exc.value.detail)

    asyncio.run(scenario())


def test_openai_generate_structured_rejects_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"

        payload = {"output": ["   "], "usage": {}}
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException):
                await rpg._openai_generate_structured(
                    model="gpt-4.1-mini",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                    record_usage=False,
                    include_thinking_budget=False,
                )

    asyncio.run(scenario())


def test_openai_generate_structured_adds_schema_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"
        requests: List[Dict[str, Any]] = []

        payload = {
            "output_text": json.dumps({"nar": "story"}),
            "usage": {},
        }
        response = _StubResponse(payload=payload)

        @asynccontextmanager
        async def _capturing_client(resp: _StubResponse):
            class _DummyClient:
                async def __aenter__(self_inner):  # noqa: ANN001
                    return self_inner

                async def __aexit__(self_inner, exc_type, exc, tb):  # noqa: ANN001
                    return False

                async def post(self_inner, url: str, headers: Dict[str, Any], json: Dict[str, Any]) -> _StubResponse:
                    requests.append({"url": url, "headers": headers, "body": json})
                    return resp

            yield _DummyClient()

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _capturing_client(response))
            m.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: None)
            result = await rpg._openai_generate_structured(
                model="o1-mini",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
                schema_name="custom",
            )

        assert result["nar"] == "story"
        body = requests[0]["body"]
        json_schema_cfg = body["response_format"]["json_schema"]
        assert json_schema_cfg["name"] == "custom"
        assert isinstance(json_schema_cfg["schema"], dict)
        assert json_schema_cfg["schema"].get("type") == "object"

    asyncio.run(scenario())


def test_openai_generate_structured_handles_none_text_and_dict_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"

        payload = {
            "output": {
                "type": "output_text",
                "text": json.dumps({"nar": "story", "img": "prompt"}),
            },
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
        response = _StubResponse(payload=payload, text=None)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            m.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: {"total_usd": 0.2})
            result = await rpg._openai_generate_structured(
                model="gpt-4.1-mini",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"
        assert rpg.game_state.session_request_count == 1

    asyncio.run(scenario())


def test_openai_generate_structured_raises_when_payload_not_object(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["openai_api_key"] = "sk-test"

        payload = {
            "output": [
                {"type": "output_text", "text": json.dumps(["not", "object"])}
            ],
            "usage": {},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._openai_generate_structured(
                    model="gpt-4.1-mini",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Malformed response" in str(exc.value.detail)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Gemini and Grok structured calls – error handling branches
# ---------------------------------------------------------------------------


def test_gemini_generate_structured_raises_for_bad_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["gemini_api_key"] = "gemini"
        response = _StubResponse(payload=None, status=200)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._gemini_generate_structured(
                    model="gemini-1.5",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Malformed" in exc.value.detail

    asyncio.run(scenario())


def test_gemini_generate_structured_handles_callable_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["gemini_api_key"] = "gem"
        rpg.game_state.settings["thinking_mode"] = "mystery"

        monkeypatch.setattr(rpg, "compute_thinking_budget", lambda model, mode: 99)
        monkeypatch.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: {"total_usd": 0.25})

        payload = {
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 6,
                "thoughtsTokenCount": 2,
            },
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps({"nar": "story", "img": "prompt"})},
                            {"text": ""},
                        ]
                    }
                }
            ],
        }

        def failing_text() -> None:
            raise ValueError("boom")

        response = _StubResponse(payload=payload, text=failing_text, headers=object())

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            result = await rpg._gemini_generate_structured(
                model="gemini-1.5",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"
        assert rpg.game_state.last_text_request["thinking_budget"] == 99
        assert rpg.game_state.session_token_usage["input"] == 4
        assert rpg.game_state.session_cost_usd == pytest.approx(0.25)

    asyncio.run(scenario())


def test_gemini_generate_structured_raises_when_json_decoder_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["gemini_api_key"] = "gemini"
        response = _StubResponse(payload=ValueError("bad json"), text=None)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._gemini_generate_structured(
                    model="gemini-1.5",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert exc.value.status_code == 502

    asyncio.run(scenario())


def test_gemini_generate_structured_sets_cost_none_when_pricing_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["gemini_api_key"] = "gem"

        payload = {
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 5,
            },
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps({"nar": "story", "img": "prompt"})}
                        ]
                    }
                }
            ],
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            m.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: None)
            result = await rpg._gemini_generate_structured(
                model="gemini-1.5",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"
        assert rpg.game_state.last_cost_usd is None

    asyncio.run(scenario())


def test_gemini_generate_structured_raises_when_payload_not_object(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["gemini_api_key"] = "gem"

        payload = {
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
            },
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(["not", "object"])}
                        ]
                    }
                }
            ],
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._gemini_generate_structured(
                    model="gemini-1.5",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Malformed" in str(exc.value.detail)

    asyncio.run(scenario())


def test_grok_generate_structured_cleans_code_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        payload = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\n  \"nar\": \"story\"\n}\n```",
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            m.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: None)
            result = await rpg._grok_generate_structured(
                model="grok-beta",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"

    asyncio.run(scenario())


def test_grok_generate_structured_coerces_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = " key "
        rpg.game_state.settings["thinking_mode"] = "unknown"

        monkeypatch.setattr(rpg, "calculate_turn_cost", lambda *args, **kwargs: {"total_usd": 0.5})

        payload = {
            "usage": {
                "prompt_tokens": "5",
                "completion_tokens": "7",
                "completion_tokens_details": {"reasoning_tokens": "11"},
            },
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": json.dumps({"nar": "story"})}
                        ]
                    }
                }
            ],
        }

        def text_none() -> None:
            return None

        response = _StubResponse(payload=payload, text=text_none)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            result = await rpg._grok_generate_structured(
                model="grok-beta",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"
        assert rpg.game_state.session_token_usage["thinking"] == 11
        assert rpg.game_state.session_cost_usd == pytest.approx(0.5)

    asyncio.run(scenario())


def test_grok_generate_structured_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        payload = {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            "choices": [
                {
                    "message": {"role": "assistant"},
                }
            ],
        }

        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Empty response" in str(exc.value.detail)

    asyncio.run(scenario())


def test_grok_generate_structured_handles_json_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        response = _StubResponse(payload=ValueError("bad json"))

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException):
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

    asyncio.run(scenario())


def test_grok_generate_structured_reads_string_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        payload = {
            "choices": [
                {
                    "content": json.dumps({"nar": "story", "img": "prompt"})
                }
            ],
            "usage": {},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            result = await rpg._grok_generate_structured(
                model="grok-beta",
                system_prompt="SYS",
                user_payload={"foo": "bar"},
                schema={"type": "object"},
            )

        assert result["nar"] == "story"

    asyncio.run(scenario())


def test_grok_generate_structured_reports_cleaned_json_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "```json\nnot valid json\n```"}
                        ]
                    }
                }
            ],
            "usage": {},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Malformed response" in str(exc.value.detail)

    asyncio.run(scenario())


def test_grok_generate_structured_raises_when_result_not_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings["grok_api_key"] = "grok"

        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "[1, 2, 3]"}
                        ]
                    }
                }
            ],
            "usage": {},
        }
        response = _StubResponse(payload=payload)

        with monkeypatch.context() as m:
            m.setattr(rpg.httpx, "AsyncClient", lambda *args, **kwargs: _stub_async_client(response))
            with pytest.raises(HTTPException) as exc:
                await rpg._grok_generate_structured(
                    model="grok-beta",
                    system_prompt="SYS",
                    user_payload={"foo": "bar"},
                    schema={"type": "object"},
                )

        assert "Malformed response" in str(exc.value.detail)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Settings API and model listings
# ---------------------------------------------------------------------------


def test_update_settings_rejects_invalid_thinking_mode() -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPException):
            await rpg.update_settings(rpg.SettingsUpdate(thinking_mode="impossible"))

    asyncio.run(scenario())


def test_update_settings_resets_elevenlabs_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = True
        saved: Dict[str, Any] = {}

        async def fake_save(settings: Dict[str, Any]) -> None:
            saved.update(settings)

        monkeypatch.setattr(rpg, "save_settings", fake_save)

        body = rpg.SettingsUpdate(elevenlabs_api_key="  new-key  ")
        result = await rpg.update_settings(body)
        assert result == {"ok": True}
        assert saved["elevenlabs_api_key"] == "new-key"
        assert rpg._ELEVENLABS_API_KEY_WARNING_LOGGED is False

    asyncio.run(scenario())


def test_api_models_raises_when_all_providers_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings.update({
            "gemini_api_key": "gem",
            "grok_api_key": "grok",
            "openai_api_key": "openai",
            "elevenlabs_api_key": "tts",
        })

        error = HTTPException(status_code=401, detail="bad key")

        async def raise_error(*args: Any, **kwargs: Any):
            raise error

        monkeypatch.setattr(rpg, "gemini_list_models", raise_error)
        monkeypatch.setattr(rpg, "grok_list_models", raise_error)
        monkeypatch.setattr(rpg, "openai_list_models", raise_error)
        monkeypatch.setattr(rpg, "elevenlabs_list_models", raise_error)

        with pytest.raises(HTTPException) as exc:
            await rpg.api_models()

        assert exc.value is error

    asyncio.run(scenario())


def test_api_models_combines_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.settings.update({
            "gemini_api_key": "gem",
            "grok_api_key": "",
            "openai_api_key": "openai",
            "elevenlabs_api_key": "tts",
        })

        async def gemini_models() -> List[Dict[str, Any]]:
            return [{"name": "model-a", "displayName": "Model A"}]

        async def openai_models(key: str) -> List[Dict[str, Any]]:
            return [{"name": "model-b", "displayName": "Model B"}]

        async def eleven_models(api_key: str) -> List[Dict[str, Any]]:
            return [{"id": "voice-1", "name": "Voice"}]

        monkeypatch.setattr(rpg, "gemini_list_models", lambda: gemini_models())
        monkeypatch.setattr(rpg, "openai_list_models", openai_models)
        monkeypatch.setattr(rpg, "elevenlabs_list_models", eleven_models)

        result = await rpg.api_models()
        assert {m["displayName"] for m in result["models"]} == {"Model A", "Model B"}
        assert result["narration_models"][0]["id"] == "voice-1"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# History summary fallback logic
# ---------------------------------------------------------------------------


def test_update_history_summary_uses_fallback_when_not_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.history = [
            rpg.TurnRecord(index=1, narrative="A brave act.", image_prompt="", timestamp=0.0),
            rpg.TurnRecord(index=2, narrative="Another story.", image_prompt="", timestamp=1.0),
        ]
        rpg.game_state.history_summary = []
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_FULL

        latest = rpg.game_state.history[-1]
        await rpg.update_history_summary(latest)
        assert rpg.game_state.history_summary
        assert rpg.game_state.history_summary[0].startswith("- Turn")

    asyncio.run(scenario())


def test_update_history_summary_falls_back_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.history = [
            rpg.TurnRecord(index=3, narrative="", image_prompt="", timestamp=0.0)
        ]
        rpg.game_state.history_summary = []
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        missing_key_error = HTTPException(status_code=401, detail="missing")
        monkeypatch.setattr(
            rpg,
            "require_text_api_key",
            mock.Mock(side_effect=missing_key_error),
        )

        latest = rpg.game_state.history[-1]
        await rpg.update_history_summary(latest)
        assert rpg.game_state.history_summary == ["- Turn 3: No narrative recorded."]

    asyncio.run(scenario())


def test_update_history_summary_truncates_long_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.history = [
            rpg.TurnRecord(index=1, narrative="A bold tale.", image_prompt="", timestamp=0.0),
            rpg.TurnRecord(index=2, narrative="Another moment.", image_prompt="", timestamp=1.0),
        ]
        rpg.game_state.history_summary = []
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY
        rpg.game_state.turn_index = 2
        rpg.game_state.players.clear()
        rpg.game_state.players["p1"] = rpg.Player(
            id="p1",
            name="Alice",
            background="Mage",
            character_class="Wizard",
            inventory=["Staff"],
        )

        monkeypatch.setattr(rpg, "require_text_api_key", lambda provider: None)

        overlong = [f"line {i}" for i in range(rpg.MAX_HISTORY_SUMMARY_BULLETS + 3)]
        overlong.insert(1, 42)

        async def fake_summary(*args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(summary=overlong)

        monkeypatch.setattr(rpg, "request_summary_payload", fake_summary)

        latest = rpg.game_state.history[-1]
        await rpg.update_history_summary(latest)

        assert len(rpg.game_state.history_summary) == rpg.MAX_HISTORY_SUMMARY_BULLETS
        assert all(line.startswith("- ") for line in rpg.game_state.history_summary)
        assert "line 0" in rpg.game_state.history_summary[0]

    asyncio.run(scenario())


def test_update_history_summary_uses_fallback_when_model_returns_no_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.history = [
            rpg.TurnRecord(index=7, narrative="", image_prompt="", timestamp=0.0)
        ]
        rpg.game_state.history_summary = []
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        monkeypatch.setattr(rpg, "require_text_api_key", lambda provider: None)

        async def fake_summary(*args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(summary=[None, 3])

        monkeypatch.setattr(rpg, "request_summary_payload", fake_summary)

        latest = rpg.game_state.history[-1]
        await rpg.update_history_summary(latest)

        assert rpg.game_state.history_summary == ["- Turn 7: No narrative recorded."]

    asyncio.run(scenario())


def test_update_history_summary_logs_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        rpg.game_state.history = [
            rpg.TurnRecord(index=4, narrative="A setback.", image_prompt="", timestamp=0.0)
        ]
        rpg.game_state.history_summary = []
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        monkeypatch.setattr(rpg, "require_text_api_key", lambda provider: None)

        async def raise_error(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("summary boom")

        monkeypatch.setattr(rpg, "request_summary_payload", raise_error)

        stderr_buffer = io.StringIO()
        monkeypatch.setattr(rpg.sys, "stderr", stderr_buffer, raising=False)

        latest = rpg.game_state.history[-1]
        await rpg.update_history_summary(latest)

        assert "History summary update failed" in stderr_buffer.getvalue()
        assert rpg.game_state.history_summary == ["- Turn 4: A setback."]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Image usage accounting
# ---------------------------------------------------------------------------


def test_record_image_usage_without_pricing() -> None:
    rpg.game_state.current_turn_image_counts = {}
    rpg.game_state.current_turn_index_for_image_counts = None
    rpg.game_state.session_image_kind_counts = {}
    rpg.record_image_usage(None, purpose="other", images=2, turn_index=None)
    assert rpg.game_state.session_image_requests == 2
    assert rpg.game_state.current_turn_image_counts["other"] == 2


def test_project_root_points_to_repo_root() -> None:
    path_utils = importlib.import_module("scripts.path_utils")
    root = path_utils.project_root().resolve()
    assert root == rpg.APP_DIR.resolve()
    readme = path_utils.resolve_repo_path("README.md")
    assert readme == root / "README.md"
