import asyncio
import base64
import builtins
import hashlib
import io
import json
import os
import runpy
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

import rpg


def reset_state():
    """Utility to return the global game state to a clean baseline for tests."""
    rpg.game_state.settings = rpg.DEFAULT_SETTINGS.copy()
    rpg.game_state.players.clear()
    rpg.game_state.departed_players.clear()
    rpg.game_state.submissions.clear()
    rpg.game_state.current_narrative = ""
    rpg.game_state.turn_index = 0
    rpg.game_state.history.clear()
    rpg.game_state.history_summary.clear()
    rpg.game_state.lock = rpg.LockState(active=False, reason="")
    rpg.game_state.global_sockets.clear()
    rpg.game_state.language = rpg.DEFAULT_LANGUAGE
    rpg.game_state.last_image_data_url = None
    rpg.game_state.last_image_prompt = None
    rpg.game_state.last_video_prompt = None
    rpg.game_state.last_video_negative_prompt = None
    rpg.game_state.scene_video = None
    rpg.game_state.last_scene_video_turn_index = None
    rpg.game_state.last_video_model = None
    rpg.game_state.last_video_tier = None
    rpg.game_state.last_video_cost_usd = None
    rpg.game_state.last_video_usd_per_second = None
    rpg.game_state.last_video_seconds = None
    rpg.game_state.last_video_turn_index = None
    rpg.game_state.session_video_cost_usd = 0.0
    rpg.game_state.session_video_seconds = 0.0
    rpg.game_state.session_video_requests = 0
    rpg.game_state.last_image_model = None
    rpg.game_state.last_image_tier = None
    rpg.game_state.last_image_kind = None
    rpg.game_state.last_image_cost_usd = None
    rpg.game_state.last_image_usd_per_image = None
    rpg.game_state.last_image_tokens = None
    rpg.game_state.last_image_count = 0
    rpg.game_state.last_image_turn_index = None
    rpg.game_state.last_token_usage = {}
    rpg.game_state.last_turn_runtime = None
    rpg.game_state.session_token_usage = {"input": 0, "output": 0, "thinking": 0}
    rpg.game_state.session_request_count = 0
    rpg.game_state.last_cost_usd = None
    rpg.game_state.session_cost_usd = 0.0
    rpg.game_state.session_image_cost_usd = 0.0
    rpg.game_state.session_image_requests = 0
    rpg.game_state.session_image_kind_counts = {}
    rpg.game_state.last_scene_image_cost_usd = None
    rpg.game_state.last_scene_image_usd_per_image = None
    rpg.game_state.last_scene_image_model = None
    rpg.game_state.last_scene_image_turn_index = None
    rpg.game_state.current_turn_image_counts = {}
    rpg.game_state.current_turn_index_for_image_counts = None
    rpg.game_state.image_counts_by_turn = {}
    rpg.game_state.auto_image_enabled = False
    rpg.game_state.auto_video_enabled = False
    rpg.game_state.auto_tts_enabled = False
    rpg.game_state.last_tts_model = None
    rpg.game_state.last_tts_voice_id = None
    rpg.game_state.last_tts_request_id = None
    rpg.game_state.last_tts_characters = None
    rpg.game_state.last_tts_character_source = None
    rpg.game_state.last_tts_credits = None
    rpg.game_state.last_tts_cost_usd = None
    rpg.game_state.last_tts_headers = {}
    rpg.game_state.last_tts_turn_index = None
    rpg.game_state.last_tts_total_credits = None
    rpg.game_state.last_tts_remaining_credits = None
    rpg.game_state.last_tts_next_reset_unix = None
    rpg.game_state.session_tts_characters = 0
    rpg.game_state.session_tts_credits = 0.0
    rpg.game_state.session_tts_cost_usd = 0.0
    rpg.game_state.session_tts_requests = 0


def load_rpg_main_namespace() -> Dict[str, Any]:
    """Execute rpg.py as __main__ to expose CLI helpers for testing."""

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *args, **kwargs: None

    uvicorn_config = types.ModuleType("uvicorn.config")

    class DummyConfig:
        def __init__(self, *args, **kwargs) -> None:
            self.port = kwargs.get("port", 8000)
            self.backlog = kwargs.get("backlog", 2048)
            self.is_ssl = False
            self.should_reload = False
            self.workers = 1

    uvicorn_config.Config = DummyConfig

    uvicorn_server = types.ModuleType("uvicorn.server")

    class DummyServer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs) -> None:
            return None

    uvicorn_server.Server = DummyServer

    uvicorn_supervisors = types.ModuleType("uvicorn.supervisors")

    class DummyChangeReload:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self) -> None:
            return None

    class DummyMultiprocess:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self) -> None:
            return None

    uvicorn_supervisors.ChangeReload = DummyChangeReload
    uvicorn_supervisors.Multiprocess = DummyMultiprocess

    fake_uvicorn.config = uvicorn_config
    fake_uvicorn.server = uvicorn_server
    fake_uvicorn.supervisors = uvicorn_supervisors

    modules = {
        "uvicorn": fake_uvicorn,
        "uvicorn.config": uvicorn_config,
        "uvicorn.server": uvicorn_server,
        "uvicorn.supervisors": uvicorn_supervisors,
    }

    with mock.patch.dict(sys.modules, modules, clear=False), \
            mock.patch.dict(os.environ, {"ORPG_HOST": "127.0.0.1"}, clear=False):
        return runpy.run_module("rpg", run_name="__main__", alter_sys=True)


def reset_public_ip_cache():
    rpg.PUBLIC_IP_CACHE.clear()
    rpg.PUBLIC_IP_CACHE.update({"ip": None, "cached_at": 0.0, "last_failure_at": 0.0})


class SanitizeNarrativeEdgeTests(unittest.TestCase):
    def test_decodes_unicode_escape_sequences(self):
        raw = "Die H\\u00fctte steht am Flu\\u00dfufer."
        cleaned = rpg.sanitize_narrative(raw)
        self.assertEqual(cleaned, "Die Hütte steht am Flussufer.")


class IndexRouteTests(unittest.TestCase):
    def test_serves_static_index_when_present(self):
        client = TestClient(rpg.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.strip())

    def test_falls_back_to_placeholder_when_missing(self):
        client = TestClient(rpg.app)
        original_app_dir = rpg.APP_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                rpg.APP_DIR = Path(tmp_dir)
                response = client.get("/")
        finally:
            rpg.APP_DIR = original_app_dir
        self.assertEqual(response.status_code, 200)
        self.assertIn("UI not found", response.text)


class StaticAssetRouteTests(unittest.TestCase):
    def test_serves_favicon_when_present(self):
        client = TestClient(rpg.app)
        response = client.get("/favicon.ico")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.headers["content-type"], {"image/x-icon", "image/vnd.microsoft.icon"})
        self.assertTrue(response.content)

    def test_join_backgrounds_enumerates_supported_images(self):
        original_static_dir = rpg.static_dir
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                images_dir = tmp_path / "img"
                images_dir.mkdir()
                (images_dir / "alpha.jpg").write_bytes(b"jpg")
                (images_dir / "beta.png").write_bytes(b"png")
                (images_dir / "ignore.txt").write_text("skip")
                rpg.static_dir = tmp_path
                with TestClient(rpg.app) as client:
                    response = client.get("/api/join_backgrounds")
        finally:
            rpg.static_dir = original_static_dir

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backgrounds"], ["/static/img/alpha.jpg", "/static/img/beta.png"])

    def test_join_songs_lists_mp3_metadata(self):
        original_static_dir = rpg.static_dir
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                songs_dir = tmp_path / "songs"
                songs_dir.mkdir()
                (songs_dir / "adventure.mp3").write_bytes(b"mp3")
                (songs_dir / "heroic-theme.mp3").write_bytes(b"mp3")
                (songs_dir / "skip.wav").write_bytes(b"wav")
                rpg.static_dir = tmp_path
                with TestClient(rpg.app) as client:
                    response = client.get("/api/join_songs")
        finally:
            rpg.static_dir = original_static_dir

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["songs"],
            [
                {"id": "adventure", "src": "/static/songs/adventure.mp3", "title": "Adventure"},
                {
                    "id": "heroic-theme",
                    "src": "/static/songs/heroic-theme.mp3",
                    "title": "Heroic Theme",
                },
            ],
        )


class GeneratedMediaArchiveTests(unittest.TestCase):
    def test_archive_generated_media_writes_file_with_sanitized_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_dir = rpg.GENERATED_MEDIA_DIR
            rpg.GENERATED_MEDIA_DIR = Path(tmp_dir)
            try:
                path = rpg._archive_generated_media(b"data", prefix="Sparks & Dust", suffix="png")
                self.assertTrue(path.exists())
                self.assertTrue(path.name.startswith("sparks_dust_"))
                self.assertTrue(path.name.endswith(".png"))
                self.assertEqual(path.read_bytes(), b"data")
            finally:
                rpg.GENERATED_MEDIA_DIR = original_dir

    def test_sanitize_media_prefix_falls_back_to_default(self):
        self.assertEqual(rpg._sanitize_media_prefix("!!!"), "media")
        self.assertEqual(rpg._sanitize_media_prefix("Arcane Gate"), "arcane_gate")


class SupportedListNormalizationTests(unittest.TestCase):
    def test_normalize_supported_list_filters_falsey_entries(self):
        raw = {"generateContent": True, "responses": 1, "batch": 0, "stream": ""}

        normalized = rpg._normalize_supported_list(raw)

        self.assertEqual(normalized, ["generateContent", "responses"])

    def test_normalize_supported_list_uses_fallback_and_deduplicates(self):
        normalized = rpg._normalize_supported_list([], fallback=["Chat", "chat", None, "Batch"])

        self.assertEqual(normalized, ["Chat", "Batch"])


class MimeSuffixTests(unittest.TestCase):
    def test_suffix_from_mime_maps_common_types(self):
        self.assertEqual(rpg._suffix_from_mime("image/png"), ".png")
        self.assertEqual(rpg._suffix_from_mime(" image/jpeg "), ".jpg")

    def test_suffix_from_mime_handles_unknown_and_missing_values(self):
        self.assertEqual(
            rpg._suffix_from_mime("application/x-custom+json; charset=UTF-8"),
            ".x-custom+json",
        )
        self.assertEqual(rpg._suffix_from_mime(None), ".bin")


def _mp4_atom(atom_type: bytes, payload: bytes) -> bytes:
    size = len(payload) + 8
    return struct.pack('>I4s', size, atom_type) + payload


def _make_valid_mp4_bytes(*, timescale: int = 1_000, duration: int = 5_000) -> bytes:
    mvhd_payload = struct.pack('>B3sIIII', 0, b'\x00\x00\x00', 0, 0, timescale, duration)
    mvhd_atom = _mp4_atom(b'mvhd', mvhd_payload)
    moov_atom = _mp4_atom(b'moov', mvhd_atom)
    ftyp_atom = _mp4_atom(b'ftyp', b'isom0000isom')
    mdat_atom = _mp4_atom(b'mdat', b'\x00' * 16)
    return ftyp_atom + moov_atom + mdat_atom


def _make_truncated_mp4_bytes() -> bytes:
    mvhd_atom = _mp4_atom(b'mvhd', b'\x00\x00\x00\x00')
    moov_atom = _mp4_atom(b'moov', mvhd_atom)
    return moov_atom


def _make_no_moov_mp4_bytes() -> bytes:
    ftyp_atom = _mp4_atom(b'ftyp', b'isom0000isom')
    mdat_atom = _mp4_atom(b'mdat', b'\x00' * 32)
    return ftyp_atom + mdat_atom


class Mp4DurationProbeTests(unittest.TestCase):
    def test_read_mp4_duration_from_stream(self):
        mp4_bytes = _make_valid_mp4_bytes(timescale=1_000, duration=9_000)

        duration = rpg._read_mp4_duration_from_stream(io.BytesIO(mp4_bytes))

        self.assertAlmostEqual(duration, 9.0)

    def test_read_mp4_duration_returns_none_for_missing_moov(self):
        mp4_bytes = _make_no_moov_mp4_bytes()

        duration = rpg._read_mp4_duration_from_stream(io.BytesIO(mp4_bytes))

        self.assertIsNone(duration)

    def test_read_mp4_duration_returns_none_for_truncated_mvhd(self):
        mp4_bytes = _make_truncated_mp4_bytes()

        duration = rpg._read_mp4_duration_from_stream(io.BytesIO(mp4_bytes))

        self.assertIsNone(duration)

    def test_probe_mp4_duration_seconds_reads_file(self):
        mp4_bytes = _make_valid_mp4_bytes(timescale=2_000, duration=5_000)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(mp4_bytes)
            tmp_path = Path(tmp.name)

        try:
            result = rpg._probe_mp4_duration_seconds(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertAlmostEqual(result, 2.5)

    def test_probe_mp4_duration_seconds_handles_missing_file(self):
        missing_path = Path(tempfile.gettempdir()) / "nonexistent-video.mp4"

        self.assertIsNone(rpg._probe_mp4_duration_seconds(missing_path))

    def test_probe_mp4_duration_seconds_handles_truncated_content(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(_make_truncated_mp4_bytes())
            tmp_path = Path(tmp.name)

        try:
            result = rpg._probe_mp4_duration_seconds(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertIsNone(result)


class WorldStylesDataTests(unittest.TestCase):
    def test_world_styles_json_structure(self):
        world_styles_path = rpg.APP_DIR / "static" / "world-styles.json"
        raw = world_styles_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        self.assertIsInstance(data, list)
        self.assertTrue(data)
        for entry in data:
            self.assertIsInstance(entry, dict)
            self.assertIn("name", entry)
            self.assertIn("children", entry)
            self.assertIsInstance(entry["name"], str)
            self.assertTrue(entry["name"].strip())
            self.assertIsInstance(entry["children"], list)
            self.assertTrue(entry["children"])
            for child in entry["children"]:
                self.assertIsInstance(child, str)
                self.assertTrue(child.strip())


class ResolveTurnStatusWordTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_blank_status_word_becomes_unknown(self):
        pid = "player1"
        rpg.game_state.players[pid] = rpg.Player(id=pid, name="Test", background="BG", token="tok1")

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                narrative="Test scenario",
                image_prompt="prompt",
                public_statuses=[rpg.PublicStatus(player_id=pid, status_word="   ")],
                updates=[
                    rpg.PlayerUpdate(
                        player_id=pid,
                        character_class="Wizard",
                        abilities=[rpg.Ability(name="Arcana", expertise="novice")],
                        inventory=["staff"],
                        conditions=["healthy"],
                    )
                ],
            )

        original = rpg.request_turn_payload
        rpg.request_turn_payload = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.request_turn_payload = original

        player = rpg.game_state.players[pid]
        self.assertEqual(player.status_word, "unknown")
        self.assertEqual(player.character_class, "Wizard")
        self.assertEqual(rpg.game_state.turn_index, 1)

    async def test_status_word_trimmed_and_lowercased(self):
        pid = "player2"
        rpg.game_state.players[pid] = rpg.Player(id=pid, name="Other", background="BG", token="tok2")

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                narrative="Another scenario",
                image_prompt="prompt",
                public_statuses=[rpg.PublicStatus(player_id=pid, status_word="  MiGhTy Hero  ")],
                updates=[
                    rpg.PlayerUpdate(
                        player_id=pid,
                        character_class="Bard",
                        abilities=[],
                        inventory=[],
                        conditions=[],
                    )
                ],
            )

        original = rpg.request_turn_payload
        rpg.request_turn_payload = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.request_turn_payload = original

        player = rpg.game_state.players[pid]
        self.assertEqual(player.status_word, "mighty")
        self.assertEqual(player.character_class, "Bard")
        self.assertEqual(rpg.game_state.turn_index, 1)


class ResolveTurnVideoPromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_dict_video_prompt_updates_state(self) -> None:
        pid = "player-video"
        rpg.game_state.players[pid] = rpg.Player(id=pid, name="Vid", background="", token="tok")
        rpg.game_state.turn_index = 1

        turn = types.SimpleNamespace(
            narrative="Narrative",
            image_prompt="image prompt",
            video={"prompt": "  cinematic duel  ", "negativePrompt": "  less smoke  "},
            updates=[
                rpg.PlayerUpdate(
                    player_id=pid,
                    character_class="Rogue",
                    abilities=[rpg.Ability(name="Sneak", expertise="expert")],
                    inventory=["dagger"],
                    conditions=["alert"],
                ),
                rpg.PlayerUpdate(
                    player_id="unknown",
                    character_class="",
                    abilities=[],
                    inventory=[],
                    conditions=[],
                ),
            ],
            public_statuses=[],
        )

        with (
            mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(return_value=turn)),
            mock.patch("rpg.update_history_summary", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_tts", new=mock.AsyncMock()),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
            mock.patch("rpg.send_private", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_scene_image", new=mock.AsyncMock()) as image_mock,
            mock.patch("rpg.schedule_auto_scene_video", new=mock.AsyncMock()) as video_mock,
        ):
            await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.game_state.last_video_prompt, "cinematic duel")
        self.assertEqual(rpg.game_state.last_video_negative_prompt, "less smoke")
        self.assertEqual(rpg.game_state.last_image_prompt, "image prompt")
        video_mock.assert_awaited()
        img_args = image_mock.await_args.args
        self.assertEqual(img_args[0], "image prompt")
        self.assertEqual(video_mock.await_args.args[0], "cinematic duel")

    async def test_string_video_prompt_updates_state(self) -> None:
        pid = "player-video-string"
        rpg.game_state.players[pid] = rpg.Player(id=pid, name="Vid", background="", token="tok")
        rpg.game_state.turn_index = 2

        turn = types.SimpleNamespace(
            narrative="Narrative",
            image_prompt="image prompt",
            video="  sweeping vista  ",
            updates=[
                rpg.PlayerUpdate(
                    player_id=pid,
                    character_class="Mage",
                    abilities=[],
                    inventory=[],
                    conditions=[],
                )
            ],
            public_statuses=[],
        )

        with (
            mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(return_value=turn)),
            mock.patch("rpg.update_history_summary", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_tts", new=mock.AsyncMock()),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
            mock.patch("rpg.send_private", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_scene_image", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_scene_video", new=mock.AsyncMock()) as video_mock,
        ):
            await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.game_state.last_video_prompt, "sweeping vista")
        self.assertIsNone(rpg.game_state.last_video_negative_prompt)
        video_mock.assert_awaited()

    async def test_pending_leave_player_removed_after_turn(self):
        stay_id = "stay"
        leave_id = "leave"
        rpg.game_state.players[stay_id] = rpg.Player(
            id=stay_id,
            name="Stayed",
            background="BG",
            token="staytok",
            pending_join=False,
        )
        rpg.game_state.players[leave_id] = rpg.Player(
            id=leave_id,
            name="Drifter",
            background="BG",
            token="leavetok",
            pending_join=False,
            pending_leave=True,
        )

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                narrative="The drifter says farewell and walks into the mist.",
                image_prompt="farewell",
                public_statuses=[
                    rpg.PublicStatus(player_id=stay_id, status_word="ready"),
                    rpg.PublicStatus(player_id=leave_id, status_word="departed"),
                ],
                updates=[
                    rpg.PlayerUpdate(
                        player_id=stay_id,
                        character_class="Guardian",
                        abilities=[rpg.Ability(name="Watch", expertise="novice")],
                        inventory=["shield"],
                        conditions=["steady"],
                    )
                ],
            )

        original = rpg.request_turn_payload
        rpg.request_turn_payload = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.request_turn_payload = original

        self.assertIn(stay_id, rpg.game_state.players)
        self.assertNotIn(leave_id, rpg.game_state.players)
        self.assertEqual(rpg.game_state.players[stay_id].character_class, "Guardian")
        self.assertFalse(rpg.game_state.players[stay_id].pending_leave)
        self.assertEqual(rpg.game_state.turn_index, 1)
        normalized = rpg.normalize_player_name("Drifter")
        self.assertIn(normalized, rpg.game_state.departed_players)
        self.assertEqual(rpg.game_state.departed_players[normalized], "Drifter")


class ResolveTurnAdditionalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_raises_when_lock_active(self) -> None:
        rpg.game_state.lock = rpg.LockState(active=True, reason="busy")
        with self.assertRaises(HTTPException) as excinfo:
            await rpg.resolve_turn(initial=False)
        self.assertEqual(excinfo.exception.status_code, 409)

    async def test_structured_video_prompt_updates_state(self) -> None:
        class FakeTurnIndex:
            def __init__(self, value: int):
                self.value = value

            def __iadd__(self, other: int):
                self.value += other
                return self

            def __int__(self):
                return self.value

            def __repr__(self) -> str:
                return f"FakeTurnIndex({self.value})"

            def __hash__(self) -> int:
                return hash(self.value)

            def __eq__(self, other) -> bool:
                if isinstance(other, FakeTurnIndex):
                    return self.value == other.value
                if isinstance(other, int):
                    return self.value == other
                return False

        rpg.game_state.turn_index = FakeTurnIndex(0)
        rpg.game_state.players["p1"] = rpg.Player(id="p1", name="Hero", background="", token="tok", pending_join=False)

        turn_payload = rpg.TurnStructured(
            narrative="The story continues",
            image_prompt="A vivid scene",
            video=rpg.VideoPromptStructured(prompt="Video prompt", negative_prompt="No shadows"),
            public_statuses=[rpg.PublicStatus(player_id="p1", status_word="Brave")],
            updates=[
                rpg.PlayerUpdate(
                    player_id="p1",
                    character_class="Mage",
                    abilities=[],
                    inventory=[],
                    conditions=[],
                )
            ],
        )

        with (
            mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(return_value=turn_payload)),
            mock.patch("rpg.reset_session_if_inactive", return_value=False),
            mock.patch("rpg.schedule_auto_tts", new=mock.AsyncMock()) as tts_mock,
            mock.patch("rpg.schedule_auto_scene_image", new=mock.AsyncMock()) as image_mock,
            mock.patch("rpg.schedule_auto_scene_video", new=mock.AsyncMock()) as video_mock,
            mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private_mock,
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
        ):
            await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.game_state.players["p1"].character_class, "Mage")
        self.assertEqual(rpg.game_state.last_video_prompt, "Video prompt")
        self.assertEqual(rpg.game_state.last_video_negative_prompt, "No shadows")
        self.assertGreaterEqual(rpg.game_state.turn_index.value, 1)
        self.assertTrue(rpg.game_state.history)
        tts_mock.assert_awaited_once()
        image_mock.assert_awaited_once()
        video_mock.assert_awaited_once()
        args = video_mock.await_args.args
        self.assertEqual(args[0], "Video prompt")
        self.assertEqual(int(args[1]), rpg.game_state.turn_index.value)
        send_private_mock.assert_awaited()

    async def test_dict_video_prompt_fallback(self) -> None:
        rpg.game_state.players["p1"] = rpg.Player(id="p1", name="Hero", background="", token="tok", pending_join=False)

        turn_payload = rpg.TurnStructured(
            narrative="Story",
            image_prompt="Scene",
            video={"prompt": "Dict prompt", "negativePrompt": "Steady"},
            public_statuses=[],
            updates=[],
        )

        with (
            mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(return_value=turn_payload)),
            mock.patch("rpg.reset_session_if_inactive", return_value=False),
            mock.patch("rpg.schedule_auto_tts", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_scene_image", new=mock.AsyncMock()),
            mock.patch("rpg.schedule_auto_scene_video", new=mock.AsyncMock()) as video_mock,
            mock.patch("rpg.send_private", new=mock.AsyncMock()),
            mock.patch("rpg.broadcast_public", new=mock.AsyncMock()),
            mock.patch("rpg.announce", new=mock.AsyncMock()),
        ):
            await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.game_state.last_video_prompt, "Dict prompt")
        self.assertEqual(rpg.game_state.last_video_negative_prompt, "Steady")
        video_mock.assert_awaited_once()


class WebSocketEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self.client = TestClient(rpg.app)

    def tearDown(self) -> None:
        self.client.close()
        reset_state()

    def test_private_channel_disconnect_marks_player(self) -> None:
        player = rpg.Player(id="p1", name="Hero", background="", token="tok", pending_join=False)
        rpg.game_state.players[player.id] = player

        broadcast_mock = mock.AsyncMock()
        announce_mock = mock.AsyncMock()

        with (
            mock.patch("rpg.broadcast_public", new=broadcast_mock),
            mock.patch("rpg.announce", new=announce_mock),
            mock.patch("rpg.schedule_session_reset_check") as reset_mock,
            mock.patch("rpg.cancel_pending_reset_task") as cancel_mock,
        ):
            with self.client.websocket_connect(f"/ws?player_id={player.id}&auth_token={player.token}") as ws:
                initial = ws.receive_json()
                self.assertEqual(initial["event"], "state")
                private = ws.receive_json()
                self.assertEqual(private["event"], "private")
                self.assertTrue(player.connected)

        self.assertFalse(rpg.game_state.global_sockets)
        self.assertFalse(player.connected)
        self.assertTrue(player.pending_leave)
        broadcast_mock.assert_awaited()
        announce_mock.assert_awaited_once_with("Hero has left the party.")
        reset_mock.assert_called_once()
        cancel_mock.assert_called_once()


class CreateListenersTests(unittest.TestCase):
    def setUp(self) -> None:
        namespace = load_rpg_main_namespace()
        self.create_listeners = namespace["_create_listeners"]
        self.log_bindings = namespace["_log_bindings"]
        self.socket_module = self.create_listeners.__globals__["socket"]
        self.logging_module = self.log_bindings.__globals__["logging"]

    class _FakeSocket:
        def __init__(self, family, name):
            self.family = family
            self._name = name
            self.inheritable = False
            self.bound = None
            self.listen_backlog = None

        def set_inheritable(self, flag: bool) -> None:
            self.inheritable = flag

        def setsockopt(self, *args, **kwargs) -> None:
            return None

        def bind(self, address) -> None:
            self.bound = address

        def listen(self, backlog: int) -> None:
            self.listen_backlog = backlog

        def getsockname(self):
            return self._name

    def test_dualstack_socket_path(self) -> None:
        fake_sock = self._FakeSocket(self.socket_module.AF_INET6, ("::", 9000))
        with mock.patch.object(self.socket_module, "create_server", return_value=fake_sock) as create_mock, \
                mock.patch.object(self.socket_module, "socket") as socket_ctor:
            sockets = self.create_listeners(9000, 100)

        self.assertEqual(sockets, [fake_sock])
        self.assertTrue(fake_sock.inheritable)
        create_mock.assert_called_once()
        socket_ctor.assert_not_called()

    def test_fallback_binds_ipv6_then_ipv4(self) -> None:
        fake_sock6 = self._FakeSocket(self.socket_module.AF_INET6, ("::", 9100))
        fake_sock4 = self._FakeSocket(self.socket_module.AF_INET, ("0.0.0.0", 9100))

        def create_server_fail(*args, **kwargs):
            raise OSError("no dual stack")

        def socket_factory(family, sock_type):
            if family == self.socket_module.AF_INET6:
                return fake_sock6
            if family == self.socket_module.AF_INET:
                return fake_sock4
            raise AssertionError(f"Unexpected family: {family}")

        with mock.patch.object(self.socket_module, "create_server", side_effect=create_server_fail), \
                mock.patch.object(self.socket_module, "socket", side_effect=socket_factory) as socket_ctor:
            sockets = self.create_listeners(9100, 50)

        self.assertEqual(sockets, [fake_sock6, fake_sock4])
        self.assertTrue(fake_sock6.inheritable)
        self.assertTrue(fake_sock4.inheritable)
        self.assertEqual(fake_sock6.bound, ("::", 9100))
        self.assertEqual(fake_sock4.bound, ("0.0.0.0", 9100))
        self.assertEqual(fake_sock4.listen_backlog, 50)
        self.assertGreaterEqual(socket_ctor.call_count, 2)

    def test_log_bindings_formats_addresses(self) -> None:
        fake_logger = mock.Mock()
        ipv6 = self._FakeSocket(self.socket_module.AF_INET6, ("::1", 8001))
        ipv4 = self._FakeSocket(self.socket_module.AF_INET, ("127.0.0.1", 8002))

        with mock.patch.object(self.logging_module, "getLogger", return_value=fake_logger) as get_logger_mock:
            self.log_bindings([ipv6, ipv4], is_ssl=True)

        get_logger_mock.assert_called_once_with("uvicorn.error")
        fake_logger.info.assert_has_calls(
            [
                mock.call("Uvicorn running on %s://%s:%d (Press CTRL+C to quit)", "https", "[::1]", 8001),
                mock.call("Uvicorn running on %s://%s:%d (Press CTRL+C to quit)", "https", "127.0.0.1", 8002),
            ],
            any_order=False,
        )


class OpenAIResponseFormatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.settings["openai_api_key"] = "test-key"

    async def test_json_schema_sent_via_response_format(self):
        schema = rpg.build_turn_schema()
        captured: Dict[str, Any] = {}

        class DummyResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {}
                self._payload = {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {"nar": "story", "img": "prompt", "pub": [], "upd": []}
                                    ),
                                }
                            ],
                        }
                    ],
                    "usage": {},
                }
                self.text = json.dumps(self._payload)

            def json(self):
                return self._payload

        class DummyAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers=None, json=None):
                captured["request"] = {"url": url, "headers": headers, "json": json}
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyAsyncClient):
            result = await rpg._openai_generate_structured(
                model="gpt-4.1-mini",
                system_prompt="do stuff",
                user_payload={"input": "value"},
                schema=schema,
                temperature=None,
                record_usage=False,
                include_thinking_budget=False,
                dev_snapshot="test",
                schema_name="turn_payload",
            )

        self.assertEqual(result["nar"], "story")
        request_body = captured["request"]["json"]
        self.assertIn("response_format", request_body)
        self.assertNotIn("text", request_body)
        self.assertEqual(request_body["response_format"].get("type"), "json_schema")
        json_schema_cfg = request_body["response_format"].get("json_schema")
        self.assertIsInstance(json_schema_cfg, dict)
        self.assertIn("schema", json_schema_cfg)
        self.assertIn("name", json_schema_cfg)


class OpenAIRequestBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.settings["openai_api_key"] = "sk-openai"

    async def _invoke(self, model: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        captured: Dict[str, Any] = {}

        class DummyResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {}
                self._payload = {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps({
                                        "nar": "story",
                                        "img": "prompt",
                                        "pub": [],
                                        "upd": [],
                                    }),
                                }
                            ],
                        }
                    ],
                    "usage": {},
                }
                self.text = json.dumps(self._payload)

            def json(self):
                return self._payload

        class DummyAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers=None, json=None):
                captured["request"] = {"url": url, "headers": headers, "json": json}
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyAsyncClient):
            result = await rpg._openai_generate_structured(
                model=model,
                system_prompt="SYS",
                user_payload={"input": "value"},
                schema=rpg.build_turn_schema(),
                record_usage=False,
                include_thinking_budget=True,
                dev_snapshot="test",
                schema_name="turn_payload",
            )

        self.assertIn("request", captured)
        return result, captured["request"]

    async def test_gpt5_omits_temperature_but_adds_reasoning(self):
        rpg.game_state.settings["thinking_mode"] = "deep"

        result, request = await self._invoke("gpt-5o-mini")

        self.assertEqual(result["nar"], "story")
        self.assertIn("reasoning", request["json"])
        self.assertEqual(request["json"]["reasoning"], {"effort": "high"})
        self.assertNotIn("temperature", request["json"])
        self.assertNotIn("OpenAI-Beta", request["headers"])

    async def test_o1_models_require_reasoning_header(self):
        rpg.game_state.settings["thinking_mode"] = "brief"

        result, request = await self._invoke("o1-preview")

        self.assertEqual(result["nar"], "story")
        self.assertEqual(request["headers"].get("OpenAI-Beta"), "reasoning")
        self.assertEqual(request["json"].get("temperature"), rpg.DEFAULT_TEXT_TEMPERATURE)
        self.assertEqual(request["json"].get("reasoning"), {"effort": "low"})


class CompileUserPayloadTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_payload_includes_history_and_serializes_abilities(self):
        rpg.game_state.turn_index = 3
        rpg.game_state.settings["world_style"] = "Solarpunk"
        rpg.game_state.settings["difficulty"] = "Hard"

        rpg.game_state.history.append(
            rpg.TurnRecord(index=0, narrative="Intro", image_prompt="orb", timestamp=123.0)
        )

        p1 = "p1"
        p2 = "p2"
        rpg.game_state.players[p1] = rpg.Player(
            id=p1,
            name="Alice",
            background="Scholar",
            character_class="Mage",
            abilities=[rpg.Ability(name="Arcana", expertise="expert")],
            inventory=["orb"],
            conditions=["healed"],
            status_word="ready",
            pending_join=False,
            token="tokA",
        )
        rpg.game_state.players[p2] = rpg.Player(
            id=p2,
            name="Bryn",
            background="Scout",
            character_class="",
            abilities=[{"n": "Stealth", "x": "novice"}],
            inventory=["cloak"],
            conditions=[],
            status_word="unknown",
            pending_join=True,
            token="tokB",
        )
        rpg.game_state.submissions[p1] = "Scout ahead"
        rpg.game_state.submissions[p2] = "Hold position"
        departed_name = "Cassidy"
        rpg.game_state.departed_players[rpg.normalize_player_name(departed_name)] = departed_name

        payload = rpg.build_turn_request_payload()

        self.assertEqual(payload["turn_index"], 3)
        self.assertEqual(payload["world_style"], "Solarpunk")
        self.assertEqual(payload["difficulty"], "Hard")
        self.assertEqual(payload["history"][0]["turn"], 0)
        self.assertEqual(payload["history"][0]["image_prompt"], "orb")
        self.assertEqual(payload["history_mode"], rpg.HISTORY_MODE_FULL)
        self.assertEqual(payload["history_summary"], [])

        players = payload["players"]
        self.assertEqual(players[p1]["ab"], [{"n": "Arcana", "x": "expert"}])
        self.assertEqual(players[p2]["ab"], [{"n": "Stealth", "x": "novice"}])
        self.assertFalse(players[p1]["pending_join"])
        self.assertTrue(players[p2]["pending_join"])
        self.assertFalse(players[p1]["pending_leave"])
        self.assertFalse(players[p2]["pending_leave"])

        self.assertEqual(payload["submissions"][p1], "Scout ahead")
        self.assertEqual(payload["submissions"][p2], "Hold position")
        self.assertEqual(payload["departed_players"], [departed_name])

    def test_summary_history_mode_uses_bullets(self):
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY
        rpg.game_state.turn_index = 2
        r1 = rpg.TurnRecord(
            index=0,
            narrative="The heroes arrived at the ruins.",
            image_prompt="ruins",
            timestamp=1.0,
        )
        r2 = rpg.TurnRecord(
            index=1,
            narrative="They defeated the guardian and claimed a relic.",
            image_prompt="guardian",
            timestamp=2.0,
        )
        rpg.game_state.history.extend([r1, r2])
        rpg.game_state.history_summary = ["- Turn 0: The heroes arrived at the ruins."]

        payload = rpg.build_turn_request_payload()

        self.assertEqual(payload["history_mode"], rpg.HISTORY_MODE_SUMMARY)
        self.assertEqual(payload["history"], rpg.game_state.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in payload["history"]))

        # When cached summary is empty, fallback summarization should kick in.
        rpg.game_state.history_summary.clear()
        payload = rpg.build_turn_request_payload()
        self.assertTrue(payload["history"])
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in payload["history"]))


class HistorySummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_update_summary_uses_fallback_without_api_key(self):
        rec = rpg.TurnRecord(index=0, narrative="A mysterious storm rolls in.", image_prompt="storm", timestamp=1.0)
        rpg.game_state.history.append(rec)

        await rpg.update_history_summary(rec)

        self.assertTrue(rpg.game_state.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.game_state.history_summary))

    async def test_update_summary_uses_model_output(self):
        rec = rpg.TurnRecord(
            index=1,
            narrative="The heroes rescue villagers from the cellar.",
            image_prompt="cellar",
            timestamp=2.0,
        )
        rpg.game_state.history.append(rec)
        rpg.game_state.settings["gemini_api_key"] = "test-key"
        rpg.game_state.settings["grok_api_key"] = "test-key"
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        fake_summary = rpg.SummaryStructured(summary=["Rescued villagers from the cellar."])

        with mock.patch("rpg.request_summary_payload", new=mock.AsyncMock(return_value=fake_summary)):
            await rpg.update_history_summary(rec)

        self.assertEqual(rpg.game_state.history_summary, ["- Rescued villagers from the cellar."])

    async def test_update_summary_falls_back_on_error(self):
        rec = rpg.TurnRecord(index=2, narrative="A dragon attacks the camp.", image_prompt="dragon", timestamp=3.0)
        rpg.game_state.history.append(rec)
        rpg.game_state.settings["gemini_api_key"] = "test-key"
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        failing = mock.AsyncMock(side_effect=RuntimeError("summary boom"))

        with mock.patch("rpg.request_summary_payload", new=failing):
            await rpg.update_history_summary(rec)

        self.assertTrue(rpg.game_state.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.game_state.history_summary))

    async def test_update_summary_skips_model_when_mode_full(self):
        rec = rpg.TurnRecord(
            index=3,
            narrative="Ancient wards flare and seal the tomb.",
            image_prompt="wards",
            timestamp=4.0,
        )
        rpg.game_state.history.append(rec)
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_FULL
        rpg.game_state.settings["gemini_api_key"] = "present"

        sentinel = mock.AsyncMock()

        with mock.patch("rpg.request_summary_payload", new=sentinel):
            await rpg.update_history_summary(rec)

        sentinel.assert_not_awaited()
        self.assertTrue(rpg.game_state.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.game_state.history_summary))


class ThinkingBudgetTests(unittest.TestCase):
    def test_flash_lite_brief_budget(self):
        self.assertEqual(rpg.compute_thinking_budget("gemini-2.5-flash-lite", "brief"), 512)

    def test_flash_dynamic_for_deep(self):
        self.assertEqual(rpg.compute_thinking_budget("gemini-2.5-flash", "deep"), -1)

    def test_pro_min_budget_when_disabled(self):
        self.assertEqual(rpg.compute_thinking_budget("gemini-2.5-pro", "none"), 128)

    def test_non_thinking_model_returns_none(self):
        self.assertIsNone(rpg.compute_thinking_budget("gemini-1.5-flash", "balanced"))

    def test_grok_models_have_no_thinking_budget(self):
        self.assertIsNone(rpg.compute_thinking_budget("grok-4", "deep"))


class StructuredGenerationDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_generate_json_uses_grok_dispatch(self):
        fake_payload = {
            "nar": "Scene",
            "img": "",
            "pub": [],
            "upd": [],
        }
        with mock.patch(
            "rpg._grok_generate_structured",
            new=mock.AsyncMock(return_value=fake_payload),
        ) as grok_mock:
            result = await rpg.request_turn_payload(
                model="grok-4",
                system_prompt="SYS",
                user_payload={},
                schema=rpg.build_turn_schema(),
            )

        grok_mock.assert_awaited_once()
        self.assertIsInstance(result, rpg.TurnStructured)

    async def test_generate_summary_uses_grok_dispatch(self):
        fake_response = {"summary": ["Item"]}
        with mock.patch(
            "rpg._grok_generate_structured",
            new=mock.AsyncMock(return_value=fake_response),
        ) as grok_mock:
            result = await rpg.request_summary_payload(
                model="models/grok-4-fast",
                system_prompt="SYS",
                user_payload={},
                schema=rpg.SUMMARY_SCHEMA,
            )

        grok_mock.assert_awaited_once()
        self.assertEqual(result.summary, ["Item"])


class CostCalculationTests(unittest.TestCase):
    def test_completion_pricing_uses_completion_tokens(self):
        result = rpg.calculate_turn_cost(
            model="gemini-2.5-pro",
            prompt_tokens=220_000,
            completion_tokens=1_000,
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.55)
        self.assertAlmostEqual(result["completion_usd"], 0.01)
        self.assertAlmostEqual(result["total_usd"], 0.56)


class JoinGameTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_first_join_triggers_initial_turn(self):
        pid = "fixedpid"
        issued_token = "issued"

        async def fake_resolve(initial=False):
            self.assertTrue(initial)
            player = rpg.game_state.players[pid]
            player.pending_join = False
            rpg.game_state.turn_index += 1

        with mock.patch("secrets.token_hex", side_effect=[pid, issued_token]), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.resolve_turn", new=mock.AsyncMock(side_effect=fake_resolve)) as resolve_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.JoinBody(name="Alice", background="Ranger")
            result = await rpg.join_game(body)

        self.assertEqual(result["player_id"], pid)
        self.assertEqual(result["auth_token"], issued_token)
        self.assertIn(pid, rpg.game_state.players)
        self.assertEqual(rpg.game_state.turn_index, 1)
        self.assertFalse(rpg.game_state.players[pid].pending_join)
        self.assertEqual(rpg.game_state.players[pid].token, issued_token)
        announce_mock.assert_awaited_once()
        self.assertIn("starting a new world", announce_mock.await_args.args[0])
        resolve_mock.assert_awaited_once_with(initial=True)

    async def test_failed_initial_turn_removes_player(self):
        pid = "fixedpid"
        issued_token = "issued"

        async def failing_resolve(initial=False):
            raise RuntimeError("boom")

        with mock.patch("secrets.token_hex", side_effect=[pid, issued_token]), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.resolve_turn", new=mock.AsyncMock(side_effect=failing_resolve)) as resolve_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.JoinBody(name="Alice", background="Ranger")
            with self.assertRaises(RuntimeError):
                await rpg.join_game(body)

        self.assertNotIn(pid, rpg.game_state.players)
        announce_mock.assert_awaited_once()
        resolve_mock.assert_awaited_once_with(initial=True)

    async def test_subsequent_join_skips_initial_turn(self):
        rpg.game_state.turn_index = 1
        rpg.game_state.current_narrative = "World is underway"
        long_name = "A" * 60
        long_background = "B" * 220

        pid = "latejoin"
        issued_token = "newtoken"

        with mock.patch("secrets.token_hex", side_effect=[pid, issued_token]), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.resolve_turn", new=mock.AsyncMock()) as resolve_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock:
            body = rpg.JoinBody(name=long_name, background=long_background)
            result = await rpg.join_game(body)

        self.assertEqual(result["player_id"], pid)
        self.assertEqual(result["auth_token"], issued_token)
        resolve_mock.assert_not_awaited()
        announce_mock.assert_not_awaited()
        broadcast_mock.assert_awaited()
        player = rpg.game_state.players[pid]
        self.assertEqual(player.name, long_name[:40])
        self.assertEqual(player.background, long_background[:200])
        self.assertTrue(player.pending_join)

    async def test_join_removes_departed_name(self):
        name = "Elder"
        normalized = rpg.normalize_player_name(name)
        rpg.game_state.turn_index = 1
        rpg.game_state.current_narrative = "World continues"
        rpg.game_state.departed_players[normalized] = name
        pid = "returning"
        issued_token = "ret-token"

        with mock.patch("secrets.token_hex", side_effect=[pid, issued_token]), \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.JoinBody(name=name, background="Wizard")
            result = await rpg.join_game(body)

        self.assertEqual(result["player_id"], pid)
        self.assertEqual(result["auth_token"], issued_token)
        self.assertNotIn(normalized, rpg.game_state.departed_players)

class SubmitActionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-submit")

    async def test_known_player_submission_trimmed(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="p1", token="tok-submit", text="  Attack the gate   ")
            response = await rpg.submit_action(body)

        self.assertEqual(response, {"ok": True})
        self.assertEqual(rpg.game_state.submissions["p1"], "Attack the gate")

    async def test_unknown_player_submission_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="missing", token="tok-submit", text="observe")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(rpg.game_state.submissions, {})

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="p1", token="wrong", text="observe")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(rpg.game_state.submissions, {})

    async def test_submission_rejected_when_busy(self):
        rpg.game_state.lock = rpg.LockState(active=True, reason="resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.SubmitBody(player_id="p1", token="tok-submit", text="charge")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(rpg.game_state.submissions, {})
        broadcast.assert_not_awaited()


class CreateImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-image")
        rpg.game_state.last_image_prompt = "Ancient ruins shrouded in mist"

    async def test_unknown_player_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreateImageBody(player_id="ghost", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertFalse(rpg.game_state.lock.active)

    async def test_known_player_generates_image(self):
        fake_data_url = "data:image/png;base64,abc"
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()), \
                mock.patch("rpg.announce", new=mock.AsyncMock()), \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(return_value=fake_data_url)) as gen:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            result = await rpg.create_image(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.last_image_data_url, fake_data_url)
        self.assertFalse(rpg.game_state.lock.active)
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "scene")

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreateImageBody(player_id="p1", token="wrong")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertFalse(rpg.game_state.lock.active)

    async def test_requires_image_prompt(self):
        rpg.game_state.last_image_prompt = None
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertFalse(rpg.game_state.lock.active)
        broadcast.assert_not_awaited()

    async def test_conflict_when_busy(self):
        rpg.game_state.lock = rpg.LockState(active=True, reason="resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(rpg.game_state.lock.active)
        self.assertEqual(rpg.game_state.lock.reason, "resolving_turn")
        broadcast.assert_not_awaited()

    async def test_lock_released_on_generation_failure(self):
        fake_error = RuntimeError("generation failed")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(side_effect=fake_error)) as gen:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(RuntimeError):
                await rpg.create_image(body)

        self.assertFalse(rpg.game_state.lock.active)
        self.assertEqual(rpg.game_state.lock.reason, "")
        self.assertIsNone(rpg.game_state.last_image_data_url)
        self.assertEqual(broadcast.await_count, 2)
        announce.assert_not_awaited()
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "scene")


class MaybeQueueSceneImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.auto_image_enabled = True

    async def test_late_scene_image_records_original_turn(self):
        requested_turn = 5
        rpg.game_state.turn_index = requested_turn

        async def fake_generate(model, prompt, *, purpose, turn_index):
            self.assertEqual(purpose, "scene")
            self.assertEqual(turn_index, requested_turn)
            # Simulate the turn advancing before usage is recorded
            rpg.game_state.turn_index = requested_turn + 1
            rpg.record_image_usage(model, purpose=purpose, images=1, turn_index=turn_index)
            return "data:image/png;base64,ok"

        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()), \
                mock.patch("rpg.gemini_generate_image", new=fake_generate):
            await rpg.schedule_auto_scene_image("Forest glade", requested_turn)
            # Allow the scheduled task to run to completion
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        self.assertEqual(rpg.game_state.last_scene_image_turn_index, requested_turn)
        self.assertEqual(rpg.game_state.last_image_turn_index, requested_turn)
        self.assertFalse(rpg.game_state.lock.active)
        self.assertGreaterEqual(broadcast.await_count, 1)


class MaybeQueueSceneVideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.auto_video_enabled = True
        rpg.game_state.last_video_prompt = "Arcane camera sweep"
        rpg.game_state.last_image_data_url = "data:image/png;base64,example"

    async def test_generates_video_and_tracks_turn(self):
        requested_turn = 7
        rpg.game_state.turn_index = requested_turn

        fake_video = rpg.SceneVideo(
            url="/generated_media/auto.mp4",
            prompt="Arcane camera sweep",
            negative_prompt=None,
            model="veo-3.0-generate-001",
            updated_at=123.0,
            file_path=str(rpg.GENERATED_MEDIA_DIR / "auto.mp4"),
        )

        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.generate_scene_video", new=mock.AsyncMock(return_value=fake_video)) as gen_mock:
            await rpg.schedule_auto_scene_video("Arcane camera sweep", requested_turn)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        self.assertIs(rpg.game_state.scene_video, fake_video)
        self.assertEqual(rpg.game_state.last_video_prompt, "Arcane camera sweep")
        self.assertEqual(rpg.game_state.last_scene_video_turn_index, requested_turn)
        self.assertFalse(rpg.game_state.lock.active)
        announce_mock.assert_awaited_once()
        gen_mock.assert_awaited()
        self.assertEqual(gen_mock.await_args.args[0], "Arcane camera sweep")
        self.assertEqual(gen_mock.await_args.kwargs.get("turn_index"), requested_turn)
        self.assertNotIn("image_data_url", gen_mock.await_args.kwargs)
        self.assertGreaterEqual(broadcast.await_count, 1)

    async def test_skips_when_disabled(self):
        rpg.game_state.auto_video_enabled = False
        with mock.patch("rpg.generate_scene_video", new=mock.AsyncMock()) as gen_mock:
            await rpg.schedule_auto_scene_video("Any", 1)
            await asyncio.sleep(0)
        gen_mock.assert_not_awaited()


class GenerateSceneVideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def _invoke(self, **kwargs):
        payload_holder = {}
        image_instances = []
        saved_paths = []

        class FakeImage:
            def __init__(self, image_bytes=None, mime_type=None):
                self.image_bytes = image_bytes
                self.mime_type = mime_type
                image_instances.append(self)

        class FakeVideoFile:
            def save(self, path):
                Path(path).write_bytes(b"video-bytes")
                saved_paths.append(path)

        fake_generated = types.SimpleNamespace(video=FakeVideoFile())

        class FakeOperation:
            done = True
            error = None
            response = types.SimpleNamespace(generated_videos=[fake_generated])

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.models = self
                self.operations = self
                self.files = self

            def generate_videos(self, **payload):
                payload_holder["payload"] = payload
                return FakeOperation()

            def get(self, operation=None):
                return operation or FakeOperation()

            def download(self, file):
                return None

        async def fake_to_thread(func, *f_args, **f_kwargs):
            return func(*f_args, **f_kwargs)

        with mock.patch.object(rpg, "require_gemini_api_key", return_value="api-key"), \
             mock.patch.object(rpg, "genai", types.SimpleNamespace(Client=FakeClient)), \
             mock.patch.object(rpg, "genai_types", types.SimpleNamespace(Image=FakeImage)), \
             mock.patch("rpg.record_video_usage") as record_usage, \
             mock.patch("rpg._probe_mp4_duration_seconds", return_value=4), \
             mock.patch("rpg._archive_generated_media") as archive_media, \
             mock.patch("asyncio.to_thread", new=fake_to_thread):
            result = await rpg.generate_scene_video(**kwargs)

        for path in saved_paths:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

        return result, payload_holder.get("payload", {}), image_instances, record_usage, archive_media

    async def test_includes_reference_image_when_data_url_valid(self):
        image_data = base64.b64encode(b"frame").decode("ascii")
        data_url = f"data:image/png;base64,{image_data}"
        prompt = "Cinematic sweep"

        result, payload, image_instances, record_usage, archive_media = await self._invoke(
            prompt=prompt,
            model="veo-3.0",
            image_data_url=data_url,
        )

        self.assertIsInstance(result, rpg.SceneVideo)
        self.assertIn("image", payload)
        self.assertEqual(payload.get("prompt"), prompt)
        self.assertTrue(image_instances)
        self.assertIs(payload["image"], image_instances[0])
        self.assertEqual(image_instances[0].mime_type, "image/png")
        self.assertEqual(image_instances[0].image_bytes, b"frame")
        record_usage.assert_called_once_with("veo-3.0", seconds=4, turn_index=None)
        archive_media.assert_not_called()

    async def test_skips_reference_image_when_data_invalid(self):
        data_url = "data:image/png;base64,@@not-valid@@"

        result, payload, image_instances, record_usage, archive_media = await self._invoke(
            prompt="Arcane pan",
            model="veo-3.0",
            image_data_url=data_url,
        )

        self.assertIsInstance(result, rpg.SceneVideo)
        self.assertNotIn("image", payload)
        self.assertFalse(image_instances)
        record_usage.assert_called_once_with("veo-3.0", seconds=4, turn_index=None)
        archive_media.assert_not_called()


class BuildSceneImagePartsTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        rpg.game_state.settings["world_style"] = "Steampunk"

    def test_returns_none_when_no_portraits_available(self):
        rpg.game_state.players["p1"] = rpg.Player(
            id="p1",
            name="Scout",
            background="Ranger",
        )

        parts = rpg._build_scene_image_parts("Forest watch")

        self.assertIsNone(parts)

    def test_inlines_portraits_and_builds_directives(self):
        portrait_a = rpg.PlayerPortrait(
            data_url="data:image/png;base64," + base64.b64encode(b"portrait-a").decode("ascii"),
            prompt="Hero portrait",
            updated_at=10.0,
        )
        portrait_b = rpg.PlayerPortrait(
            data_url="data:image/png;base64," + base64.b64encode(b"portrait-b").decode("ascii"),
            prompt="Rogue portrait",
            updated_at=11.0,
        )
        rpg.game_state.players["alpha"] = rpg.Player(
            id="alpha",
            name="Hero",
            background="Scholar",
            character_class="Wizard",
            inventory=["wand", "satchel"],
            conditions=["wounded"],
            status_word="Focused",
            portrait=portrait_a,
        )
        rpg.game_state.players["beta"] = rpg.Player(
            id="beta",
            name="Shade",
            background="Scout",
            character_class="Rogue",
            inventory=["dagger"],
            conditions=[],
            status_word="",  # ensure empty values are ignored
            portrait=portrait_b,
        )

        parts = rpg._build_scene_image_parts("Fog-laden city rooftops")

        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 3)  # two portraits + directive block
        inline_part = parts[0]["inlineData"]
        self.assertEqual(inline_part["mimeType"], "image/png")
        self.assertEqual(inline_part["data"], base64.b64encode(b"portrait-a").decode("ascii"))

        text_block = parts[-1]["text"]
        self.assertIn("World style: Steampunk", text_block)
        self.assertIn("Hero: match the portrait reference (Wizard, Scholar, mood: focused).", text_block)
        self.assertIn("Hero: include visible gear from inventory (wand, satchel).", text_block)
        self.assertIn("Hero: reflect current conditions (wounded).", text_block)
        self.assertIn("Scene prompt: Fog-laden city rooftops", text_block)
        # Shade has no additional descriptors beyond class/background line
        self.assertIn("Shade: match the portrait reference (Rogue, Scout).", text_block)


class PublicSnapshotPrivacyTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        player = rpg.Player(id="p1", name="Alice", background="Mage", token="tok")
        rpg.game_state.players[player.id] = player
        rpg.game_state.submissions[player.id] = "Explore the ruins"

    def test_public_snapshot_includes_id_without_token(self):
        snap = rpg.game_state.public_snapshot()
        self.assertTrue(snap["players"])
        self.assertIn("id", snap["players"][0])
        self.assertNotIn("token", snap["players"][0])

    def test_submission_ids_not_exposed(self):
        snap = rpg.game_state.public_snapshot()
        self.assertTrue(snap["submissions"])
        self.assertNotIn("player_id", snap["submissions"][0])

    def test_public_snapshot_includes_submission_per_player(self):
        snap = rpg.game_state.public_snapshot()

        self.assertEqual(snap["players"][0]["submission"], "Explore the ruins")

    def test_public_snapshot_submission_none_when_absent(self):
        rpg.game_state.submissions.clear()

        snap = rpg.game_state.public_snapshot()

        self.assertIsNone(snap["players"][0]["submission"])

    def test_public_snapshot_includes_language(self):
        snap = rpg.game_state.public_snapshot()

        self.assertEqual(snap["language"], rpg.DEFAULT_LANGUAGE)

    def test_public_snapshot_exposes_history(self):
        rec = rpg.TurnRecord(index=0, narrative="Intro", image_prompt="torch", timestamp=42.0)
        rpg.game_state.history.append(rec)

        snap = rpg.game_state.public_snapshot()

        self.assertIn("history", snap)
        self.assertEqual(len(snap["history"]), 1)
        self.assertEqual(snap["history"][0]["turn"], 0)
        self.assertEqual(snap["history"][0]["narrative"], "Intro")
        self.assertEqual(snap["history"][0]["image_prompt"], "torch")

    def test_public_snapshot_exposes_class_name(self):
        rpg.game_state.players["p1"].character_class = "Battle Mage"

        snap = rpg.game_state.public_snapshot()

        self.assertIn("cls", snap["players"][0])
        self.assertEqual(snap["players"][0]["cls"], "Battle Mage")

    def test_public_snapshot_includes_portrait_payload(self):
        rpg.game_state.players["p1"].portrait = rpg.PlayerPortrait(
            data_url="data:image/png;base64,portrait",
            prompt="prompt text",
            updated_at=123.45,
        )

        snap = rpg.game_state.public_snapshot()
        portrait = snap["players"][0].get("portrait")

        self.assertIsInstance(portrait, dict)
        self.assertEqual(portrait["data_url"], "data:image/png;base64,portrait")
        self.assertEqual(portrait["prompt"], "prompt text")

    def test_snapshot_includes_image_and_token_stats(self):
        rpg.game_state.last_image_data_url = "data:image/png;base64,xyz"
        rpg.game_state.last_image_prompt = "An ancient gate"
        rpg.game_state.turn_index = 5
        rpg.game_state.last_token_usage = {"input": 20, "output": 30, "thinking": 10}
        rpg.game_state.last_turn_runtime = 4
        rpg.game_state.last_cost_usd = 1.23
        rpg.game_state.session_cost_usd = 4.56
        rpg.game_state.last_image_model = "gemini-2.5-flash-image-preview"
        rpg.game_state.last_image_kind = "scene"
        rpg.game_state.last_image_cost_usd = 0.039
        rpg.game_state.last_image_usd_per_image = 0.039
        rpg.game_state.last_image_tokens = 1290
        rpg.game_state.last_image_count = 1
        rpg.game_state.last_image_turn_index = 5
        rpg.game_state.last_scene_image_cost_usd = 0.039
        rpg.game_state.last_scene_image_usd_per_image = 0.039
        rpg.game_state.last_scene_image_model = "gemini-2.5-flash-image-preview"
        rpg.game_state.last_scene_image_turn_index = 5
        rpg.game_state.session_image_cost_usd = 0.078
        rpg.game_state.session_image_requests = 2
        rpg.game_state.session_image_kind_counts = {"scene": 2}
        rpg.game_state.image_counts_by_turn = {5: {"scene": 1, "portrait": 1}}
        rpg.game_state.current_turn_image_counts = rpg.game_state.image_counts_by_turn[5]
        rpg.game_state.current_turn_index_for_image_counts = 5
        rpg.game_state.last_scene_video_turn_index = 5
        rpg.game_state.last_video_model = "veo-3.0-generate-001"
        rpg.game_state.last_video_tier = "default"
        rpg.game_state.last_video_cost_usd = 0.6
        rpg.game_state.last_video_usd_per_second = 0.4
        rpg.game_state.last_video_seconds = 1.5
        rpg.game_state.last_video_turn_index = 5
        rpg.game_state.session_video_cost_usd = 1.2
        rpg.game_state.session_video_seconds = 3.0
        rpg.game_state.session_video_requests = 2

        snap = rpg.game_state.public_snapshot()

        self.assertEqual(snap["image"]["data_url"], "data:image/png;base64,xyz")
        self.assertEqual(snap["image"]["prompt"], "An ancient gate")
        token_usage = snap["token_usage"]
        self.assertEqual(token_usage["last_turn"]["input"], 20)
        self.assertEqual(token_usage["last_turn"]["output"], 30)
        self.assertEqual(token_usage["last_turn"]["thinking"], 10)
        self.assertEqual(token_usage["last_turn"]["total"], 60)
        self.assertEqual(token_usage["last_turn"]["tokens_per_sec"], 15.0)
        self.assertEqual(token_usage["last_turn"]["cost_usd"], 1.23)
        self.assertEqual(token_usage["session"]["input"], 0)
        self.assertEqual(token_usage["session"]["total"], 0)
        self.assertEqual(token_usage["session"]["cost_usd"], 4.56)
        totals = token_usage["totals"]
        self.assertAlmostEqual(totals["last_usd"], 1.869, places=3)
        self.assertAlmostEqual(totals["session_usd"], 5.838, places=3)
        self.assertEqual(totals["breakdown"]["image_session_usd"], 0.078)
        self.assertEqual(totals["breakdown"]["video_session_usd"], 1.2)
        self.assertAlmostEqual(totals["breakdown"]["video_last_turn_usd"], 0.6, places=3)
        video_stats = token_usage["video"]
        self.assertEqual(video_stats["last"]["model"], "veo-3.0-generate-001")
        self.assertEqual(video_stats["last"]["tier"], "default")
        self.assertAlmostEqual(video_stats["last"]["cost_usd"], 0.6, places=3)
        self.assertAlmostEqual(video_stats["last"]["seconds"], 1.5, places=3)
        self.assertEqual(video_stats["last"]["turn_index"], 5)
        self.assertEqual(video_stats["session"]["requests"], 2)
        self.assertAlmostEqual(video_stats["session"]["seconds"], 3.0, places=3)
        self.assertAlmostEqual(video_stats["session"]["avg_usd_per_second"], 0.4, places=3)
        self.assertAlmostEqual(video_stats["session"]["avg_seconds_per_request"], 1.5, places=3)
        image_stats = token_usage["image"]
        self.assertEqual(image_stats["last"]["model"], "gemini-2.5-flash-image-preview")
        self.assertEqual(image_stats["last"]["kind"], "scene")
        self.assertEqual(image_stats["last"]["turn_index"], 5)
        self.assertEqual(image_stats["session"]["images"], 2)
        self.assertAlmostEqual(image_stats["session"]["avg_usd_per_image"], 0.039, places=3)
        self.assertEqual(image_stats["session"]["by_kind"], {"scene": 2})
        self.assertEqual(image_stats["turn"]["by_kind"], {"scene": 1, "portrait": 1})
        self.assertEqual(image_stats["scene"]["last_turn_index"], 5)
        self.assertAlmostEqual(image_stats["scene"]["last_cost_usd"], 0.039, places=3)


class SettingsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_get_settings_returns_api_key(self):
        rpg.game_state.settings["gemini_api_key"] = "sk-alpha123456"

        result = await rpg.get_settings()

        self.assertEqual(result["gemini_api_key"], "sk-alpha123456")
        self.assertEqual(rpg.game_state.settings["gemini_api_key"], "sk-alpha123456")

    async def test_update_settings_trims_and_persists(self):
        body = rpg.SettingsUpdate(gemini_api_key="  sk-newkey  ", world_style="Noir", thinking_mode="Deep")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.settings["gemini_api_key"], "sk-newkey")
        self.assertEqual(rpg.game_state.settings["world_style"], "Noir")
        self.assertEqual(rpg.game_state.settings["thinking_mode"], "deep")
        save_mock.assert_awaited_once_with(rpg.game_state.settings)

    async def test_update_settings_sets_narration_model(self):
        body = rpg.SettingsUpdate(narration_model="eleven_flash_v2_1")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.settings["narration_model"], "eleven_flash_v2_1")
        save_mock.assert_awaited_once_with(rpg.game_state.settings)

    async def test_update_settings_sets_openai_key(self):
        body = rpg.SettingsUpdate(openai_api_key="sk-openai-test")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.settings["openai_api_key"], "sk-openai-test")
        save_mock.assert_awaited_once_with(rpg.game_state.settings)

    async def test_update_settings_ignores_blank_narration_model(self):
        rpg.game_state.settings["narration_model"] = "eleven_flash_v2_5"
        body = rpg.SettingsUpdate(narration_model="   ")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.settings["narration_model"], "eleven_flash_v2_5")
        save_mock.assert_not_awaited()

    async def test_update_settings_normalizes_history_mode(self):
        rpg.game_state.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY
        body = rpg.SettingsUpdate(history_mode=" timeline ")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.game_state.settings["history_mode"], rpg.HISTORY_MODE_FULL)
        save_mock.assert_awaited_once_with(rpg.game_state.settings)

    async def test_update_settings_rejects_unknown_thinking_mode(self):
        body = rpg.SettingsUpdate(thinking_mode="cosmic")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.update_settings(body)

        self.assertEqual(ctx.exception.status_code, 400)
        save_mock.assert_not_awaited()

    async def test_get_settings_returns_blank_api_key_when_missing(self):
        rpg.game_state.settings["gemini_api_key"] = ""

        result = await rpg.get_settings()

        self.assertEqual(result["gemini_api_key"], "")
        self.assertIn("elevenlabs_api_key", result)


class ModelsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_api_models_includes_narration_models(self):
        async def fake_gemini_list_models():
            return [
                {
                    "name": "models/gemini-express",
                    "displayName": "Gemini Express",
                    "supported": ["generateContent"],
                    "provider": rpg.TEXT_PROVIDER_GEMINI,
                    "category": "text",
                    "modelId": "gemini-express",
                }
            ]

        async def fake_elevenlabs_list_models(api_key):
            self.assertEqual(api_key, "mock-eleven-key")
            return [{
                "id": "eleven_flash_v2_5",
                "name": "Eleven Flash v2.5",
                "languages": ["English", "German"],
                "language_codes": ["en", "de"],
            }]

        rpg.game_state.settings["elevenlabs_api_key"] = "mock-eleven-key"
        original_gemini = rpg.gemini_list_models
        original_eleven = rpg.elevenlabs_list_models
        rpg.gemini_list_models = fake_gemini_list_models
        rpg.elevenlabs_list_models = fake_elevenlabs_list_models
        try:
            result = await rpg.api_models()
        finally:
            rpg.gemini_list_models = original_gemini
            rpg.elevenlabs_list_models = original_eleven

        self.assertIn("models", result)
        self.assertIn("narration_models", result)
        self.assertEqual(result["narration_models"], [{
            "id": "eleven_flash_v2_5",
            "name": "Eleven Flash v2.5",
            "languages": ["English", "German"],
            "language_codes": ["en", "de"],
        }])

    async def test_api_models_without_elevenlabs_key_returns_empty_list(self):
        async def fake_gemini_list_models():
            return []

        original_gemini = rpg.gemini_list_models
        rpg.gemini_list_models = fake_gemini_list_models
        try:
            result = await rpg.api_models()
        finally:
            rpg.gemini_list_models = original_gemini

        self.assertIn("narration_models", result)
        self.assertEqual(result["narration_models"], [])

    async def test_api_models_returns_grok_models_when_configured(self):
        async def fake_grok_list_models(api_key):
            self.assertEqual(api_key, "grok-secret")
            return [
                {
                    "name": "grok-4",
                    "displayName": "Grok 4",
                    "supported": ["chat.completions"],
                    "provider": rpg.TEXT_PROVIDER_GROK,
                    "category": "text",
                }
            ]

        reset_state()
        rpg.game_state.settings["gemini_api_key"] = ""
        rpg.game_state.settings["grok_api_key"] = "grok-secret"

        original_gemini = rpg.gemini_list_models
        original_grok = rpg.grok_list_models
        rpg.gemini_list_models = mock.AsyncMock(side_effect=AssertionError("gemini_list_models should not be called"))
        rpg.grok_list_models = fake_grok_list_models
        try:
            result = await rpg.api_models()
        finally:
            rpg.gemini_list_models = original_gemini
            rpg.grok_list_models = original_grok

        self.assertTrue(result["models"])  # at least one
        entry = result["models"][0]
        self.assertEqual(entry["provider"], rpg.TEXT_PROVIDER_GROK)
        self.assertEqual(entry["name"], "grok-4")

    async def test_api_models_returns_openai_models_when_configured(self):
        async def fake_openai_list_models(api_key):
            self.assertEqual(api_key, "openai-secret")
            return [
                {
                    "name": "gpt-4o-mini",
                    "displayName": "GPT-4o Mini",
                    "supported": ["responses", "chat_completions"],
                    "provider": rpg.TEXT_PROVIDER_OPENAI,
                    "category": "text",
                }
            ]

        reset_state()
        rpg.game_state.settings["gemini_api_key"] = ""
        rpg.game_state.settings["openai_api_key"] = "openai-secret"

        original_gemini = rpg.gemini_list_models
        original_openai = rpg.openai_list_models
        rpg.gemini_list_models = mock.AsyncMock(side_effect=AssertionError("gemini_list_models should not be called"))
        rpg.openai_list_models = fake_openai_list_models
        try:
            result = await rpg.api_models()
        finally:
            rpg.gemini_list_models = original_gemini
            rpg.openai_list_models = original_openai

        self.assertTrue(result["models"])  # at least one
        entry = result["models"][0]
        self.assertEqual(entry["provider"], rpg.TEXT_PROVIDER_OPENAI)
        self.assertEqual(entry["name"], "gpt-4o-mini")
        self.assertIn("responses", entry["supported"])

    async def test_api_models_sorted_by_display_name(self):
        async def fake_gemini_list_models():
            return [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supported": ["generateContent"],
                    "provider": rpg.TEXT_PROVIDER_GEMINI,
                    "category": "text",
                    "modelId": "gemini-2.5-pro",
                },
                {
                    "name": "models/gemini-1.5-flash",
                    "displayName": "Gemini 1.5 Flash",
                    "supported": ["generateContent"],
                    "provider": rpg.TEXT_PROVIDER_GEMINI,
                    "category": "text",
                    "modelId": "gemini-1.5-flash",
                },
            ]

        async def fake_grok_list_models(api_key):
            self.assertEqual(api_key, "grok-key")
            return [
                {
                    "name": "grok-4-fast",
                    "displayName": "Grok 4 Fast",
                    "supported": ["chat.completions"],
                    "provider": rpg.TEXT_PROVIDER_GROK,
                    "category": "text",
                },
                {
                    "name": "grok-4",
                    "displayName": "Grok 4",
                    "supported": ["chat.completions"],
                    "provider": rpg.TEXT_PROVIDER_GROK,
                    "category": "text",
                },
            ]

        reset_state()
        rpg.game_state.settings["gemini_api_key"] = "gemini-key"
        rpg.game_state.settings["grok_api_key"] = "grok-key"

        original_gemini = rpg.gemini_list_models
        original_grok = rpg.grok_list_models
        rpg.gemini_list_models = fake_gemini_list_models
        rpg.grok_list_models = fake_grok_list_models
        try:
            result = await rpg.api_models()
        finally:
            rpg.gemini_list_models = original_gemini
            rpg.grok_list_models = original_grok

        names = [entry["displayName"] for entry in result["models"]]
        self.assertEqual(names, [
            "Gemini 1.5 Flash",
            "Gemini 2.5 Pro",
            "Grok 4",
            "Grok 4 Fast",
        ])


class ModelCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        rpg._MODEL_CACHE.clear()

    async def test_cache_key_obfuscates_api_key(self):
        digest = hashlib.sha256(b"secret").hexdigest()

        result = rpg._models_cache_key("gemini", "  secret  ")

        self.assertEqual(result, ("gemini", digest))
        self.assertEqual(rpg._models_cache_key("gemini", None), ("gemini", ""))

    async def test_reuses_cached_models_within_ttl(self):
        calls = 0

        async def loader():
            nonlocal calls
            calls += 1
            return [{"name": "model"}]

        first = await rpg._get_cached_models("provider", "key", loader)
        self.assertEqual(calls, 1)
        first.append({"name": "mutated"})

        second = await rpg._get_cached_models("provider", "key", loader)

        self.assertEqual(calls, 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["name"], "model")

    async def test_refreshes_models_after_ttl_expired(self):
        ttl = rpg.MODEL_CACHE_TTL_SECONDS
        times = iter([1000.0, 1000.0, 1000.0 + ttl + 1, 1000.0 + ttl + 1])
        produced = []

        async def loader():
            model = {"name": f"model-{len(produced)}"}
            produced.append(model)
            return [model]

        def fake_time():
            return next(times)

        with mock.patch("rpg.time.time", side_effect=fake_time):
            first = await rpg._get_cached_models("provider", "key", loader)
            second = await rpg._get_cached_models("provider", "key", loader)

        self.assertEqual(len(produced), 2)
        self.assertEqual(first[0]["name"], "model-0")
        self.assertEqual(second[0]["name"], "model-1")


class TextProviderDetectionTests(unittest.TestCase):
    def test_detects_grok_models_by_prefix(self):
        self.assertEqual(rpg.detect_text_provider("grok-4"), rpg.TEXT_PROVIDER_GROK)
        self.assertEqual(rpg.detect_text_provider("models/grok-4"), rpg.TEXT_PROVIDER_GROK)

    def test_defaults_to_gemini_when_unspecified(self):
        self.assertEqual(rpg.detect_text_provider("gemini-2.5-flash"), rpg.TEXT_PROVIDER_GEMINI)
        self.assertEqual(rpg.detect_text_provider(""), rpg.TEXT_PROVIDER_GEMINI)

    def test_detects_openai_models_by_pattern(self):
        self.assertEqual(rpg.detect_text_provider("gpt-4o"), rpg.TEXT_PROVIDER_OPENAI)
        self.assertEqual(rpg.detect_text_provider("openai/gpt-4.1"), rpg.TEXT_PROVIDER_OPENAI)
        self.assertEqual(rpg.detect_text_provider("chatgpt-4o-latest"), rpg.TEXT_PROVIDER_OPENAI)


class RequireTextApiKeyTests(unittest.TestCase):
    def test_missing_keys_raise_http_errors(self):
        cases = (
            rpg.TEXT_PROVIDER_GROK,
            rpg.TEXT_PROVIDER_OPENAI,
            rpg.TEXT_PROVIDER_GEMINI,
        )
        for provider in cases:
            with self.subTest(provider=provider):
                reset_state()
                with self.assertRaises(rpg.HTTPException) as ctx:
                    rpg.require_text_api_key(provider)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_returns_trimmed_keys_for_each_provider(self):
        cases = (
            (rpg.TEXT_PROVIDER_GROK, "grok_api_key"),
            (rpg.TEXT_PROVIDER_OPENAI, "openai_api_key"),
            (rpg.TEXT_PROVIDER_GEMINI, "gemini_api_key"),
        )
        for provider, field in cases:
            with self.subTest(provider=provider):
                reset_state()
                rpg.game_state.settings[field] = "  secret-value  "
                result = rpg.require_text_api_key(provider)
                self.assertEqual(result, "secret-value")


class SanitizeRequestHeadersTests(unittest.TestCase):
    def test_redacts_sensitive_authorization_headers(self):
        headers = {
            "Authorization": "Bearer token",
            "X-Goog-Api-Key": "key",
            "Accept": "application/json",
        }

        sanitized = rpg._sanitize_request_headers(headers)

        self.assertEqual(sanitized["Authorization"], "***")
        self.assertEqual(sanitized["X-Goog-Api-Key"], "***")
        self.assertEqual(sanitized["Accept"], "application/json")


class SanitizeNarrativeTests(unittest.TestCase):
    def test_none_returns_empty_string(self):
        self.assertEqual(rpg.sanitize_narrative(None), "")

    def test_sanitizes_control_characters_and_midword_breaks(self):
        raw = "  \u200bThe\u00ad quest\r\ncontinues \n\tNOW \r\n"

        result = rpg.sanitize_narrative(raw)

        self.assertEqual(result, "The quest continues NOW")

    def test_preserves_paragraph_breaks(self):
        raw = "Line one\n\nLine two"

        result = rpg.sanitize_narrative(raw)

        self.assertEqual(result, "Line one\n\nLine two")


class FormatElevenLabsExceptionTests(unittest.TestCase):
    def test_includes_message_status_body_and_headers(self):
        class DummyError(Exception):
            def __init__(self):
                super().__init__("Request throttled")
                self.status_code = 503
                self.body = {"error": "Service Unavailable"}
                self.headers = {"Retry-After": "1"}

        message = rpg._format_elevenlabs_exception(DummyError())

        self.assertIn("Request throttled", message)
        self.assertIn("HTTP 503", message)
        self.assertIn('"error": "Service Unavailable"', message)
        self.assertIn('"Retry-After": "1"', message)

    def test_defaults_to_class_name_when_no_details(self):
        class EmptyError(Exception):
            pass

        message = rpg._format_elevenlabs_exception(EmptyError())

        self.assertEqual(message, "EmptyError")


class ElevenLabsLanguageParsingTests(unittest.TestCase):
    def test_normalize_language_code_handles_variants(self):
        self.assertEqual(rpg._normalize_language_code("German (Standard)"), "de")
        self.assertEqual(rpg._normalize_language_code("de-DE"), "de")
        self.assertEqual(rpg._normalize_language_code("Deutsch"), "de")
        self.assertEqual(rpg._normalize_language_code("English"), "en")
        self.assertIsNone(rpg._normalize_language_code("Martian"))


class ElevenLabsConvertTests(unittest.TestCase):
    def setUp(self):
        rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False

    def test_returns_empty_when_library_missing(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("elevenlabs"):
                raise ImportError("module unavailable")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = rpg._elevenlabs_convert_to_base64("Echo", api_key="key-123")

        self.assertEqual(result["audio_base64"], None)
        self.assertEqual(result["metadata"], {})
        self.assertTrue(rpg._ELEVENLABS_IMPORT_ERROR_LOGGED)


class ElevenLabsDependencyTests(unittest.TestCase):
    def test_library_available_when_modules_present(self):
        calls: list[str] = []

        def fake_import(name: str):
            calls.append(name)
            return types.SimpleNamespace(__name__=name)

        with mock.patch("importlib.import_module", side_effect=fake_import) as import_mock:
            self.assertTrue(rpg._elevenlabs_library_available())

        self.assertEqual(calls, ["elevenlabs.client", "elevenlabs.types"])
        self.assertEqual(import_mock.call_count, 2)

    def test_library_unavailable_on_import_error(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            self.assertFalse(rpg._elevenlabs_library_available())


class MaybeQueueTtsWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg._ELEVENLABS_API_KEY_WARNING_LOGGED = False
        rpg._ELEVENLABS_LIBRARY_WARNING_LOGGED = False
        rpg._ELEVENLABS_IMPORT_ERROR_LOGGED = False

    async def test_skips_when_api_key_missing(self):
        rpg.game_state.auto_tts_enabled = True

        with mock.patch("rpg.asyncio.create_task") as create_task:
            await rpg.schedule_auto_tts("A tale", 2)

        create_task.assert_not_called()
        self.assertTrue(rpg._ELEVENLABS_API_KEY_WARNING_LOGGED)

    async def test_dispatches_audio_payload_and_updates_state(self):
        rpg.game_state.auto_tts_enabled = True
        rpg.game_state.settings["elevenlabs_api_key"] = "key"
        metadata = {
            "model_id": "eleven_flash",
            "character_source": "x-characters-used",
            "characters_final": 42,
            "estimated_cost_usd": 0.12,
            "estimated_credits": 0.34,
            "usd_per_million": 15.0,
            "headers": {"X-Request-ID": "req-1"},
            "request_id": "req-1",
            "subscription_total_credits": 1000,
            "subscription_remaining_credits": 900,
            "subscription_next_reset_unix": 1700000000,
        }
        payload = {"audio_base64": base64.b64encode(b"voice").decode("ascii"), "metadata": metadata}
        send_mock = mock.AsyncMock()
        broadcast_error = mock.AsyncMock()
        to_thread = mock.AsyncMock(return_value=payload)
        tasks = []
        loop = asyncio.get_running_loop()

        def create_task_and_track(coro):
            task = loop.create_task(coro)
            tasks.append(task)
            return task

        with mock.patch("rpg._elevenlabs_library_available", return_value=True), \
                mock.patch("rpg._send_json_to_sockets", send_mock), \
                mock.patch("rpg._broadcast_tts_error", broadcast_error), \
                mock.patch("rpg.asyncio.to_thread", to_thread), \
                mock.patch("rpg.asyncio.create_task", side_effect=create_task_and_track):
            await rpg.schedule_auto_tts("Narrate", 5)
            self.assertEqual(len(tasks), 1)
            await asyncio.gather(*tasks)

        broadcast_error.assert_not_awaited()
        to_thread.assert_awaited_once()
        send_mock.assert_awaited_once()
        sockets_arg, payload_arg = send_mock.await_args.args
        self.assertEqual(sockets_arg, rpg.game_state.global_sockets)
        self.assertEqual(payload_arg["event"], "tts_audio")
        self.assertEqual(payload_arg["data"]["turn_index"], 5)
        self.assertEqual(rpg.game_state.last_tts_model, "eleven_flash")
        self.assertEqual(rpg.game_state.session_tts_requests, 1)
        self.assertAlmostEqual(rpg.game_state.session_tts_cost_usd, 0.12)
        self.assertAlmostEqual(rpg.game_state.session_tts_credits, 0.34)
        self.assertEqual(rpg.game_state.last_tts_headers.get("X-Request-ID"), "req-1")

    async def test_broadcasts_error_on_worker_failure(self):
        rpg.game_state.auto_tts_enabled = True
        rpg.game_state.settings["elevenlabs_api_key"] = "key"
        tasks = []
        loop = asyncio.get_running_loop()

        def create_task_and_track(coro):
            task = loop.create_task(coro)
            tasks.append(task)
            return task

        to_thread = mock.AsyncMock(side_effect=rpg.ElevenLabsNarrationError("failed"))
        broadcast_error = mock.AsyncMock()
        send_mock = mock.AsyncMock()

        with mock.patch("rpg._elevenlabs_library_available", return_value=True), \
                mock.patch("rpg._send_json_to_sockets", send_mock), \
                mock.patch("rpg._broadcast_tts_error", broadcast_error), \
                mock.patch("rpg.asyncio.to_thread", to_thread), \
                mock.patch("rpg.asyncio.create_task", side_effect=create_task_and_track):
            await rpg.schedule_auto_tts("Narrate", 6)
            self.assertEqual(len(tasks), 1)
            await asyncio.gather(*tasks)

        broadcast_error.assert_awaited_once()
        send_mock.assert_not_awaited()
        self.assertEqual(rpg.game_state.session_tts_requests, 0)


class ToggleEndpointsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        self.player = rpg.Player(id="p1", name="Tester", background="Mage", token="tok")
        rpg.game_state.players[self.player.id] = self.player
        rpg.game_state.current_narrative = "A tense negotiation"
        rpg.game_state.turn_index = 7
        rpg.game_state.last_image_prompt = "misty forest"
        rpg.game_state.last_video_prompt = "hero walks"
        rpg.game_state.auto_tts_enabled = False

    async def test_toggle_tts_enables_and_queues_audio(self):
        rpg.game_state.settings["elevenlabs_api_key"] = "secret"
        body = rpg.ToggleTtsBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg._elevenlabs_library_available", return_value=True), \
                mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_tts", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_tts(body)

        self.assertTrue(result["auto_tts_enabled"])
        self.assertTrue(rpg.game_state.auto_tts_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()
        args, kwargs = queue_mock.await_args
        self.assertEqual(args[0], rpg.game_state.current_narrative)
        self.assertEqual(args[1], rpg.game_state.turn_index)
        self.assertFalse(kwargs)

    async def test_toggle_tts_requires_api_key(self):
        body = rpg.ToggleTtsBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_tts", new_callable=mock.AsyncMock) as queue_mock:
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.toggle_tts(body)

        self.assertEqual(excinfo.exception.status_code, 400)
        self.assertFalse(rpg.game_state.auto_tts_enabled)
        queue_mock.assert_not_called()
        broadcast_mock.assert_not_called()

    async def test_toggle_tts_requires_library(self):
        rpg.game_state.settings["elevenlabs_api_key"] = "secret"
        body = rpg.ToggleTtsBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg._elevenlabs_library_available", return_value=False), \
                mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_tts", new_callable=mock.AsyncMock) as queue_mock:
            with self.assertRaises(HTTPException) as excinfo:
                await rpg.toggle_tts(body)

        self.assertEqual(excinfo.exception.status_code, 500)
        queue_mock.assert_not_called()
        broadcast_mock.assert_not_called()
        self.assertFalse(rpg.game_state.auto_tts_enabled)

    async def test_toggle_tts_disable_stops_queue(self):
        rpg.game_state.auto_tts_enabled = True
        body = rpg.ToggleTtsBody(player_id=self.player.id, token=self.player.token, enabled=False)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_tts", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_tts(body)

        self.assertFalse(result["auto_tts_enabled"])
        self.assertFalse(rpg.game_state.auto_tts_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_not_called()

    async def test_toggle_scene_image_enables_and_forces_queue(self):
        rpg.game_state.auto_video_enabled = True
        body = rpg.ToggleSceneImageBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_scene_image", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_scene_image(body)

        self.assertTrue(result["auto_image_enabled"])
        self.assertFalse(result["auto_video_enabled"])
        self.assertTrue(rpg.game_state.auto_image_enabled)
        self.assertFalse(rpg.game_state.auto_video_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()
        args, kwargs = queue_mock.await_args
        self.assertEqual(args[0], rpg.game_state.last_image_prompt)
        self.assertEqual(args[1], rpg.game_state.turn_index)
        self.assertTrue(kwargs["force"])

    async def test_toggle_scene_image_disable_skips_queue(self):
        rpg.game_state.auto_image_enabled = True
        rpg.game_state.auto_video_enabled = True
        body = rpg.ToggleSceneImageBody(player_id=self.player.id, token=self.player.token, enabled=False)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_scene_image", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_scene_image(body)

        self.assertFalse(result["auto_image_enabled"])
        self.assertTrue(result["auto_video_enabled"])
        self.assertFalse(rpg.game_state.auto_image_enabled)
        self.assertTrue(rpg.game_state.auto_video_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_not_called()

    async def test_toggle_scene_video_enables_and_forces_queue(self):
        rpg.game_state.auto_image_enabled = True
        body = rpg.ToggleSceneVideoBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_scene_video", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_scene_video(body)

        self.assertTrue(result["auto_video_enabled"])
        self.assertFalse(result["auto_image_enabled"])
        self.assertTrue(rpg.game_state.auto_video_enabled)
        self.assertFalse(rpg.game_state.auto_image_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()
        args, kwargs = queue_mock.await_args
        self.assertEqual(args[0], "hero walks")
        self.assertEqual(args[1], rpg.game_state.turn_index)
        self.assertTrue(kwargs["force"])

    async def test_toggle_scene_video_uses_fallback_prompt_when_missing(self):
        rpg.game_state.last_video_prompt = None
        rpg.game_state.last_image_prompt = "fallback"
        body = rpg.ToggleSceneVideoBody(player_id=self.player.id, token=self.player.token, enabled=True)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock), \
                mock.patch("rpg.schedule_auto_scene_video", new_callable=mock.AsyncMock) as queue_mock:
            await rpg.toggle_scene_video(body)

        args, kwargs = queue_mock.await_args
        self.assertEqual(args[0], "fallback")
        self.assertTrue(kwargs["force"])

    async def test_toggle_scene_video_disable_skips_queue(self):
        rpg.game_state.auto_video_enabled = True
        body = rpg.ToggleSceneVideoBody(player_id=self.player.id, token=self.player.token, enabled=False)

        with mock.patch("rpg.broadcast_public", new_callable=mock.AsyncMock) as broadcast_mock, \
                mock.patch("rpg.schedule_auto_scene_video", new_callable=mock.AsyncMock) as queue_mock:
            result = await rpg.toggle_scene_video(body)

        self.assertFalse(result["auto_video_enabled"])
        self.assertFalse(rpg.game_state.auto_video_enabled)
        broadcast_mock.assert_awaited_once()
        queue_mock.assert_not_called()


class SceneVideoGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()

    async def test_generate_scene_video_writes_file_and_records_usage(self) -> None:
        base_img = base64.b64encode(b"reference-image").decode("ascii")
        image_data_url = f"data:image/png;base64,{base_img}"
        rpg.game_state.settings["video_model"] = "models/video-prod"

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        output_dir = Path(temp_dir.name)
        video_path_holder: Dict[str, str] = {}

        class DummyVideoFile:
            def save(self_inner, path: str) -> None:
                Path(path).write_bytes(b"mp4-data")
                video_path_holder["path"] = path

        class DummyOperation:
            def __init__(self) -> None:
                self.done = True
                self.error = None
                self.response = types.SimpleNamespace(
                    generated_videos=[types.SimpleNamespace(video=DummyVideoFile())]
                )

        created_clients: list[Any] = []

        class DummyClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.generate_calls: list[Dict[str, Any]] = []
                created_clients.append(self)
                self.models = types.SimpleNamespace(generate_videos=self._generate_videos)
                self.operations = types.SimpleNamespace(get=lambda op: op)
                self.files = types.SimpleNamespace(download=lambda file: None)

            def _generate_videos(self, **kwargs: Any) -> DummyOperation:
                self.generate_calls.append(kwargs)
                return DummyOperation()

        class DummyImage:
            def __init__(self, *, image_bytes: bytes, mime_type: str):
                self.image_bytes = image_bytes
                self.mime_type = mime_type

        async def immediate_to_thread(func, *args, **kwargs):  # type: ignore[override]
            return func(*args, **kwargs)

        with mock.patch.object(rpg, "GENERATED_MEDIA_DIR", output_dir), \
                mock.patch("rpg.require_gemini_api_key", return_value="genai-key"), \
                mock.patch("rpg.genai", types.SimpleNamespace(Client=DummyClient)), \
                mock.patch("rpg.genai_types", types.SimpleNamespace(Image=DummyImage)), \
                mock.patch("rpg.asyncio.to_thread", immediate_to_thread), \
                mock.patch("rpg._probe_mp4_duration_seconds", return_value=3.5) as duration_mock, \
                mock.patch("rpg.record_video_usage") as record_usage_mock, \
                mock.patch("rpg.secrets.token_hex", return_value="abcd"), \
                mock.patch("rpg.time.time", side_effect=[1000.0, 1001.0, 1002.0]):
            video = await rpg.generate_scene_video(
                prompt="   An epic duel   ",
                model="models/video-prod",
                image_data_url=image_data_url,
                negative_prompt="no blood",
                turn_index=42,
            )

        self.assertTrue(Path(video.file_path).exists())
        self.assertTrue(video.file_path.startswith(str(output_dir)))
        self.assertEqual(video.prompt, "An epic duel")
        self.assertEqual(video.negative_prompt, "no blood")
        self.assertEqual(video.model, "models/video-prod")
        self.assertEqual(video.url, "/generated_media/scene_1000_abcd.mp4")
        record_usage_mock.assert_called_once_with("models/video-prod", seconds=3.5, turn_index=42)
        duration_mock.assert_called_once()

        self.assertEqual(created_clients[0].api_key, "genai-key")
        request_kwargs = created_clients[0].generate_calls[0]
        self.assertEqual(request_kwargs["model"], "models/video-prod")
        self.assertEqual(request_kwargs["prompt"], "An epic duel")
        self.assertEqual(request_kwargs["config"], {"negative_prompt": "no blood"})
        image_arg = request_kwargs["image"]
        self.assertEqual(image_arg.mime_type, "image/png")
        self.assertEqual(image_arg.image_bytes, base64.b64decode(base_img))


class NormalizeLanguageTests(unittest.TestCase):
    def test_normalize_language_accepts_aliases(self):
        self.assertEqual(rpg.normalize_language("German (Standard)"), "de")
        self.assertEqual(rpg.normalize_language("english"), "en")

    def test_normalize_language_defaults_to_supported_default(self):
        self.assertEqual(rpg.normalize_language("Martian"), rpg.DEFAULT_LANGUAGE)


class NormalizeHistoryModeTests(unittest.TestCase):
    def test_normalize_history_mode_accepts_known_values(self):
        self.assertEqual(rpg.normalize_history_mode("full"), rpg.HISTORY_MODE_FULL)
        self.assertEqual(rpg.normalize_history_mode(" summary "), rpg.HISTORY_MODE_SUMMARY)

    def test_normalize_history_mode_defaults_to_full(self):
        self.assertEqual(rpg.normalize_history_mode("timeline"), rpg.HISTORY_MODE_FULL)
        self.assertEqual(rpg.normalize_history_mode(None), rpg.HISTORY_MODE_FULL)


class SetLanguageIfChangedTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_set_language_if_changed_updates_state_on_change(self):
        changed = rpg.set_language_if_changed("German (Standard)")

        self.assertTrue(changed)
        self.assertEqual(rpg.game_state.language, "de")
        self.assertEqual(rpg.game_state.settings["language"], "de")

    def test_set_language_if_changed_returns_false_for_none(self):
        rpg.game_state.language = "en"
        rpg.game_state.settings["language"] = "en"

        changed = rpg.set_language_if_changed(None)

        self.assertFalse(changed)
        self.assertEqual(rpg.game_state.language, "en")
        self.assertEqual(rpg.game_state.settings["language"], "en")

    def test_set_language_if_changed_returns_false_when_language_unchanged(self):
        rpg.game_state.language = "de"
        rpg.game_state.settings["language"] = "de"

        changed = rpg.set_language_if_changed("de-DE")

        self.assertFalse(changed)
        self.assertEqual(rpg.game_state.language, "de")
        self.assertEqual(rpg.game_state.settings["language"], "de")


class PricingCalculationTests(unittest.TestCase):
    def test_calculate_turn_cost_flash_lite(self):
        result = rpg.calculate_turn_cost("gemini-2.5-flash-lite", 1000, 2000)
        self.assertIsNotNone(result)
        expected_prompt = (1000 / 1_000_000) * 0.1
        expected_completion = (2000 / 1_000_000) * 0.4
        self.assertAlmostEqual(result["prompt_usd"], expected_prompt)
        self.assertAlmostEqual(result["completion_usd"], expected_completion)
        self.assertAlmostEqual(result["total_usd"], expected_prompt + expected_completion)

    def test_calculate_turn_cost_unknown_model(self):
        self.assertIsNone(rpg.calculate_turn_cost("unknown-model", 100, 200))

    def test_calculate_turn_cost_grok_4(self):
        result = rpg.calculate_turn_cost("grok-4", 200_000, 100_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.6)
        self.assertAlmostEqual(result["completion_usd"], 1.5)
        self.assertAlmostEqual(result["total_usd"], 2.1)

    def test_calculate_turn_cost_grok_code_fast(self):
        result = rpg.calculate_turn_cost("grok-code-fast-1", 200_000, 100_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.04)
        self.assertAlmostEqual(result["completion_usd"], 0.15)
        self.assertAlmostEqual(result["total_usd"], 0.19)

    def test_calculate_turn_cost_grok_fast_reasoning(self):
        result = rpg.calculate_turn_cost("grok-4-fast-reasoning", 1_000_000, 1_000_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.2)
        self.assertAlmostEqual(result["completion_usd"], 0.5)
        self.assertAlmostEqual(result["total_usd"], 0.7)

    def test_calculate_turn_cost_exceeds_known_tier_marks_unknown(self):
        result = rpg.calculate_turn_cost("grok-4", 300_000, 1_000)

        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "grok-4")
        self.assertIsNone(result["prompt_usd"])
        self.assertAlmostEqual(result["completion_usd"], (1_000 / 1_000_000) * 15.0)
        self.assertIsNone(result["total_usd"])


class VideoCostCalculationTests(unittest.TestCase):
    def test_calculate_video_cost_uses_default_tier(self):
        result = rpg.calculate_video_cost("veo-3.0-generate-001", seconds=5, tier="default")

        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "veo-3.0-generate-001")
        self.assertEqual(result["tier"], "default")
        self.assertEqual(result["seconds"], 5.0)
        self.assertAlmostEqual(result["usd_per_second"], 0.4)
        self.assertAlmostEqual(result["cost_usd"], 2.0)

    def test_calculate_video_cost_falls_back_when_tier_missing(self):
        custom_pricing = {
            "premium": {"usd_per_second": 0.9},
            "legacy": {"usd_per_second": 0.4},
        }
        with mock.patch.dict(rpg.VIDEO_MODEL_PRICES, {"custom-video": custom_pricing}, clear=False):
            result = rpg.calculate_video_cost("custom-video", seconds=3, tier="unknown")

        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], "premium")
        self.assertEqual(result["seconds"], 3.0)
        self.assertAlmostEqual(result["usd_per_second"], 0.9)
        self.assertAlmostEqual(result["cost_usd"], 2.7)


class ImageCostCalculationTests(unittest.TestCase):
    def test_calculate_image_cost_uses_standard_tier(self):
        result = rpg.calculate_image_cost("gemini-2.5-flash-image-preview", tier="standard", images=2)

        self.assertIsNotNone(result)
        self.assertEqual(result["model"], "gemini-2.5-flash-image-preview")
        self.assertEqual(result["tier"], "standard")
        self.assertEqual(result["images"], 2)
        self.assertAlmostEqual(result["usd_per_image"], 0.039, places=3)
        self.assertEqual(result["tokens_per_image"], 1290)
        self.assertAlmostEqual(result["cost_usd"], 0.078, places=3)

    def test_calculate_image_cost_falls_back_to_available_tier(self):
        custom_pricing = {
            "alternate": {
                "usd_per_image": 0.05,
                "tokens_per_image": 900,
            }
        }
        with mock.patch.dict(rpg.IMAGE_MODEL_PRICES, {"custom-model": custom_pricing}, clear=False):
            result = rpg.calculate_image_cost("custom-model", tier="missing", images=3)

        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], "alternate")
        self.assertEqual(result["images"], 3)
        self.assertAlmostEqual(result["usd_per_image"], 0.05)
        self.assertEqual(result["tokens_per_image"], 900)
        self.assertAlmostEqual(result["cost_usd"], 0.15)


class RecordImageUsageTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_record_image_usage_updates_scene_costs_and_counters(self):
        rpg.game_state.turn_index = 4

        rpg.record_image_usage(
            "gemini-2.5-flash-image-preview",
            purpose="scene",
            tier="standard",
            images=2,
        )

        self.assertEqual(rpg.game_state.last_image_model, "gemini-2.5-flash-image-preview")
        self.assertEqual(rpg.game_state.last_image_kind, "scene")
        self.assertEqual(rpg.game_state.last_image_count, 2)
        self.assertEqual(rpg.game_state.last_image_turn_index, 4)
        self.assertEqual(rpg.game_state.last_image_tier, "standard")
        self.assertAlmostEqual(rpg.game_state.last_image_cost_usd, 0.078, places=3)
        self.assertAlmostEqual(rpg.game_state.last_image_usd_per_image, 0.039, places=3)
        self.assertEqual(rpg.game_state.last_image_tokens, 1290)
        self.assertAlmostEqual(rpg.game_state.session_image_cost_usd, 0.078, places=3)
        self.assertEqual(rpg.game_state.session_image_requests, 2)
        self.assertEqual(rpg.game_state.session_image_kind_counts["scene"], 2)
        self.assertEqual(rpg.game_state.current_turn_index_for_image_counts, 4)
        self.assertEqual(rpg.game_state.current_turn_image_counts["scene"], 2)
        self.assertEqual(rpg.game_state.image_counts_by_turn[4]["scene"], 2)
        self.assertEqual(rpg.game_state.last_scene_image_model, "gemini-2.5-flash-image-preview")
        self.assertAlmostEqual(rpg.game_state.last_scene_image_cost_usd, 0.078, places=3)
        self.assertAlmostEqual(rpg.game_state.last_scene_image_usd_per_image, 0.039, places=3)
        self.assertEqual(rpg.game_state.last_scene_image_turn_index, 4)

    def test_record_image_usage_handles_missing_pricing(self):
        rpg.game_state.turn_index = 1

        rpg.record_image_usage(
            "custom/unknown-model",
            purpose="sketch",
            tier="premium",
            images=0,
        )

        self.assertEqual(rpg.game_state.last_image_model, "unknown-model")
        self.assertEqual(rpg.game_state.last_image_kind, "sketch")
        self.assertEqual(rpg.game_state.last_image_count, 0)
        self.assertEqual(rpg.game_state.last_image_turn_index, 1)
        self.assertEqual(rpg.game_state.last_image_tier, "premium")
        self.assertIsNone(rpg.game_state.last_image_cost_usd)
        self.assertIsNone(rpg.game_state.last_image_usd_per_image)
        self.assertIsNone(rpg.game_state.last_image_tokens)
        self.assertEqual(rpg.game_state.session_image_cost_usd, 0.0)
        self.assertEqual(rpg.game_state.session_image_requests, 0)
        self.assertNotIn("sketch", rpg.game_state.session_image_kind_counts)
        self.assertEqual(rpg.game_state.current_turn_index_for_image_counts, 1)
        self.assertEqual(rpg.game_state.current_turn_image_counts["other"], 0)
        self.assertEqual(rpg.game_state.image_counts_by_turn[1]["other"], 0)
        self.assertIsNone(rpg.game_state.last_scene_image_model)
        self.assertIsNone(rpg.game_state.last_scene_image_cost_usd)
        self.assertIsNone(rpg.game_state.last_scene_image_turn_index)

    def test_record_image_usage_respects_explicit_turn_index(self):
        rpg.game_state.turn_index = 9

        rpg.record_image_usage(
            "gemini-2.5-flash-image-preview",
            purpose="scene",
            images=1,
            turn_index=3,
        )

        self.assertEqual(rpg.game_state.last_image_turn_index, 3)
        self.assertEqual(rpg.game_state.last_scene_image_turn_index, 3)
        self.assertEqual(rpg.game_state.session_image_requests, 1)
        self.assertEqual(rpg.game_state.current_turn_index_for_image_counts, 3)
        self.assertEqual(rpg.game_state.current_turn_image_counts.get("scene"), 1)
        self.assertEqual(rpg.game_state.image_counts_by_turn[3].get("scene"), 1)


class RecordVideoUsageTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_record_video_usage_updates_costs_and_counters(self):
        rpg.game_state.turn_index = 5

        rpg.record_video_usage(
            "veo-3.0-generate-001",
            seconds=4,
            tier="default",
        )

        self.assertEqual(rpg.game_state.last_video_model, "veo-3.0-generate-001")
        self.assertEqual(rpg.game_state.last_video_tier, "default")
        self.assertEqual(rpg.game_state.last_video_turn_index, 5)
        self.assertAlmostEqual(rpg.game_state.last_video_seconds, 4.0)
        self.assertAlmostEqual(rpg.game_state.last_video_cost_usd, 1.6)
        self.assertAlmostEqual(rpg.game_state.last_video_usd_per_second, 0.4)
        self.assertAlmostEqual(rpg.game_state.session_video_cost_usd, 1.6)
        self.assertAlmostEqual(rpg.game_state.session_video_seconds, 4.0)
        self.assertEqual(rpg.game_state.session_video_requests, 1)

    def test_record_video_usage_handles_missing_pricing(self):
        rpg.record_video_usage(
            "custom/unknown-video",
            seconds=7.5,
            tier="premium",
            turn_index=11,
        )

        self.assertEqual(rpg.game_state.last_video_model, "unknown-video")
        self.assertEqual(rpg.game_state.last_video_tier, "premium")
        self.assertEqual(rpg.game_state.last_video_turn_index, 11)
        self.assertAlmostEqual(rpg.game_state.last_video_seconds, 7.5)
        self.assertIsNone(rpg.game_state.last_video_cost_usd)
        self.assertIsNone(rpg.game_state.last_video_usd_per_second)
        self.assertEqual(rpg.game_state.session_video_requests, 1)
        self.assertAlmostEqual(rpg.game_state.session_video_seconds, 7.5)
        self.assertEqual(rpg.game_state.session_video_cost_usd, 0.0)


class SettingsFileTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.original_settings_path = rpg.SETTINGS_FILE
        rpg.SETTINGS_FILE = Path(self.tempdir.name) / "settings.json"
        self.addCleanup(self._restore_settings_path)

    def _restore_settings_path(self):
        rpg.SETTINGS_FILE = self.original_settings_path

    def test_ensure_settings_file_creates_defaults(self):
        rpg.ensure_settings_file()

        self.assertTrue(rpg.SETTINGS_FILE.exists())
        data = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(data, rpg.DEFAULT_SETTINGS)

    def test_ensure_settings_file_preserves_existing_contents(self):
        custom = {"world_style": "Noir", "gemini_api_key": "sk-123"}
        rpg.SETTINGS_FILE.write_text(json.dumps(custom))

        rpg.ensure_settings_file()

        self.assertEqual(json.loads(rpg.SETTINGS_FILE.read_text()), custom)

    def test_load_settings_merges_with_defaults(self):
        overrides = {"world_style": "Cyberpunk", "custom": "value"}
        rpg.SETTINGS_FILE.write_text(json.dumps(overrides))

        loaded = rpg.load_settings()

        self.assertEqual(loaded["world_style"], "Cyberpunk")
        self.assertEqual(loaded["custom"], "value")
        self.assertEqual(loaded["text_model"], rpg.DEFAULT_SETTINGS["text_model"])
        self.assertIsNot(loaded, rpg.DEFAULT_SETTINGS)

    def test_load_settings_returns_defaults_on_invalid_json(self):
        rpg.SETTINGS_FILE.write_text("not-json")

        loaded = rpg.load_settings()

        self.assertEqual(loaded, rpg.DEFAULT_SETTINGS)
        self.assertIsNot(loaded, rpg.DEFAULT_SETTINGS)

    def test_load_settings_normalizes_language_alias(self):
        rpg.SETTINGS_FILE.write_text(json.dumps({"language": "German (Standard)"}))

        loaded = rpg.load_settings()

        self.assertEqual(loaded["language"], "de")

    def test_load_settings_normalizes_history_mode(self):
        rpg.SETTINGS_FILE.write_text(json.dumps({"history_mode": "SUMMARY"}))

        loaded = rpg.load_settings()

        self.assertEqual(loaded["history_mode"], rpg.HISTORY_MODE_SUMMARY)

        rpg.SETTINGS_FILE.write_text(json.dumps({"history_mode": "timeline"}))

        loaded = rpg.load_settings()

        self.assertEqual(loaded["history_mode"], rpg.HISTORY_MODE_FULL)


class SaveSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.original_settings_path = rpg.SETTINGS_FILE
        rpg.SETTINGS_FILE = Path(self.tempdir.name) / "settings.json"
        self.addCleanup(self._restore_settings_path)

    def _restore_settings_path(self):
        rpg.SETTINGS_FILE = self.original_settings_path

    async def test_save_settings_writes_json(self):
        payload = rpg.DEFAULT_SETTINGS.copy()
        payload["difficulty"] = "Impossible"

        await rpg.save_settings(payload)

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["difficulty"], "Impossible")
        self.assertEqual(on_disk, payload)

    async def test_save_settings_preserves_existing_keys(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["gemini_api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({"world_style": "Noir"})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["gemini_api_key"], "sk-existing")
        self.assertEqual(on_disk["elevenlabs_api_key"], "voice-existing")
        self.assertEqual(on_disk["world_style"], "Noir")

    async def test_save_settings_normalizes_language_alias(self):
        await rpg.save_settings({"language": "Deutsch"})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["language"], "de")

    async def test_save_settings_normalizes_history_mode(self):
        await rpg.save_settings({"history_mode": " Summary "})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["history_mode"], rpg.HISTORY_MODE_SUMMARY)

        await rpg.save_settings({"history_mode": "timeline"})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["history_mode"], rpg.HISTORY_MODE_FULL)

    async def test_save_settings_normalizes_state_when_updating_global_dict(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["language"] = "German (Standard)"
        existing["history_mode"] = " Summary "
        rpg.game_state.settings = existing

        await rpg.save_settings(rpg.game_state.settings)

        self.assertIs(rpg.game_state.settings, existing)
        self.assertEqual(rpg.game_state.settings["language"], "de")
        self.assertEqual(rpg.game_state.settings["history_mode"], rpg.HISTORY_MODE_SUMMARY)

    async def test_save_settings_discards_transient_keys(self):
        await rpg.save_settings({
            "world_style": "Gothic",
            "gemini_api_key_preview": "partial",
            "gemini_api_key_set": True,
            "api_key_preview": "legacy",
            "api_key_set": False,
            "elevenlabs_api_key_preview": "voice-partial",
            "elevenlabs_api_key_set": True,
        })

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["world_style"], "Gothic")
        self.assertNotIn("gemini_api_key_preview", on_disk)
        self.assertNotIn("gemini_api_key_set", on_disk)
        self.assertNotIn("api_key_preview", on_disk)
        self.assertNotIn("api_key_set", on_disk)
        self.assertNotIn("elevenlabs_api_key_preview", on_disk)
        self.assertNotIn("elevenlabs_api_key_set", on_disk)

    async def test_save_settings_ignores_blank_api_key(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["gemini_api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({"gemini_api_key": "   "})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["gemini_api_key"], "sk-existing")
        self.assertEqual(on_disk["elevenlabs_api_key"], "voice-existing")

    async def test_save_settings_strips_whitespace_api_keys(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["gemini_api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({
            "gemini_api_key": "  sk-new  ",
            "elevenlabs_api_key": "\tvoice-new\n",
        })

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["gemini_api_key"], "sk-new")
        self.assertEqual(on_disk["elevenlabs_api_key"], "voice-new")


class ResolveTurnFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        self.pid = "p1"
        rpg.game_state.players[self.pid] = rpg.Player(
            id=self.pid,
            name="Alice",
            background="Cleric",
            token="tok-turn",
            pending_join=True,
        )
        rpg.game_state.submissions[self.pid] = "Provide healing"

    async def test_pending_join_triggers_announcement_and_clears_state(self):
        structured = rpg.TurnStructured(
            narrative="The caverns tremble",
            image_prompt="A holy light pierces darkness",
            public_statuses=[rpg.PublicStatus(player_id=self.pid, status_word="Ready for battle")],
            updates=[
                rpg.PlayerUpdate(
                    player_id=self.pid,
                    character_class="Cleric",
                    abilities=[rpg.Ability(name="Radiance", expertise="novice")],
                    inventory=["mace"],
                    conditions=["inspired"],
                )
            ],
        )

        with mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(return_value=structured)), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private_mock:
            await rpg.resolve_turn(initial=False)

        announce_mock.assert_awaited_once()
        self.assertIn("joined the party", announce_mock.await_args.args[0])
        self.assertEqual(rpg.game_state.turn_index, 1)
        self.assertEqual(rpg.game_state.current_narrative, "The caverns tremble")
        self.assertEqual(rpg.game_state.last_image_prompt, "A holy light pierces darkness")
        self.assertFalse(rpg.game_state.players[self.pid].pending_join)
        self.assertEqual(rpg.game_state.players[self.pid].character_class, "Cleric")
        self.assertEqual(rpg.game_state.players[self.pid].status_word, "ready")
        self.assertEqual(rpg.game_state.submissions, {})
        self.assertEqual(len(rpg.game_state.history), 1)
        self.assertFalse(rpg.game_state.lock.active)
        broadcast_mock.assert_awaited()
        send_private_mock.assert_awaited()

    async def test_failure_releases_lock_and_preserves_turn_index(self):
        async def failing_call(*_args, **_kwargs):
            raise RuntimeError("generation exploded")

        with mock.patch("rpg.request_turn_payload", new=mock.AsyncMock(side_effect=failing_call)), \
                mock.patch("rpg.announce", new=mock.AsyncMock()), \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock:
            with self.assertRaises(RuntimeError):
                await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.game_state.turn_index, 0)
        self.assertGreaterEqual(broadcast_mock.await_count, 2)
        self.assertFalse(rpg.game_state.lock.active)


class CreatePortraitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.players["p1"] = rpg.Player(
            id="p1",
            name="Alice",
            background="Mage",
            character_class="Wizard",
            status_word="resolute",
            token="tok-portrait",
        )

    async def test_unknown_player_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreatePortraitBody(player_id="ghost", token="tok-portrait")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIsNone(rpg.game_state.players["p1"].portrait)

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreatePortraitBody(player_id="p1", token="wrong")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIsNone(rpg.game_state.players["p1"].portrait)

    async def test_conflict_when_busy(self):
        rpg.game_state.lock = rpg.LockState(active=True, reason="resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(rpg.game_state.lock.active)
        self.assertEqual(rpg.game_state.lock.reason, "resolving_turn")
        broadcast.assert_not_awaited()
        self.assertIsNone(rpg.game_state.players["p1"].portrait)

    async def test_lock_released_on_generation_failure(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(side_effect=RuntimeError("fail"))) as gen:
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            with self.assertRaises(RuntimeError):
                await rpg.create_portrait(body)

        self.assertFalse(rpg.game_state.lock.active)
        self.assertEqual(rpg.game_state.lock.reason, "")
        self.assertIsNone(rpg.game_state.players["p1"].portrait)
        self.assertGreaterEqual(broadcast.await_count, 2)
        announce.assert_not_awaited()
        send_private.assert_not_awaited()
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "portrait")

    async def test_successful_generation_updates_state(self):
        expected_prompt = rpg.build_portrait_prompt(rpg.game_state.players["p1"])
        portrait_data = "data:image/png;base64," + base64.b64encode(b"portrait").decode("ascii")

        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(return_value=portrait_data)) as gen, \
                mock.patch("rpg.time.time", return_value=123.0):
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            result = await rpg.create_portrait(body)

        self.assertTrue(result["ok"])
        self.assertEqual(result["portrait"]["data_url"], portrait_data)
        self.assertEqual(result["portrait"]["prompt"], expected_prompt)
        self.assertEqual(result["portrait"]["updated_at"], 123.0)

        portrait = rpg.game_state.players["p1"].portrait
        self.assertIsNotNone(portrait)
        self.assertEqual(portrait.data_url, portrait_data)
        self.assertEqual(portrait.prompt, expected_prompt)
        self.assertEqual(portrait.updated_at, 123.0)

        self.assertGreaterEqual(broadcast.await_count, 2)
        announce.assert_awaited_once_with("Alice's portrait updated.")
        send_private.assert_awaited_once_with("p1")
        gen.assert_awaited_once()
        self.assertFalse(rpg.game_state.lock.active)

class PromptLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.original_prompt = rpg.PROMPT_FILE
        self.original_template = rpg.GM_PROMPT_TEMPLATE
        self.original_prompt_map = rpg.PROMPT_FILES.copy()
        self.original_cache = rpg._GM_PROMPT_CACHE.copy()
        self.addCleanup(self._restore_prompt)

    def _restore_prompt(self):
        rpg.PROMPT_FILE = self.original_prompt
        rpg.PROMPT_FILES.clear()
        rpg.PROMPT_FILES.update(self.original_prompt_map)
        rpg._GM_PROMPT_CACHE.clear()
        rpg._GM_PROMPT_CACHE.update(self.original_cache)
        rpg.GM_PROMPT_TEMPLATE = self.original_template

    def test_load_gm_prompt_reads_text(self):
        prompt_path = Path(self.tempdir.name) / "prompt.txt"
        prompt_path.write_text("Guide the heroes", encoding="utf-8")
        rpg.PROMPT_FILE = prompt_path
        rpg.PROMPT_FILES['en'] = prompt_path
        rpg._GM_PROMPT_CACHE.clear()

        text = rpg.load_gm_prompt()

        self.assertEqual(text, "Guide the heroes")

    def test_load_gm_prompt_missing_raises(self):
        rpg.PROMPT_FILE = Path(self.tempdir.name) / "missing.txt"
        rpg.PROMPT_FILES['en'] = rpg.PROMPT_FILE
        rpg._GM_PROMPT_CACHE.clear()

        with self.assertRaises(RuntimeError):
            rpg.load_gm_prompt()


class ThinkingDirectiveTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_build_thinking_directive_none_mode(self):
        rpg.game_state.settings["thinking_mode"] = "none"

        text = rpg.build_thinking_directive()

        self.assertIn("minimum internal deliberation", text)

    def test_build_thinking_directive_invalid_mode_defaults(self):
        rpg.game_state.settings["thinking_mode"] = "galactic"

        text = rpg.build_thinking_directive()

        self.assertIn("minimum internal deliberation", text)


class MakeGmInstructionTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.original_template = rpg.GM_PROMPT_TEMPLATE
        self.addCleanup(self._restore_template)

    def _restore_template(self):
        rpg.GM_PROMPT_TEMPLATE = self.original_template

    def test_instruction_replaces_token(self):
        rpg.GM_PROMPT_TEMPLATE = "Prelude\n<<TURN_DIRECTIVE>>\nEpilogue"
        rpg.game_state.settings["thinking_mode"] = "brief"

        text = rpg.make_gm_instruction(is_initial=True)

        self.assertTrue(text.startswith("Prelude\n"))
        self.assertIn("INITIAL TURN", text)
        self.assertIn("Take a short internal moment", text)
        self.assertTrue(text.endswith("Epilogue"))

    def test_instruction_appends_when_token_missing(self):
        rpg.GM_PROMPT_TEMPLATE = "Core guidance"
        rpg.game_state.settings["thinking_mode"] = "deep"

        text = rpg.make_gm_instruction(is_initial=False)

        self.assertTrue(text.startswith("Core guidance\n"))
        self.assertIn("ONGOING TURN", text)
        self.assertIn("thorough internal reasoning", text)


class LanguageEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        player = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-lang")
        rpg.game_state.players[player.id] = player

    async def test_set_language_updates_state(self):
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            result = await rpg.set_language(rpg.LanguageBody(language="de"))

        self.assertEqual(result, {"language": "de"})
        self.assertEqual(rpg.game_state.language, "de")
        self.assertEqual(rpg.game_state.settings["language"], "de")
        save.assert_awaited_once()
        broadcast.assert_awaited_once()

    async def test_set_language_normalizes_invalid_value(self):
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            result = await rpg.set_language(rpg.LanguageBody(language="fr"))

        self.assertEqual(result, {"language": "en"})
        self.assertEqual(rpg.game_state.language, "en")
        save.assert_not_awaited()
        broadcast.assert_not_awaited()

    async def test_set_language_authenticates_when_credentials_provided(self):
        with mock.patch("rpg.authenticate_player", wraps=rpg.authenticate_player) as auth:
            await rpg.set_language(rpg.LanguageBody(language="de", player_id="p1", token="tok-lang"))

        auth.assert_called_once_with("p1", "tok-lang")


class NextTurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-turn")

    async def test_next_turn_invokes_resolve(self):
        body = rpg.NextTurnBody(player_id="p1", token="tok-turn")
        with mock.patch("rpg.resolve_turn", new=mock.AsyncMock(return_value=None)) as resolve_mock:
            result = await rpg.next_turn(body)

        self.assertEqual(result, {"ok": True})
        resolve_mock.assert_awaited_once_with(initial=False)

    async def test_next_turn_rejects_invalid_token(self):
        body = rpg.NextTurnBody(player_id="p1", token="wrong")
        with mock.patch("rpg.resolve_turn", new=mock.AsyncMock()) as resolve_mock:
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.next_turn(body)

        self.assertEqual(ctx.exception.status_code, 403)
        resolve_mock.assert_not_awaited()


class GameStateHelperTests(unittest.TestCase):
    def test_tokens_per_second_returns_none_without_metrics(self):
        state = rpg.GameState()
        self.assertIsNone(state.tokens_per_second())

    def test_tokens_per_second_computes_value(self):
        state = rpg.GameState()
        state.last_token_usage = {"input": 100, "output": 50, "thinking": 10}
        state.last_turn_runtime = 4
        self.assertEqual(state.tokens_per_second(), 40.0)

    def test_private_snapshot_for_unknown_player_is_empty(self):
        state = rpg.GameState()
        self.assertEqual(state.private_snapshot_for("missing"), {})

    def test_private_snapshot_for_existing_player(self):
        ability = rpg.Ability(name="Arcana", expertise="expert")
        player = rpg.Player(
            id="p1",
            name="Alice",
            background="Scholar",
            character_class="Mage",
            abilities=[ability],
            pending_join=False,
        )
        state = rpg.GameState(players={"p1": player})
        snapshot = state.private_snapshot_for("p1")

        self.assertEqual(snapshot["you"]["id"], "p1")
        self.assertEqual(snapshot["you"]["cls"], "Mage")
        self.assertEqual(snapshot["you"]["abilities"], [ability.model_dump(by_alias=True)])
        self.assertFalse(snapshot["you"]["pending_join"])


class AuthenticatePlayerTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.player = rpg.Player(id="p1", name="Alice", background="Mage", token="secret")
        rpg.game_state.players[self.player.id] = self.player

    def test_authenticate_player_success(self):
        result = rpg.authenticate_player("p1", "secret")
        self.assertIs(result, self.player)

    def test_authenticate_player_unknown_id(self):
        with self.assertRaises(rpg.HTTPException) as ctx:
            rpg.authenticate_player("missing", "secret")

        self.assertEqual(ctx.exception.status_code, 404)

    def test_authenticate_player_invalid_token(self):
        with self.assertRaises(rpg.HTTPException) as ctx:
            rpg.authenticate_player("p1", "bad")

        self.assertEqual(ctx.exception.status_code, 403)


class RequireGeminiApiKeyTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_require_gemini_api_key_returns_value(self):
        rpg.game_state.settings["gemini_api_key"] = "sk-valid"

        result = rpg.require_gemini_api_key()

        self.assertEqual(result, "sk-valid")

    def test_require_gemini_api_key_raises_when_missing(self):
        rpg.game_state.settings["gemini_api_key"] = ""

        with self.assertRaises(rpg.HTTPException) as ctx:
            rpg.require_gemini_api_key()

        self.assertEqual(ctx.exception.status_code, 400)


class FetchPublicIpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_public_ip_cache()

    async def test_prefers_ipv6_and_caches_result(self):
        calls = []

        class DummyResponse:
            def __init__(self, text):
                self.text = text
                self.status_code = 200

            def raise_for_status(self):
                pass

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                calls.append(url)
                return DummyResponse(" 2001:db8::1 ")

        with mock.patch("rpg.httpx.AsyncClient", DummyClient), \
                mock.patch("rpg.time.time", side_effect=[100.0, 150.0]):
            first = await rpg.fetch_public_ip()
            second = await rpg.fetch_public_ip()

        self.assertEqual(first, "2001:db8::1")
        self.assertEqual(second, "2001:db8::1")
        self.assertEqual(calls, ["https://api64.ipify.org"])
        self.assertEqual(rpg.PUBLIC_IP_CACHE["cached_at"], 100.0)

    async def test_failure_short_circuits_during_cooldown(self):
        class BoomClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                raise RuntimeError("boom")

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with mock.patch("rpg.httpx.AsyncClient", BoomClient), \
                mock.patch("rpg.time.time", return_value=200.0):
            result = await rpg.fetch_public_ip()

        self.assertIsNone(result)
        self.assertEqual(rpg.PUBLIC_IP_CACHE["last_failure_at"], 200.0)

        with mock.patch("rpg.httpx.AsyncClient") as client_cls, \
                mock.patch("rpg.time.time", return_value=250.0):
            second_result = await rpg.fetch_public_ip()

        self.assertIsNone(second_result)
        client_cls.assert_not_called()


class GetPublicUrlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        reset_public_ip_cache()

    async def test_non_loopback_host_returns_inferred_url(self):
        request = types.SimpleNamespace(base_url=rpg.httpx.URL("https://party.example:8443/"))
        with mock.patch("rpg.fetch_public_ip", new=mock.AsyncMock()) as fetch_mock:
            result = await rpg.get_public_url(request)

        self.assertEqual(result["url"], "https://party.example:8443/")
        self.assertEqual(result["source"], "inferred_host")
        fetch_mock.assert_not_awaited()

    async def test_loopback_uses_public_ip_with_ipv6(self):
        request = types.SimpleNamespace(base_url=rpg.httpx.URL("http://localhost:8000/"))
        with mock.patch("rpg.fetch_public_ip", new=mock.AsyncMock(return_value="2001:db8::1234")):
            result = await rpg.get_public_url(request)

        self.assertEqual(result["url"], "http://[2001:db8::1234]:8000/")
        self.assertEqual(result["source"], "public_ip")

    async def test_placeholder_used_when_public_ip_missing(self):
        request = types.SimpleNamespace(base_url=rpg.httpx.URL("http://localhost/"))
        with mock.patch("rpg.fetch_public_ip", new=mock.AsyncMock(return_value=None)):
            result = await rpg.get_public_url(request)

        self.assertEqual(result["url"], "http://<your-public-ip-or-domain>/")
        self.assertEqual(result["source"], "placeholder")


class GeminiGenerateJsonTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.settings["gemini_api_key"] = "sk-test"

    async def test_generate_json_parses_and_tracks_usage(self):
        class DummyResponse:
            status_code = 200

            def json(self_inner):
                payload = {
                    "nar": "Narrative",
                    "img": "Prompt",
                    "pub": [{"pid": "p1", "word": "Brave"}],
                    "upd": [
                        {
                            "pid": "p1",
                            "cls": "Rogue",
                            "ab": [{"n": "Stealth", "x": "expert"}],
                            "inv": ["dagger"],
                            "cond": ["ready"],
                        }
                    ],
                }
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": json.dumps(payload)}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 8,
                        "thoughtsTokenCount": 4,
                    },
                }

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, headers, json):
                self.url = url
                self.headers = headers
                self.body = json
                return DummyResponse()

        schema = rpg.build_turn_schema()
        with mock.patch("httpx.AsyncClient", DummyClient):
            result = await rpg.request_turn_payload(
                model="gemini-2.5-flash-lite",
                system_prompt="sys",
                user_payload={"foo": "bar"},
                schema=schema,
            )

        self.assertIsInstance(result, rpg.TurnStructured)
        self.assertEqual(result.narrative, "Narrative")
        self.assertEqual(rpg.game_state.last_token_usage["input"], 12)
        self.assertEqual(rpg.game_state.last_token_usage["thinking"], 4)
        self.assertIsNotNone(rpg.game_state.last_turn_runtime)

    async def test_generate_json_raises_on_http_error(self):
        class DummyResponse:
            status_code = 500
            text = "boom"

            def json(self_inner):
                return {}

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyClient):
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.request_turn_payload(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    user_payload={},
                    schema=rpg.build_turn_schema(),
                )

        self.assertEqual(ctx.exception.status_code, 502)


class GeminiGenerateImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.game_state.settings["gemini_api_key"] = "sk-test"

    async def test_generate_image_returns_data_url(self):
        class DummyResponse:
            status_code = 200

            def json(self_inner):
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "data": "abc",
                                            "mimeType": "image/png",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }

        class DummyClient:
            last_json = None

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                DummyClient.last_json = kwargs.get("json")
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyClient), \
             mock.patch("rpg._archive_generated_media") as archive_mock:
            data_url = await rpg.gemini_generate_image(
                "gemini-2.5-flash-image-preview",
                "prompt",
                purpose="scene",
            )

        expected_request = {
            "contents": [
                {"role": "user", "parts": [{"text": "prompt"}]}
            ]
        }
        self.assertEqual(DummyClient.last_json, expected_request)
        self.assertEqual(data_url, "data:image/png;base64,abc")
        self.assertEqual(rpg.game_state.last_image_model, "gemini-2.5-flash-image-preview")
        self.assertEqual(rpg.game_state.session_image_requests, 1)
        self.assertEqual(rpg.game_state.session_image_kind_counts, {"scene": 1})
        self.assertAlmostEqual(rpg.game_state.last_image_cost_usd, 0.039, places=3)
        self.assertAlmostEqual(rpg.game_state.session_image_cost_usd, 0.039, places=3)
        self.assertAlmostEqual(rpg.game_state.last_scene_image_cost_usd, 0.039, places=3)
        self.assertIsInstance(rpg.game_state.current_turn_index_for_image_counts, int)
        self.assertEqual(rpg.game_state.current_turn_image_counts, {"scene": 1})
        tracked_turn = rpg.game_state.current_turn_index_for_image_counts
        self.assertEqual(rpg.game_state.image_counts_by_turn[tracked_turn], {"scene": 1})
        self.assertTrue(archive_mock.called)
        archived_args, archived_kwargs = archive_mock.call_args
        self.assertIsInstance(archived_args[0], bytes)
        self.assertEqual(archived_kwargs.get("prefix"), "image_scene")

    async def test_generate_image_skips_invalid_portrait_data(self):
        player = rpg.Player(id="p1", name="Iris", background="Mystic", character_class="Mage", token="tok")
        player.portrait = rpg.PlayerPortrait(
            data_url="data:image/png;base64",
            prompt="Portrait prompt",
            updated_at=456.0,
        )
        rpg.game_state.players[player.id] = player

        class DummyResponse:
            status_code = 200

            def json(self_inner):
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "data": "abc",
                                            "mimeType": "image/png",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }

        class DummyClient:
            last_json = None

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                DummyClient.last_json = kwargs.get("json")
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyClient):
            await rpg.gemini_generate_image(
                "gemini-2.5-flash-image-preview",
                "ruined shrine at dusk",
                purpose="scene",
            )

        request_body = DummyClient.last_json
        self.assertIsNotNone(request_body)
        parts = request_body["contents"][0]["parts"]
        self.assertEqual(parts, [{"text": "ruined shrine at dusk"}])

    async def test_generate_image_raises_when_missing_data(self):
        class DummyResponse:
            status_code = 200

            def json(self_inner):
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "no image"}
                                ]
                            }
                        }
                    ]
                }

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return DummyResponse()

        with mock.patch("httpx.AsyncClient", DummyClient):
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.gemini_generate_image(
                    "gemini-2.5-flash-image-preview",
                    "prompt",
                    purpose="scene",
                )

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(rpg.game_state.session_image_requests, 0)
        self.assertEqual(rpg.game_state.session_image_cost_usd, 0.0)
        self.assertEqual(rpg.game_state.current_turn_image_counts, {})
        self.assertIsNone(rpg.game_state.current_turn_index_for_image_counts)
        self.assertEqual(rpg.game_state.image_counts_by_turn, {})


class WebsocketEndpointTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_authenticated_disconnect_marks_player_pending_leave(self):
        player = rpg.Player(
            id="p1",
            name="Aurora",
            background="Seer",
            token="tok",
            pending_join=False,
        )
        rpg.game_state.players[player.id] = player

        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.schedule_session_reset_check") as reset_mock:
            with TestClient(rpg.app) as client:
                with client.websocket_connect("/ws?player_id=p1&auth_token=tok") as ws:
                    state_payload = ws.receive_json()
                    self.assertEqual(state_payload["event"], "state")
                    private_payload = ws.receive_json()
                    self.assertEqual(private_payload["event"], "private")
                    self.assertTrue(player.connected)
                    self.assertFalse(player.pending_leave)

            self.assertFalse(player.connected)
            self.assertTrue(player.pending_leave)
            self.assertFalse(rpg.game_state.global_sockets)
            self.assertGreaterEqual(broadcast_mock.await_count, 1)
            announce_mock.assert_awaited_once()
            reset_mock.assert_called_once()


class SessionResetHelperTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_reset_clears_session_progress_preserving_settings(self):
        rpg.game_state.settings["world_style"] = "Solarpunk"
        rpg.game_state.turn_index = 7
        rpg.game_state.current_narrative = "Ancient ruins loom"
        rpg.game_state.session_token_usage = {"input": 10, "output": 5, "thinking": 1}
        rpg.game_state.last_image_prompt = "old prompt"
        pid = "departing"
        rpg.game_state.players[pid] = rpg.Player(
            id=pid,
            name="Drifter",
            background="Nomad",
            token="tok-depart",
            connected=False,
            pending_join=False,
            pending_leave=True,
        )

        did_reset = rpg.reset_session_if_inactive()

        self.assertTrue(did_reset)
        self.assertEqual(rpg.game_state.settings["world_style"], "Solarpunk")
        self.assertEqual(rpg.game_state.players, {})
        self.assertEqual(rpg.game_state.turn_index, 0)
        self.assertEqual(rpg.game_state.current_narrative, "")
        self.assertEqual(rpg.game_state.session_token_usage, {"input": 0, "output": 0, "thinking": 0})
        self.assertIsNone(rpg.game_state.last_image_prompt)

    def test_reset_skipped_when_active_player_remains(self):
        pid = "active"
        rpg.game_state.players[pid] = rpg.Player(
            id=pid,
            name="Guardian",
            background="Knight",
            token="tok-active",
            connected=True,
            pending_join=False,
            pending_leave=False,
        )
        rpg.game_state.turn_index = 3

        did_reset = rpg.reset_session_if_inactive()

        self.assertFalse(did_reset)
        self.assertIn(pid, rpg.game_state.players)
        self.assertEqual(rpg.game_state.turn_index, 3)


class SessionResetJoinTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_join_clears_stale_players_and_runs_initial_resolve(self):
        rpg.game_state.settings["world_style"] = "Clockwork"
        old_pid = "old"
        rpg.game_state.players[old_pid] = rpg.Player(
            id=old_pid,
            name="Elder",
            background="Scholar",
            token="tok-old",
            connected=False,
            pending_join=False,
            pending_leave=True,
        )
        rpg.game_state.turn_index = 4
        rpg.game_state.current_narrative = "The tale lingers"

        async def fake_resolve_turn(initial: bool = False):
            self.assertTrue(initial)
            for player in rpg.game_state.players.values():
                player.pending_join = False
            for pid in list(rpg.game_state.players.keys()):
                if rpg.game_state.players[pid].pending_leave:
                    rpg.game_state.players.pop(pid, None)
            rpg.game_state.current_narrative = "Fresh chapter"
            rpg.game_state.turn_index = 1

        original_resolve = rpg.resolve_turn
        rpg.resolve_turn = fake_resolve_turn
        try:
            result = await rpg.join_game(rpg.JoinBody(name="Nova", background="Ranger"))
        finally:
            rpg.resolve_turn = original_resolve

        new_pid = result["player_id"]
        self.assertNotEqual(new_pid, old_pid)
        self.assertTrue(result["auth_token"])
        self.assertEqual(rpg.game_state.settings["world_style"], "Clockwork")
        self.assertEqual(len(rpg.game_state.players), 1)
        self.assertIn(new_pid, rpg.game_state.players)
        self.assertFalse(rpg.game_state.players[new_pid].pending_join)
        self.assertEqual(rpg.game_state.current_narrative, "Fresh chapter")
        self.assertNotIn(old_pid, rpg.game_state.players)


class HistorySummaryAdvancedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_state()
        rpg.game_state.history = [
            rpg.TurnRecord(index=1, narrative="A brave start", image_prompt="", timestamp=1.0),
            rpg.TurnRecord(index=2, narrative="An ominous clue", image_prompt="", timestamp=2.0),
        ]
        rpg.game_state.history_summary = ["Old bullet"]
        rpg.game_state.turn_index = 2
        rpg.game_state.settings["history_mode"] = "summary"

    async def asyncTearDown(self) -> None:
        reset_state()

    async def test_falls_back_when_api_key_missing(self) -> None:
        expected_fallback = rpg._fallback_summary_from_history()
        with mock.patch("rpg.require_text_api_key", side_effect=HTTPException(status_code=401, detail="missing")):
            await rpg.update_history_summary(rpg.game_state.history[-1])
        self.assertEqual(rpg.game_state.history_summary, expected_fallback)

    async def test_ensures_bullets_from_model_response(self) -> None:
        summary_payload = rpg.SummaryStructured(summary=[
            "First highlight",
            "* second item",
            " ",
            "- already bullet",
        ])
        with (
            mock.patch("rpg.require_text_api_key", return_value=None),
            mock.patch("rpg.request_summary_payload", return_value=summary_payload),
        ):
            await rpg.update_history_summary(rpg.game_state.history[-1])
        self.assertEqual(
            rpg.game_state.history_summary,
            [
                "- First highlight",
                "- second item",
                "- No events recorded.",
                "- already bullet",
            ][: rpg.MAX_HISTORY_SUMMARY_BULLETS],
        )

if __name__ == "__main__":
    unittest.main()
