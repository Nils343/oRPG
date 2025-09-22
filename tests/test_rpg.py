import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import rpg


def reset_state():
    """Utility to return the global game state to a clean baseline for tests."""
    rpg.STATE.settings = rpg.DEFAULT_SETTINGS.copy()
    rpg.STATE.players.clear()
    rpg.STATE.submissions.clear()
    rpg.STATE.current_scenario = ""
    rpg.STATE.turn_index = 0
    rpg.STATE.history.clear()
    rpg.STATE.history_summary.clear()
    rpg.STATE.lock = rpg.LockState(False, "")
    rpg.STATE.global_sockets.clear()
    rpg.STATE.language = rpg.DEFAULT_LANGUAGE
    rpg.STATE.last_image_data_url = None
    rpg.STATE.last_image_prompt = None
    rpg.STATE.last_image_model = None
    rpg.STATE.last_image_tier = None
    rpg.STATE.last_image_kind = None
    rpg.STATE.last_image_cost_usd = None
    rpg.STATE.last_image_usd_per = None
    rpg.STATE.last_image_tokens = None
    rpg.STATE.last_image_count = 0
    rpg.STATE.last_image_turn_index = None
    rpg.STATE.last_token_usage = {}
    rpg.STATE.last_turn_runtime = None
    rpg.STATE.session_token_usage = {"input": 0, "output": 0, "thinking": 0}
    rpg.STATE.session_request_count = 0
    rpg.STATE.last_cost_usd = None
    rpg.STATE.session_cost_usd = 0.0
    rpg.STATE.session_image_cost_usd = 0.0
    rpg.STATE.session_image_requests = 0
    rpg.STATE.session_image_kind_counts = {}
    rpg.STATE.last_scene_image_cost_usd = None
    rpg.STATE.last_scene_image_usd_per = None
    rpg.STATE.last_scene_image_model = None
    rpg.STATE.last_scene_image_turn_index = None
    rpg.STATE.turn_image_kind_counts = {}


def reset_public_ip_cache():
    rpg.PUBLIC_IP_CACHE.clear()
    rpg.PUBLIC_IP_CACHE.update({"value": None, "timestamp": 0.0, "failure_ts": 0.0})


class SanitizeNarrativeTests(unittest.TestCase):
    def test_decodes_unicode_escape_sequences(self):
        raw = "Die H\\u00fctte steht am Flu\\u00dfufer."
        cleaned = rpg.sanitize_narrative(raw)
        self.assertEqual(cleaned, "Die Hütte steht am Flussufer.")


class ResolveTurnStatusWordTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_blank_status_word_becomes_unknown(self):
        pid = "player1"
        rpg.STATE.players[pid] = rpg.Player(id=pid, name="Test", background="BG", token="tok1")

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                nar="Test scenario",
                img="prompt",
                pub=[rpg.PublicStatus(pid=pid, word="   ")],
                upd=[
                    rpg.PlayerUpdate(
                        pid=pid,
                        cls="Wizard",
                        ab=[rpg.Ability(n="Arcana", x="novice")],
                        inv=["staff"],
                        cond=["healthy"],
                    )
                ],
            )

        original = rpg.gemini_generate_json
        rpg.gemini_generate_json = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.gemini_generate_json = original

        player = rpg.STATE.players[pid]
        self.assertEqual(player.status_word, "unknown")
        self.assertEqual(player.cls, "Wizard")
        self.assertEqual(rpg.STATE.turn_index, 1)

    async def test_status_word_trimmed_and_lowercased(self):
        pid = "player2"
        rpg.STATE.players[pid] = rpg.Player(id=pid, name="Other", background="BG", token="tok2")

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                nar="Another scenario",
                img="prompt",
                pub=[rpg.PublicStatus(pid=pid, word="  MiGhTy Hero  ")],
                upd=[rpg.PlayerUpdate(pid=pid, cls="Bard", ab=[], inv=[], cond=[])],
            )

        original = rpg.gemini_generate_json
        rpg.gemini_generate_json = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.gemini_generate_json = original

        player = rpg.STATE.players[pid]
        self.assertEqual(player.status_word, "mighty")
        self.assertEqual(player.cls, "Bard")
        self.assertEqual(rpg.STATE.turn_index, 1)

    async def test_pending_leave_player_removed_after_turn(self):
        stay_id = "stay"
        leave_id = "leave"
        rpg.STATE.players[stay_id] = rpg.Player(id=stay_id, name="Stayed", background="BG", token="staytok", pending_join=False)
        rpg.STATE.players[leave_id] = rpg.Player(
            id=leave_id,
            name="Drifter",
            background="BG",
            token="leavetok",
            pending_join=False,
            pending_leave=True,
        )

        async def fake_generate_json(model, system_prompt, user_payload, schema):
            return rpg.TurnStructured(
                nar="The drifter says farewell and walks into the mist.",
                img="farewell",
                pub=[
                    rpg.PublicStatus(pid=stay_id, word="ready"),
                    rpg.PublicStatus(pid=leave_id, word="departed"),
                ],
                upd=[
                    rpg.PlayerUpdate(
                        pid=stay_id,
                        cls="Guardian",
                        ab=[rpg.Ability(n="Watch", x="novice")],
                        inv=["shield"],
                        cond=["steady"],
                    )
                ],
            )

        original = rpg.gemini_generate_json
        rpg.gemini_generate_json = fake_generate_json
        try:
            await rpg.resolve_turn(initial=False)
        finally:
            rpg.gemini_generate_json = original

        self.assertIn(stay_id, rpg.STATE.players)
        self.assertNotIn(leave_id, rpg.STATE.players)
        self.assertEqual(rpg.STATE.players[stay_id].cls, "Guardian")
        self.assertFalse(rpg.STATE.players[stay_id].pending_leave)
        self.assertEqual(rpg.STATE.turn_index, 1)


class CompileUserPayloadTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_payload_includes_history_and_serializes_abilities(self):
        rpg.STATE.turn_index = 3
        rpg.STATE.settings["world_style"] = "Solarpunk"
        rpg.STATE.settings["difficulty"] = "Hard"

        rpg.STATE.history.append(
            rpg.TurnRecord(index=0, narrative="Intro", image_prompt="orb", timestamp=123.0)
        )

        p1 = "p1"
        p2 = "p2"
        rpg.STATE.players[p1] = rpg.Player(
            id=p1,
            name="Alice",
            background="Scholar",
            cls="Mage",
            abilities=[rpg.Ability(n="Arcana", x="expert")],
            inventory=["orb"],
            conditions=["healed"],
            status_word="ready",
            pending_join=False,
            token="tokA",
        )
        rpg.STATE.players[p2] = rpg.Player(
            id=p2,
            name="Bryn",
            background="Scout",
            cls="",
            abilities=[{"n": "Stealth", "x": "novice"}],
            inventory=["cloak"],
            conditions=[],
            status_word="unknown",
            pending_join=True,
            token="tokB",
        )
        rpg.STATE.submissions[p1] = "Scout ahead"
        rpg.STATE.submissions[p2] = "Hold position"

        payload = rpg.compile_user_payload()

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

    def test_summary_history_mode_uses_bullets(self):
        rpg.STATE.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY
        rpg.STATE.turn_index = 2
        r1 = rpg.TurnRecord(index=0, narrative="The heroes arrived at the ruins.", image_prompt="ruins", timestamp=1.0)
        r2 = rpg.TurnRecord(index=1, narrative="They defeated the guardian and claimed a relic.", image_prompt="guardian", timestamp=2.0)
        rpg.STATE.history.extend([r1, r2])
        rpg.STATE.history_summary = ["- Turn 0: The heroes arrived at the ruins."]

        payload = rpg.compile_user_payload()

        self.assertEqual(payload["history_mode"], rpg.HISTORY_MODE_SUMMARY)
        self.assertEqual(payload["history"], rpg.STATE.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in payload["history"]))

        # When cached summary is empty, fallback summarization should kick in.
        rpg.STATE.history_summary.clear()
        payload = rpg.compile_user_payload()
        self.assertTrue(payload["history"])
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in payload["history"]))


class HistorySummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_update_summary_uses_fallback_without_api_key(self):
        rec = rpg.TurnRecord(index=0, narrative="A mysterious storm rolls in.", image_prompt="storm", timestamp=1.0)
        rpg.STATE.history.append(rec)

        await rpg.update_history_summary(rec)

        self.assertTrue(rpg.STATE.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.STATE.history_summary))

    async def test_update_summary_uses_model_output(self):
        rec = rpg.TurnRecord(index=1, narrative="The heroes rescue villagers from the cellar.", image_prompt="cellar", timestamp=2.0)
        rpg.STATE.history.append(rec)
        rpg.STATE.settings["api_key"] = "test-key"
        rpg.STATE.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        fake_summary = rpg.SummaryStructured(summary=["Rescued villagers from the cellar."])

        with mock.patch("rpg.gemini_generate_summary", new=mock.AsyncMock(return_value=fake_summary)):
            await rpg.update_history_summary(rec)

        self.assertEqual(rpg.STATE.history_summary, ["- Rescued villagers from the cellar."])

    async def test_update_summary_falls_back_on_error(self):
        rec = rpg.TurnRecord(index=2, narrative="A dragon attacks the camp.", image_prompt="dragon", timestamp=3.0)
        rpg.STATE.history.append(rec)
        rpg.STATE.settings["api_key"] = "test-key"
        rpg.STATE.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY

        failing = mock.AsyncMock(side_effect=RuntimeError("summary boom"))

        with mock.patch("rpg.gemini_generate_summary", new=failing):
            await rpg.update_history_summary(rec)

        self.assertTrue(rpg.STATE.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.STATE.history_summary))

    async def test_update_summary_skips_model_when_mode_full(self):
        rec = rpg.TurnRecord(index=3, narrative="Ancient wards flare and seal the tomb.", image_prompt="wards", timestamp=4.0)
        rpg.STATE.history.append(rec)
        rpg.STATE.settings["history_mode"] = rpg.HISTORY_MODE_FULL
        rpg.STATE.settings["api_key"] = "present"

        sentinel = mock.AsyncMock()

        with mock.patch("rpg.gemini_generate_summary", new=sentinel):
            await rpg.update_history_summary(rec)

        sentinel.assert_not_awaited()
        self.assertTrue(rpg.STATE.history_summary)
        self.assertTrue(all(line.startswith("- ") or line.startswith("-") for line in rpg.STATE.history_summary))


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
            result = await rpg.gemini_generate_json(
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
            result = await rpg.gemini_generate_summary(
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
            player = rpg.STATE.players[pid]
            player.pending_join = False
            rpg.STATE.turn_index += 1

        with mock.patch("secrets.token_hex", side_effect=[pid, issued_token]), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.resolve_turn", new=mock.AsyncMock(side_effect=fake_resolve)) as resolve_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.JoinBody(name="Alice", background="Ranger")
            result = await rpg.join_game(body)

        self.assertEqual(result["player_id"], pid)
        self.assertEqual(result["auth_token"], issued_token)
        self.assertIn(pid, rpg.STATE.players)
        self.assertEqual(rpg.STATE.turn_index, 1)
        self.assertFalse(rpg.STATE.players[pid].pending_join)
        self.assertEqual(rpg.STATE.players[pid].token, issued_token)
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

        self.assertNotIn(pid, rpg.STATE.players)
        announce_mock.assert_awaited_once()
        resolve_mock.assert_awaited_once_with(initial=True)

    async def test_subsequent_join_skips_initial_turn(self):
        rpg.STATE.turn_index = 1
        rpg.STATE.current_scenario = "World is underway"
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
        player = rpg.STATE.players[pid]
        self.assertEqual(player.name, long_name[:40])
        self.assertEqual(player.background, long_background[:200])
        self.assertTrue(player.pending_join)


class SubmitActionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.STATE.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-submit")

    async def test_known_player_submission_trimmed(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="p1", token="tok-submit", text="  Attack the gate   ")
            response = await rpg.submit_action(body)

        self.assertEqual(response, {"ok": True})
        self.assertEqual(rpg.STATE.submissions["p1"], "Attack the gate")

    async def test_unknown_player_submission_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="missing", token="tok-submit", text="observe")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(rpg.STATE.submissions, {})

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.SubmitBody(player_id="p1", token="wrong", text="observe")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(rpg.STATE.submissions, {})

    async def test_submission_rejected_when_busy(self):
        rpg.STATE.lock = rpg.LockState(True, "resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.SubmitBody(player_id="p1", token="tok-submit", text="charge")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.submit_action(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(rpg.STATE.submissions, {})
        broadcast.assert_not_awaited()


class CreateImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.STATE.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-image")
        rpg.STATE.last_image_prompt = "Ancient ruins shrouded in mist"

    async def test_unknown_player_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreateImageBody(player_id="ghost", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertFalse(rpg.STATE.lock.active)

    async def test_known_player_generates_image(self):
        fake_data_url = "data:image/png;base64,abc"
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()), \
                mock.patch("rpg.announce", new=mock.AsyncMock()), \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(return_value=fake_data_url)) as gen:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            result = await rpg.create_image(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.last_image_data_url, fake_data_url)
        self.assertFalse(rpg.STATE.lock.active)
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "scene")

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreateImageBody(player_id="p1", token="wrong")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertFalse(rpg.STATE.lock.active)

    async def test_requires_image_prompt(self):
        rpg.STATE.last_image_prompt = None
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertFalse(rpg.STATE.lock.active)
        broadcast.assert_not_awaited()

    async def test_conflict_when_busy(self):
        rpg.STATE.lock = rpg.LockState(True, "resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_image(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(rpg.STATE.lock.active)
        self.assertEqual(rpg.STATE.lock.reason, "resolving_turn")
        broadcast.assert_not_awaited()

    async def test_lock_released_on_generation_failure(self):
        fake_error = RuntimeError("generation failed")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(side_effect=fake_error)) as gen:
            body = rpg.CreateImageBody(player_id="p1", token="tok-image")
            with self.assertRaises(RuntimeError):
                await rpg.create_image(body)

        self.assertFalse(rpg.STATE.lock.active)
        self.assertEqual(rpg.STATE.lock.reason, "")
        self.assertIsNone(rpg.STATE.last_image_data_url)
        self.assertEqual(broadcast.await_count, 2)
        announce.assert_not_awaited()
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "scene")


class PublicSnapshotPrivacyTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        player = rpg.Player(id="p1", name="Alice", background="Mage", token="tok")
        rpg.STATE.players[player.id] = player
        rpg.STATE.submissions[player.id] = "Explore the ruins"

    def test_public_snapshot_includes_id_without_token(self):
        snap = rpg.STATE.public_snapshot()
        self.assertTrue(snap["players"])
        self.assertIn("id", snap["players"][0])
        self.assertNotIn("token", snap["players"][0])

    def test_submission_ids_not_exposed(self):
        snap = rpg.STATE.public_snapshot()
        self.assertTrue(snap["submissions"])
        self.assertNotIn("player_id", snap["submissions"][0])

    def test_public_snapshot_includes_language(self):
        snap = rpg.STATE.public_snapshot()

        self.assertEqual(snap["language"], rpg.DEFAULT_LANGUAGE)

    def test_public_snapshot_exposes_class_name(self):
        rpg.STATE.players["p1"].cls = "Battle Mage"

        snap = rpg.STATE.public_snapshot()

        self.assertIn("cls", snap["players"][0])
        self.assertEqual(snap["players"][0]["cls"], "Battle Mage")

    def test_public_snapshot_includes_portrait_payload(self):
        rpg.STATE.players["p1"].portrait = rpg.PlayerPortrait(
            data_url="data:image/png;base64,portrait",
            prompt="prompt text",
            updated_at=123.45,
        )

        snap = rpg.STATE.public_snapshot()
        portrait = snap["players"][0].get("portrait")

        self.assertIsInstance(portrait, dict)
        self.assertEqual(portrait["data_url"], "data:image/png;base64,portrait")
        self.assertEqual(portrait["prompt"], "prompt text")

    def test_snapshot_includes_image_and_token_stats(self):
        rpg.STATE.last_image_data_url = "data:image/png;base64,xyz"
        rpg.STATE.last_image_prompt = "An ancient gate"
        rpg.STATE.turn_index = 5
        rpg.STATE.last_token_usage = {"input": 20, "output": 30, "thinking": 10}
        rpg.STATE.last_turn_runtime = 4
        rpg.STATE.last_cost_usd = 1.23
        rpg.STATE.session_cost_usd = 4.56
        rpg.STATE.last_image_model = "gemini-2.5-flash-image-preview"
        rpg.STATE.last_image_kind = "scene"
        rpg.STATE.last_image_cost_usd = 0.039
        rpg.STATE.last_image_usd_per = 0.039
        rpg.STATE.last_image_tokens = 1290
        rpg.STATE.last_image_count = 1
        rpg.STATE.last_image_turn_index = 5
        rpg.STATE.last_scene_image_cost_usd = 0.039
        rpg.STATE.last_scene_image_usd_per = 0.039
        rpg.STATE.last_scene_image_model = "gemini-2.5-flash-image-preview"
        rpg.STATE.last_scene_image_turn_index = 5
        rpg.STATE.session_image_cost_usd = 0.078
        rpg.STATE.session_image_requests = 2
        rpg.STATE.session_image_kind_counts = {"scene": 2}
        rpg.STATE.turn_image_kind_counts = {"scene": 1, "portrait": 1}

        snap = rpg.STATE.public_snapshot()

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
        self.assertAlmostEqual(totals["last_usd"], 1.269, places=3)
        self.assertAlmostEqual(totals["session_usd"], 4.638, places=3)
        self.assertEqual(totals["breakdown"]["image_session_usd"], 0.078)
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
        rpg.STATE.settings["api_key"] = "sk-alpha123456"

        result = await rpg.get_settings()

        self.assertEqual(result["api_key"], "sk-alpha123456")
        self.assertEqual(rpg.STATE.settings["api_key"], "sk-alpha123456")

    async def test_update_settings_trims_and_persists(self):
        body = rpg.SettingsUpdate(api_key="  sk-newkey  ", world_style="Noir", thinking_mode="Deep")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.settings["api_key"], "sk-newkey")
        self.assertEqual(rpg.STATE.settings["world_style"], "Noir")
        self.assertEqual(rpg.STATE.settings["thinking_mode"], "deep")
        save_mock.assert_awaited_once_with(rpg.STATE.settings)

    async def test_update_settings_sets_narration_model(self):
        body = rpg.SettingsUpdate(narration_model="eleven_flash_v2_1")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.settings["narration_model"], "eleven_flash_v2_1")
        save_mock.assert_awaited_once_with(rpg.STATE.settings)

    async def test_update_settings_sets_openai_key(self):
        body = rpg.SettingsUpdate(openai_api_key="sk-openai-test")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.settings["openai_api_key"], "sk-openai-test")
        save_mock.assert_awaited_once_with(rpg.STATE.settings)

    async def test_update_settings_ignores_blank_narration_model(self):
        rpg.STATE.settings["narration_model"] = "eleven_flash_v2_5"
        body = rpg.SettingsUpdate(narration_model="   ")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.settings["narration_model"], "eleven_flash_v2_5")
        save_mock.assert_not_awaited()

    async def test_update_settings_normalizes_history_mode(self):
        rpg.STATE.settings["history_mode"] = rpg.HISTORY_MODE_SUMMARY
        body = rpg.SettingsUpdate(history_mode=" timeline ")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            result = await rpg.update_settings(body)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(rpg.STATE.settings["history_mode"], rpg.HISTORY_MODE_FULL)
        save_mock.assert_awaited_once_with(rpg.STATE.settings)

    async def test_update_settings_rejects_unknown_thinking_mode(self):
        body = rpg.SettingsUpdate(thinking_mode="cosmic")
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save_mock:
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.update_settings(body)

        self.assertEqual(ctx.exception.status_code, 400)
        save_mock.assert_not_awaited()

    async def test_get_settings_returns_blank_api_key_when_missing(self):
        rpg.STATE.settings["api_key"] = ""

        result = await rpg.get_settings()

        self.assertEqual(result["api_key"], "")
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
                    "supportedGenerationMethods": ["generateContent"],
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

        rpg.STATE.settings["elevenlabs_api_key"] = "mock-eleven-key"
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
                    "id": "grok-4",
                    "display_name": "Grok 4",
                    "capabilities": ["chat.completions"],
                }
            ]

        reset_state()
        rpg.STATE.settings["api_key"] = ""
        rpg.STATE.settings["grok_api_key"] = "grok-secret"

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
                    "id": "gpt-4o-mini",
                    "display_name": "GPT-4o Mini",
                    "capabilities": {
                        "responses": True,
                        "chat_completions": True,
                    },
                }
            ]

        reset_state()
        rpg.STATE.settings["api_key"] = ""
        rpg.STATE.settings["openai_api_key"] = "openai-secret"

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
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-1.5-flash",
                    "displayName": "Gemini 1.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]

        async def fake_grok_list_models(api_key):
            self.assertEqual(api_key, "grok-key")
            return [
                {"id": "grok-4-fast", "display_name": "Grok 4 Fast", "capabilities": ["chat.completions"]},
                {"id": "grok-4", "display_name": "Grok 4", "capabilities": ["chat.completions"]},
            ]

        reset_state()
        rpg.STATE.settings["api_key"] = "gemini-key"
        rpg.STATE.settings["grok_api_key"] = "grok-key"

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


class ApplyLanguageValueTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_apply_language_value_updates_state_on_change(self):
        changed = rpg.apply_language_value("German (Standard)")

        self.assertTrue(changed)
        self.assertEqual(rpg.STATE.language, "de")
        self.assertEqual(rpg.STATE.settings["language"], "de")

    def test_apply_language_value_returns_false_for_none(self):
        rpg.STATE.language = "en"
        rpg.STATE.settings["language"] = "en"

        changed = rpg.apply_language_value(None)

        self.assertFalse(changed)
        self.assertEqual(rpg.STATE.language, "en")
        self.assertEqual(rpg.STATE.settings["language"], "en")

    def test_apply_language_value_returns_false_when_language_unchanged(self):
        rpg.STATE.language = "de"
        rpg.STATE.settings["language"] = "de"

        changed = rpg.apply_language_value("de-DE")

        self.assertFalse(changed)
        self.assertEqual(rpg.STATE.language, "de")
        self.assertEqual(rpg.STATE.settings["language"], "de")


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
        result = rpg.calculate_turn_cost("grok-4", 1_000_000, 2_000_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 3.0)
        self.assertAlmostEqual(result["completion_usd"], 30.0)
        self.assertAlmostEqual(result["total_usd"], 33.0)

    def test_calculate_turn_cost_grok_code_fast(self):
        result = rpg.calculate_turn_cost("grok-code-fast-1", 500_000, 250_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.1)
        self.assertAlmostEqual(result["completion_usd"], 0.375)
        self.assertAlmostEqual(result["total_usd"], 0.475)

    def test_calculate_turn_cost_grok_fast_reasoning(self):
        result = rpg.calculate_turn_cost("grok-4-fast-reasoning", 1_000_000, 1_000_000)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["prompt_usd"], 0.2)
        self.assertAlmostEqual(result["completion_usd"], 0.5)
        self.assertAlmostEqual(result["total_usd"], 0.7)


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
        rpg.STATE.turn_index = 4

        rpg.record_image_usage(
            "gemini-2.5-flash-image-preview",
            purpose="scene",
            tier="standard",
            images=2,
        )

        self.assertEqual(rpg.STATE.last_image_model, "gemini-2.5-flash-image-preview")
        self.assertEqual(rpg.STATE.last_image_kind, "scene")
        self.assertEqual(rpg.STATE.last_image_count, 2)
        self.assertEqual(rpg.STATE.last_image_turn_index, 4)
        self.assertEqual(rpg.STATE.last_image_tier, "standard")
        self.assertAlmostEqual(rpg.STATE.last_image_cost_usd, 0.078, places=3)
        self.assertAlmostEqual(rpg.STATE.last_image_usd_per, 0.039, places=3)
        self.assertEqual(rpg.STATE.last_image_tokens, 1290)
        self.assertAlmostEqual(rpg.STATE.session_image_cost_usd, 0.078, places=3)
        self.assertEqual(rpg.STATE.session_image_requests, 2)
        self.assertEqual(rpg.STATE.session_image_kind_counts["scene"], 2)
        self.assertEqual(rpg.STATE.turn_image_kind_counts["scene"], 2)
        self.assertEqual(rpg.STATE.last_scene_image_model, "gemini-2.5-flash-image-preview")
        self.assertAlmostEqual(rpg.STATE.last_scene_image_cost_usd, 0.078, places=3)
        self.assertAlmostEqual(rpg.STATE.last_scene_image_usd_per, 0.039, places=3)
        self.assertEqual(rpg.STATE.last_scene_image_turn_index, 4)

    def test_record_image_usage_handles_missing_pricing(self):
        rpg.STATE.turn_index = 1

        rpg.record_image_usage(
            "custom/unknown-model",
            purpose="sketch",
            tier="premium",
            images=0,
        )

        self.assertEqual(rpg.STATE.last_image_model, "unknown-model")
        self.assertEqual(rpg.STATE.last_image_kind, "sketch")
        self.assertEqual(rpg.STATE.last_image_count, 0)
        self.assertEqual(rpg.STATE.last_image_turn_index, 1)
        self.assertEqual(rpg.STATE.last_image_tier, "premium")
        self.assertIsNone(rpg.STATE.last_image_cost_usd)
        self.assertIsNone(rpg.STATE.last_image_usd_per)
        self.assertIsNone(rpg.STATE.last_image_tokens)
        self.assertEqual(rpg.STATE.session_image_cost_usd, 0.0)
        self.assertEqual(rpg.STATE.session_image_requests, 0)
        self.assertNotIn("sketch", rpg.STATE.session_image_kind_counts)
        self.assertEqual(rpg.STATE.turn_image_kind_counts["other"], 0)
        self.assertIsNone(rpg.STATE.last_scene_image_model)
        self.assertIsNone(rpg.STATE.last_scene_image_cost_usd)
        self.assertIsNone(rpg.STATE.last_scene_image_turn_index)

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
        custom = {"world_style": "Noir", "api_key": "sk-123"}
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
        existing["api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({"world_style": "Noir"})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["api_key"], "sk-existing")
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
        rpg.STATE.settings = existing

        await rpg.save_settings(rpg.STATE.settings)

        self.assertIs(rpg.STATE.settings, existing)
        self.assertEqual(rpg.STATE.settings["language"], "de")
        self.assertEqual(rpg.STATE.settings["history_mode"], rpg.HISTORY_MODE_SUMMARY)

    async def test_save_settings_discards_transient_keys(self):
        await rpg.save_settings({
            "world_style": "Gothic",
            "api_key_preview": "partial",
            "api_key_set": True,
            "elevenlabs_api_key_preview": "voice-partial",
            "elevenlabs_api_key_set": True,
        })

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["world_style"], "Gothic")
        self.assertNotIn("api_key_preview", on_disk)
        self.assertNotIn("api_key_set", on_disk)
        self.assertNotIn("elevenlabs_api_key_preview", on_disk)
        self.assertNotIn("elevenlabs_api_key_set", on_disk)

    async def test_save_settings_ignores_blank_api_key(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({"api_key": "   "})

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["api_key"], "sk-existing")
        self.assertEqual(on_disk["elevenlabs_api_key"], "voice-existing")

    async def test_save_settings_strips_whitespace_api_keys(self):
        existing = rpg.DEFAULT_SETTINGS.copy()
        existing["api_key"] = "sk-existing"
        existing["elevenlabs_api_key"] = "voice-existing"
        rpg.SETTINGS_FILE.write_text(json.dumps(existing))

        await rpg.save_settings({
            "api_key": "  sk-new  ",
            "elevenlabs_api_key": "\tvoice-new\n",
        })

        on_disk = json.loads(rpg.SETTINGS_FILE.read_text())
        self.assertEqual(on_disk["api_key"], "sk-new")
        self.assertEqual(on_disk["elevenlabs_api_key"], "voice-new")


class ResolveTurnFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        self.pid = "p1"
        rpg.STATE.players[self.pid] = rpg.Player(
            id=self.pid,
            name="Alice",
            background="Cleric",
            token="tok-turn",
            pending_join=True,
        )
        rpg.STATE.submissions[self.pid] = "Provide healing"

    async def test_pending_join_triggers_announcement_and_clears_state(self):
        structured = rpg.TurnStructured(
            nar="The caverns tremble",
            img="A holy light pierces darkness",
            pub=[rpg.PublicStatus(pid=self.pid, word="Ready for battle")],
            upd=[
                rpg.PlayerUpdate(
                    pid=self.pid,
                    cls="Cleric",
                    ab=[rpg.Ability(n="Radiance", x="novice")],
                    inv=["mace"],
                    cond=["inspired"],
                )
            ],
        )

        with mock.patch("rpg.gemini_generate_json", new=mock.AsyncMock(return_value=structured)), \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce_mock, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private_mock:
            await rpg.resolve_turn(initial=False)

        announce_mock.assert_awaited_once()
        self.assertIn("joined the party", announce_mock.await_args.args[0])
        self.assertEqual(rpg.STATE.turn_index, 1)
        self.assertEqual(rpg.STATE.current_scenario, "The caverns tremble")
        self.assertEqual(rpg.STATE.last_image_prompt, "A holy light pierces darkness")
        self.assertFalse(rpg.STATE.players[self.pid].pending_join)
        self.assertEqual(rpg.STATE.players[self.pid].cls, "Cleric")
        self.assertEqual(rpg.STATE.players[self.pid].status_word, "ready")
        self.assertEqual(rpg.STATE.submissions, {})
        self.assertEqual(len(rpg.STATE.history), 1)
        self.assertFalse(rpg.STATE.lock.active)
        broadcast_mock.assert_awaited()
        send_private_mock.assert_awaited()

    async def test_failure_releases_lock_and_preserves_turn_index(self):
        async def failing_call(*_args, **_kwargs):
            raise RuntimeError("generation exploded")

        with mock.patch("rpg.gemini_generate_json", new=mock.AsyncMock(side_effect=failing_call)), \
                mock.patch("rpg.announce", new=mock.AsyncMock()), \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast_mock:
            with self.assertRaises(RuntimeError):
                await rpg.resolve_turn(initial=False)

        self.assertEqual(rpg.STATE.turn_index, 0)
        self.assertGreaterEqual(broadcast_mock.await_count, 2)
        self.assertFalse(rpg.STATE.lock.active)


class CreatePortraitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.STATE.players["p1"] = rpg.Player(
            id="p1",
            name="Alice",
            background="Mage",
            cls="Wizard",
            status_word="resolute",
            token="tok-portrait",
        )

    async def test_unknown_player_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreatePortraitBody(player_id="ghost", token="tok-portrait")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIsNone(rpg.STATE.players["p1"].portrait)

    async def test_invalid_token_rejected(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()):
            body = rpg.CreatePortraitBody(player_id="p1", token="wrong")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIsNone(rpg.STATE.players["p1"].portrait)

    async def test_conflict_when_busy(self):
        rpg.STATE.lock = rpg.LockState(True, "resolving_turn")
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            with self.assertRaises(rpg.HTTPException) as ctx:
                await rpg.create_portrait(body)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(rpg.STATE.lock.active)
        self.assertEqual(rpg.STATE.lock.reason, "resolving_turn")
        broadcast.assert_not_awaited()
        self.assertIsNone(rpg.STATE.players["p1"].portrait)

    async def test_generates_portrait_and_announces(self):
        fake_data_url = "data:image/png;base64,portrait"
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(return_value=fake_data_url)) as gen:
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            result = await rpg.create_portrait(body)

        self.assertEqual(result["ok"], True)
        player = rpg.STATE.players["p1"]
        self.assertIsNotNone(player.portrait)
        self.assertEqual(player.portrait.data_url, fake_data_url)
        self.assertIn("portrait", player.portrait.prompt.lower())
        self.assertFalse(rpg.STATE.lock.active)
        announce.assert_awaited()
        send_private.assert_awaited_once_with("p1")
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "portrait")
        self.assertGreaterEqual(broadcast.await_count, 2)

    async def test_lock_released_on_generation_failure(self):
        with mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast, \
                mock.patch("rpg.announce", new=mock.AsyncMock()) as announce, \
                mock.patch("rpg.send_private", new=mock.AsyncMock()) as send_private, \
                mock.patch("rpg.gemini_generate_image", new=mock.AsyncMock(side_effect=RuntimeError("fail"))) as gen:
            body = rpg.CreatePortraitBody(player_id="p1", token="tok-portrait")
            with self.assertRaises(RuntimeError):
                await rpg.create_portrait(body)

        self.assertFalse(rpg.STATE.lock.active)
        self.assertEqual(rpg.STATE.lock.reason, "")
        self.assertIsNone(rpg.STATE.players["p1"].portrait)
        self.assertGreaterEqual(broadcast.await_count, 2)
        announce.assert_not_awaited()
        send_private.assert_not_awaited()
        gen.assert_awaited_once()
        self.assertEqual(gen.await_args.kwargs.get("purpose"), "portrait")

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
        rpg.STATE.settings["thinking_mode"] = "none"

        text = rpg.build_thinking_directive()

        self.assertIn("minimum internal deliberation", text)

    def test_build_thinking_directive_invalid_mode_defaults(self):
        rpg.STATE.settings["thinking_mode"] = "galactic"

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
        rpg.STATE.settings["thinking_mode"] = "brief"

        text = rpg.make_gm_instruction(is_initial=True)

        self.assertTrue(text.startswith("Prelude\n"))
        self.assertIn("INITIAL TURN", text)
        self.assertIn("Take a short internal moment", text)
        self.assertTrue(text.endswith("Epilogue"))

    def test_instruction_appends_when_token_missing(self):
        rpg.GM_PROMPT_TEMPLATE = "Core guidance"
        rpg.STATE.settings["thinking_mode"] = "deep"

        text = rpg.make_gm_instruction(is_initial=False)

        self.assertTrue(text.startswith("Core guidance\n"))
        self.assertIn("ONGOING TURN", text)
        self.assertIn("thorough internal reasoning", text)


class LanguageEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        player = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-lang")
        rpg.STATE.players[player.id] = player

    async def test_set_language_updates_state(self):
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            result = await rpg.set_language(rpg.LanguageBody(language="de"))

        self.assertEqual(result, {"language": "de"})
        self.assertEqual(rpg.STATE.language, "de")
        self.assertEqual(rpg.STATE.settings["language"], "de")
        save.assert_awaited_once()
        broadcast.assert_awaited_once()

    async def test_set_language_normalizes_invalid_value(self):
        with mock.patch("rpg.save_settings", new=mock.AsyncMock()) as save, \
                mock.patch("rpg.broadcast_public", new=mock.AsyncMock()) as broadcast:
            result = await rpg.set_language(rpg.LanguageBody(language="fr"))

        self.assertEqual(result, {"language": "en"})
        self.assertEqual(rpg.STATE.language, "en")
        save.assert_not_awaited()
        broadcast.assert_not_awaited()

    async def test_set_language_authenticates_when_credentials_provided(self):
        with mock.patch("rpg.authenticate_player", wraps=rpg.authenticate_player) as auth:
            await rpg.set_language(rpg.LanguageBody(language="de", player_id="p1", token="tok-lang"))

        auth.assert_called_once_with("p1", "tok-lang")


class NextTurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.STATE.players["p1"] = rpg.Player(id="p1", name="Alice", background="Mage", token="tok-turn")

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
        ability = rpg.Ability(n="Arcana", x="expert")
        player = rpg.Player(
            id="p1",
            name="Alice",
            background="Scholar",
            cls="Mage",
            abilities=[ability],
            pending_join=False,
        )
        state = rpg.GameState(players={"p1": player})
        snapshot = state.private_snapshot_for("p1")

        self.assertEqual(snapshot["you"]["id"], "p1")
        self.assertEqual(snapshot["you"]["class"], "Mage")
        self.assertEqual(snapshot["you"]["abilities"], [ability.model_dump()])
        self.assertFalse(snapshot["you"]["pending_join"])


class AuthenticatePlayerTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.player = rpg.Player(id="p1", name="Alice", background="Mage", token="secret")
        rpg.STATE.players[self.player.id] = self.player

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


class CheckApiKeyTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_check_api_key_returns_value(self):
        rpg.STATE.settings["api_key"] = "sk-valid"

        result = rpg.check_api_key()

        self.assertEqual(result, "sk-valid")

    def test_check_api_key_raises_when_missing(self):
        rpg.STATE.settings["api_key"] = ""

        with self.assertRaises(rpg.HTTPException) as ctx:
            rpg.check_api_key()

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
        self.assertEqual(rpg.PUBLIC_IP_CACHE["timestamp"], 100.0)

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
        self.assertEqual(rpg.PUBLIC_IP_CACHE["failure_ts"], 200.0)

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
        rpg.STATE.settings["api_key"] = "sk-test"

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
            result = await rpg.gemini_generate_json(
                model="gemini-2.5-flash-lite",
                system_prompt="sys",
                user_payload={"foo": "bar"},
                schema=schema,
            )

        self.assertIsInstance(result, rpg.TurnStructured)
        self.assertEqual(result.nar, "Narrative")
        self.assertEqual(rpg.STATE.last_token_usage["input"], 12)
        self.assertEqual(rpg.STATE.last_token_usage["thinking"], 4)
        self.assertIsNotNone(rpg.STATE.last_turn_runtime)

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
                await rpg.gemini_generate_json(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    user_payload={},
                    schema=rpg.build_turn_schema(),
                )

        self.assertEqual(ctx.exception.status_code, 502)


class GeminiGenerateImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()
        rpg.STATE.settings["api_key"] = "sk-test"

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

        with mock.patch("httpx.AsyncClient", DummyClient):
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
        self.assertEqual(rpg.STATE.last_image_model, "gemini-2.5-flash-image-preview")
        self.assertEqual(rpg.STATE.session_image_requests, 1)
        self.assertEqual(rpg.STATE.session_image_kind_counts, {"scene": 1})
        self.assertAlmostEqual(rpg.STATE.last_image_cost_usd, 0.039, places=3)
        self.assertAlmostEqual(rpg.STATE.session_image_cost_usd, 0.039, places=3)
        self.assertAlmostEqual(rpg.STATE.last_scene_image_cost_usd, 0.039, places=3)
        self.assertEqual(rpg.STATE.turn_image_kind_counts, {"scene": 1})

    async def test_generate_image_includes_portrait_references(self):
        player = rpg.Player(id="p1", name="Tess", background="Scout", cls="Ranger", token="tok")
        player.portrait = rpg.PlayerPortrait(
            data_url="data:image/png;base64,ZmFrZQ==",
            prompt="Portrait prompt",
            updated_at=123.0,
        )
        rpg.STATE.players[player.id] = player

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
                "dramatic scene on a bridge",
                purpose="scene",
            )

        request_body = DummyClient.last_json
        self.assertIsNotNone(request_body)
        parts = request_body["contents"][0]["parts"]
        self.assertGreaterEqual(len(parts), 2)
        first_part = parts[0]
        self.assertIn("inlineData", first_part)
        self.assertEqual(first_part["inlineData"]["mimeType"], "image/png")
        self.assertEqual(first_part["inlineData"]["data"], "ZmFrZQ==")
        directive_text = parts[-1]["text"]
        self.assertIn("Use the provided player portraits", directive_text)
        self.assertIn("Scene prompt: dramatic scene on a bridge", directive_text)
        self.assertIn("Tess", directive_text)

    async def test_generate_image_skips_invalid_portrait_data(self):
        player = rpg.Player(id="p1", name="Iris", background="Mystic", cls="Mage", token="tok")
        player.portrait = rpg.PlayerPortrait(
            data_url="data:image/png;base64",
            prompt="Portrait prompt",
            updated_at=456.0,
        )
        rpg.STATE.players[player.id] = player

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
        self.assertEqual(rpg.STATE.session_image_requests, 0)
        self.assertEqual(rpg.STATE.session_image_cost_usd, 0.0)
        self.assertEqual(rpg.STATE.turn_image_kind_counts, {})


class SessionResetHelperTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_reset_clears_session_progress_preserving_settings(self):
        rpg.STATE.settings["world_style"] = "Solarpunk"
        rpg.STATE.turn_index = 7
        rpg.STATE.current_scenario = "Ancient ruins loom"
        rpg.STATE.session_token_usage = {"input": 10, "output": 5, "thinking": 1}
        rpg.STATE.last_image_prompt = "old prompt"
        pid = "departing"
        rpg.STATE.players[pid] = rpg.Player(
            id=pid,
            name="Drifter",
            background="Nomad",
            token="tok-depart",
            connected=False,
            pending_join=False,
            pending_leave=True,
        )

        did_reset = rpg.maybe_reset_session_if_empty()

        self.assertTrue(did_reset)
        self.assertEqual(rpg.STATE.settings["world_style"], "Solarpunk")
        self.assertEqual(rpg.STATE.players, {})
        self.assertEqual(rpg.STATE.turn_index, 0)
        self.assertEqual(rpg.STATE.current_scenario, "")
        self.assertEqual(rpg.STATE.session_token_usage, {"input": 0, "output": 0, "thinking": 0})
        self.assertIsNone(rpg.STATE.last_image_prompt)

    def test_reset_skipped_when_active_player_remains(self):
        pid = "active"
        rpg.STATE.players[pid] = rpg.Player(
            id=pid,
            name="Guardian",
            background="Knight",
            token="tok-active",
            connected=True,
            pending_join=False,
            pending_leave=False,
        )
        rpg.STATE.turn_index = 3

        did_reset = rpg.maybe_reset_session_if_empty()

        self.assertFalse(did_reset)
        self.assertIn(pid, rpg.STATE.players)
        self.assertEqual(rpg.STATE.turn_index, 3)


class SessionResetJoinTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state()

    async def test_join_clears_stale_players_and_runs_initial_resolve(self):
        rpg.STATE.settings["world_style"] = "Clockwork"
        old_pid = "old"
        rpg.STATE.players[old_pid] = rpg.Player(
            id=old_pid,
            name="Elder",
            background="Scholar",
            token="tok-old",
            connected=False,
            pending_join=False,
            pending_leave=True,
        )
        rpg.STATE.turn_index = 4
        rpg.STATE.current_scenario = "The tale lingers"

        async def fake_resolve_turn(initial: bool = False):
            self.assertTrue(initial)
            for player in rpg.STATE.players.values():
                player.pending_join = False
            for pid in list(rpg.STATE.players.keys()):
                if rpg.STATE.players[pid].pending_leave:
                    rpg.STATE.players.pop(pid, None)
            rpg.STATE.current_scenario = "Fresh chapter"
            rpg.STATE.turn_index = 1

        original_resolve = rpg.resolve_turn
        rpg.resolve_turn = fake_resolve_turn
        try:
            result = await rpg.join_game(rpg.JoinBody(name="Nova", background="Ranger"))
        finally:
            rpg.resolve_turn = original_resolve

        new_pid = result["player_id"]
        self.assertNotEqual(new_pid, old_pid)
        self.assertTrue(result["auth_token"])
        self.assertEqual(rpg.STATE.settings["world_style"], "Clockwork")
        self.assertEqual(len(rpg.STATE.players), 1)
        self.assertIn(new_pid, rpg.STATE.players)
        self.assertFalse(rpg.STATE.players[new_pid].pending_join)
        self.assertEqual(rpg.STATE.current_scenario, "Fresh chapter")
        self.assertNotIn(old_pid, rpg.STATE.players)

if __name__ == "__main__":
    unittest.main()
