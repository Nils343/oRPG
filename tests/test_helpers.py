import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import rpg


class NormalizeSupportedListTests(unittest.TestCase):
    def test_filters_truthy_dict_entries(self) -> None:
        raw = {"Alpha": True, "Beta": False, "Gamma": "yes"}
        result = rpg._normalize_supported_list(raw)
        self.assertEqual(result, ["Alpha", "Gamma"])

    def test_deduplicates_case_insensitively(self) -> None:
        raw = ["Model", "model", "Other", None]
        result = rpg._normalize_supported_list(raw)
        self.assertEqual(result, ["Model", "Other"])

    def test_uses_fallback_when_empty(self) -> None:
        fallback = ("one", "two")
        result = rpg._normalize_supported_list(None, fallback=fallback)
        self.assertEqual(result, ["one", "two"])

    def test_accepts_scalars(self) -> None:
        result = rpg._normalize_supported_list("Solo")
        self.assertEqual(result, ["Solo"])


class SuffixFromMimeTests(unittest.TestCase):
    def test_known_mime_mapping(self) -> None:
        self.assertEqual(rpg._suffix_from_mime("image/png"), ".png")

    def test_handles_parameters(self) -> None:
        self.assertEqual(rpg._suffix_from_mime("image/jpeg; charset=utf-8"), ".jpeg")

    def test_unknown_or_empty_defaults_to_bin(self) -> None:
        self.assertEqual(rpg._suffix_from_mime(None), ".bin")
        self.assertEqual(rpg._suffix_from_mime(""), ".bin")

    def test_sanitizes_custom_subtypes(self) -> None:
        self.assertEqual(
            rpg._suffix_from_mime("application/x-custom+json "),
            ".x-custom+json",
        )

    def test_handles_schema_without_separator(self) -> None:
        self.assertEqual(rpg._suffix_from_mime("weird"), ".bin")


class LoadModelPricingTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_path = Path(tmp_dir) / "missing.json"
            with mock.patch.object(rpg, "PRICING_FILE", fake_path):
                self.assertEqual(rpg._load_model_pricing(), {})

    def test_invalid_json_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_path = Path(tmp_dir) / "invalid.json"
            fake_path.write_text("{not_json", encoding="utf-8")
            with mock.patch.object(rpg, "PRICING_FILE", fake_path):
                self.assertEqual(rpg._load_model_pricing(), {})

    def test_valid_json_returns_mapping(self) -> None:
        payload = {"text": {"models": {"alpha": {"price": 1.23}}}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_path = Path(tmp_dir) / "pricing.json"
            fake_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(rpg, "PRICING_FILE", fake_path):
                self.assertEqual(rpg._load_model_pricing(), payload)


class ParseDataUrlTests(unittest.TestCase):
    def test_valid_data_url_returns_mime_and_data(self) -> None:
        mime, data = rpg._parse_data_url("data:image/png;base64,QUJD")
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, "QUJD")

    def test_whitespace_and_default_mime(self) -> None:
        mime_data = "  data:text/plain;base64, IS0t\n"
        mime, data = rpg._parse_data_url(mime_data)
        self.assertEqual(mime, "text/plain")
        self.assertEqual(data, "IS0t")

    def test_invalid_inputs_return_none(self) -> None:
        self.assertIsNone(rpg._parse_data_url("not a data url"))
        self.assertIsNone(rpg._parse_data_url("data:image/png;base64,"))


class DecodeBase64DataTests(unittest.TestCase):
    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(rpg._decode_base64_data(""))

    def test_handles_missing_padding(self) -> None:
        trimmed = "QUJD"[:-1]  # intentionally broken padding
        result = rpg._decode_base64_data(trimmed)
        self.assertEqual(result, b"AB")

    def test_invalid_input_returns_none(self) -> None:
        self.assertEqual(rpg._decode_base64_data("!!!"), b"")


class EnvFloatTests(unittest.TestCase):
    def test_returns_default_when_missing_or_invalid(self) -> None:
        with mock.patch.dict(os.environ, {"ORPG_VALUE": "not-a-float"}, clear=False):
            self.assertEqual(rpg._env_float("ORPG_VALUE", default=3.2), 3.2)
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual(rpg._env_float("ORPG_VALUE", default=1.5), 1.5)

    def test_parses_float_from_environment(self) -> None:
        with mock.patch.dict(os.environ, {"ORPG_VALUE": "2.75"}, clear=False):
            self.assertAlmostEqual(rpg._env_float("ORPG_VALUE", default=0.0), 2.75)


class SummaryBulletTests(unittest.TestCase):
    def test_normalizes_empty_and_existing_bullets(self) -> None:
        self.assertEqual(rpg._ensure_summary_bullet(""), "- No events recorded.")
        self.assertEqual(rpg._ensure_summary_bullet("- Already bullet"), "- Already bullet")
        self.assertEqual(rpg._ensure_summary_bullet("* star bullet"), "- star bullet")
        long_text = "A" * (rpg.MAX_HISTORY_SUMMARY_CHARS + 10)
        result = rpg._ensure_summary_bullet(long_text)
        self.assertTrue(result.endswith("..."))
        self.assertTrue(result.startswith("- "))


class FetchPublicIpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        rpg.PUBLIC_IP_CACHE.clear()
        rpg.PUBLIC_IP_CACHE.update({"ip": None, "cached_at": 0.0, "last_failure_at": 0.0})

    async def asyncTearDown(self) -> None:
        rpg.PUBLIC_IP_CACHE.clear()
        rpg.PUBLIC_IP_CACHE.update({"ip": None, "cached_at": 0.0, "last_failure_at": 0.0})

    async def test_skips_requests_after_recent_failure(self) -> None:
        rpg.PUBLIC_IP_CACHE["last_failure_at"] = time.time()
        with mock.patch("rpg.httpx.AsyncClient") as client_mock:
            result = await rpg.fetch_public_ip()
        self.assertIsNone(result)
        client_mock.assert_not_called()

    async def test_prefers_ipv6_and_caches_result(self) -> None:
        ipv6 = "2001:db8::1"
        ipv4 = "203.0.113.1"

        class DummyResponse:
            def __init__(self, payload: str) -> None:
                self.text = payload

            def raise_for_status(self) -> None:
                return None

        async def fake_get(url):
            payload = ipv6 if "api64" in url else ipv4
            return DummyResponse(payload)

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                return await fake_get(url)

        with mock.patch("rpg.httpx.AsyncClient", return_value=FakeClient()):
            result = await rpg.fetch_public_ip()

        self.assertEqual(result, ipv6)
        self.assertEqual(rpg.PUBLIC_IP_CACHE["ip"], ipv6)
        self.assertGreater(rpg.PUBLIC_IP_CACHE["cached_at"], 0.0)


class RecordImageUsageTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_rpg import reset_state

        reset_state()

    def tearDown(self) -> None:
        from tests.test_rpg import reset_state

        reset_state()

    def test_handles_unknown_pricing_data(self) -> None:
        with mock.patch("rpg.calculate_image_cost", return_value=None):
            rpg.record_image_usage("mystery", purpose="scene", images=0, turn_index=4)
        self.assertIsNone(rpg.game_state.last_image_cost_usd)
        self.assertIsNone(rpg.game_state.last_image_usd_per_image)
        self.assertEqual(rpg.game_state.last_scene_image_turn_index, 4)
        self.assertEqual(rpg.game_state.session_image_requests, 0)
        self.assertIn("scene", rpg._bind_turn_image_bucket(4))


if __name__ == "__main__":
    unittest.main()
