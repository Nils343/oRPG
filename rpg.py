# rpg.py
# Multiplayer text RPG server using FastAPI + WebSockets
# - settings.json (same folder) stores API key, world style, difficulty, and model choices
# - Uses Google Gemini API via REST (httpx)
# - One text-generation call per turn (after initial world gen) with structured JSON output
# - Optional image generation per turn via gemini-2.5-flash-image-preview
# - __main__ entry-point wraps uvicorn with IPv4/IPv6 helpers for local hosting

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import secrets
import sys
import time
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
RESOURCES_DIR = APP_DIR / "resources"
SETTINGS_FILE = APP_DIR / "settings.json"
PROMPT_FILE = APP_DIR / "gm_prompt.txt"
PROMPT_FILES = {
    "en": PROMPT_FILE,
    "de": APP_DIR / "gm_prompt.de.txt",
}
_GM_PROMPT_CACHE: Dict[str, str] = {}
FAVICON_FILE = APP_DIR / "static" / "favicon.ico"
TURN_DIRECTIVE_TOKEN = "<<TURN_DIRECTIVE>>"

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {"en", "de"}

HISTORY_MODE_FULL = "full"
HISTORY_MODE_SUMMARY = "summary"
HISTORY_MODE_OPTIONS = {HISTORY_MODE_FULL, HISTORY_MODE_SUMMARY}
MAX_HISTORY_SUMMARY_BULLETS = 12
MAX_HISTORY_SUMMARY_CHARS = 280

_LANGUAGE_CODE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "american english": "en",
    "british english": "en",
    "us english": "en",
    "uk english": "en",
    "en-us": "en",
    "en-gb": "en",
    "en-au": "en",
    "en-ca": "en",
    "en-in": "en",
    "global english": "en",
    "general english": "en",
    "de": "de",
    "ger": "de",
    "deu": "de",
    "german": "de",
    "standard german": "de",
    "german standard": "de",
    "german (standard)": "de",
    "deutsch": "de",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
}


def normalize_language(lang: Optional[str]) -> str:
    normalized = _normalize_language_code(lang)
    if normalized:
        return normalized
    return DEFAULT_LANGUAGE


def _normalize_language_code(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # Remove trailing qualifiers so "German (Standard)" becomes "german".
    for sep in ("(", "/", "|"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    text = text.replace("_", "-")
    candidates = {text}
    if "-" in text:
        head = text.split("-", 1)[0].strip()
        if head:
            candidates.add(head)
        candidates.add(text.replace("-", ""))
    condensed = text.replace("-", " ")
    if condensed:
        candidates.add(condensed)
    normalized = None
    for candidate in candidates:
        if not candidate:
            continue
        mapped = _LANGUAGE_CODE_ALIASES.get(candidate)
        if mapped:
            normalized = mapped
            break
        if candidate in SUPPORTED_LANGUAGES:
            normalized = candidate
            break
    return normalized


def normalize_history_mode(value: Any) -> str:
    if value is None:
        return HISTORY_MODE_FULL
    text = str(value).strip().lower()
    if text in HISTORY_MODE_OPTIONS:
        return text
    return HISTORY_MODE_FULL


_ZW_AND_SOFT = re.compile(r"[\u00AD\u200B\u200C\u200D\u2060\uFEFF]")
_MIDWORD_NL = re.compile(r"(?<=\S)[ \t]*\n[ \t]*(?=\S)")
_ESCAPED_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")


def sanitize_narrative(text: Any) -> str:
    """Normalize GM narrative text to avoid mid-word breaks and hidden control chars."""
    if not isinstance(text, str):
        if text is None:
            return ""
        text = str(text)
    normalized = unicodedata.normalize("NFC", text)
    cleaned = _ZW_AND_SOFT.sub("", normalized)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MIDWORD_NL.sub(" ", cleaned)
    if "\\u" in cleaned:
        cleaned = _ESCAPED_UNICODE.sub(lambda m: chr(int(m.group(1), 16)), cleaned)
    return cleaned.strip()


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_API_BASE", os.getenv("ELEVENLABS_BASE_URL", "https://api-global-preview.elevenlabs.io"))
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
ELEVENLABS_VOICE_SETTINGS_DEFAULTS = {
    "stability": _env_float("ELEVENLABS_STABILITY", 0.30),
    "similarity_boost": _env_float("ELEVENLABS_SIMILARITY_BOOST", 0.85),
    "style": _env_float("ELEVENLABS_STYLE", 0.60),
    "speed": _env_float("ELEVENLABS_SPEED", 0.98),
}
_ELEVENLABS_IMPORT_ERROR_LOGGED = False
_ELEVENLABS_API_KEY_WARNING_LOGGED = False
_ELEVENLABS_LIBRARY_WARNING_LOGGED = False


class ElevenLabsNarrationError(Exception):
    """Error raised when ElevenLabs narration fails in a recoverable way."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _format_elevenlabs_exception(exc: Exception) -> str:
    """Extract a human-friendly description from ElevenLabs client exceptions."""
    parts: List[str] = []

    text = str(exc).strip()
    if text and text.lower() not in {"", exc.__class__.__name__.lower()}:
        parts.append(text)

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        parts.append(f"HTTP {status_code}")

    body = getattr(exc, "body", None)
    if body:
        try:
            body_text = json.dumps(body, ensure_ascii=True)
        except TypeError:
            body_text = repr(body)
        parts.append(body_text)

    headers = getattr(exc, "headers", None)
    if headers:
        try:
            header_text = json.dumps(headers, ensure_ascii=True)
        except TypeError:
            header_text = repr(headers)
        parts.append(header_text)

    if not parts:
        parts.append(exc.__class__.__name__)

    return "; ".join(parts)


PUBLIC_IP_CACHE: Dict[str, Optional[str]] = {"value": None, "timestamp": 0.0, "failure_ts": 0.0}
PUBLIC_IP_CACHE_TTL = 300.0


def load_gm_prompt(language: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(language)
    cached = _GM_PROMPT_CACHE.get(lang)
    if cached is not None:
        return cached

    path = PROMPT_FILES.get(lang)
    if not path or not path.exists():
        if lang != DEFAULT_LANGUAGE:
            fallback = load_gm_prompt(DEFAULT_LANGUAGE)
            _GM_PROMPT_CACHE[lang] = fallback
            return fallback
        raise RuntimeError(f"Missing GM prompt template: {PROMPT_FILE}")

    text = path.read_text(encoding="utf-8")
    _GM_PROMPT_CACHE[lang] = text
    return text


async def fetch_public_ip() -> Optional[str]:
    """Best-effort lookup of the host's public IP for sharing with remote players."""
    now = time.time()
    cached = PUBLIC_IP_CACHE.get("value")
    if cached and (now - PUBLIC_IP_CACHE.get("timestamp", 0.0)) < PUBLIC_IP_CACHE_TTL:
        return cached

    last_failure = PUBLIC_IP_CACHE.get("failure_ts", 0.0)
    if (cached is None) and (now - last_failure) < 60:
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            preferred_ip: Optional[str] = None
            fallback_ip: Optional[str] = None
            for endpoint in (
                "https://api64.ipify.org",
                "https://api6.ipify.org",
                "https://api.ipify.org",
            ):
                try:
                    resp = await client.get(endpoint)
                    resp.raise_for_status()
                    ip = resp.text.strip()
                except Exception:
                    continue

                if not ip:
                    continue
                if ":" in ip:
                    preferred_ip = ip
                    break
                if fallback_ip is None:
                    fallback_ip = ip

            ip = preferred_ip or fallback_ip
            if ip:
                PUBLIC_IP_CACHE["value"] = ip
                PUBLIC_IP_CACHE["timestamp"] = now
                PUBLIC_IP_CACHE["failure_ts"] = 0.0
                return ip
    except Exception:
        PUBLIC_IP_CACHE["failure_ts"] = now
    return None


GM_PROMPT_TEMPLATE = load_gm_prompt()

# -------- Defaults --------
DEFAULT_SETTINGS = {
    "api_key": "",
    "grok_api_key": "",
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "text_model": "grok-4-fast-non-reasoning",
    "image_model": "gemini-2.5-flash-image-preview",
    "narration_model": ELEVENLABS_MODEL_ID,
    "world_style": "High fantasy",
    "difficulty": "Normal",  # Trivial, Easy, Normal, Hard, Impossible
    "thinking_mode": "none",  # none, brief, balanced, deep
    "language": DEFAULT_LANGUAGE,
    "history_mode": HISTORY_MODE_FULL,
}

LANGUAGE_RULES = {
    "en": "LANGUAGE: Write all narrative text and player-facing strings in English.\n",
    "de": (
        "SPRACHE: Formuliere sämtliche narrativen Texte und spielerseitigen Ausgaben vollständig auf Deutsch. "
        "Das Feld img muss jedoch auf Englisch bleiben.\n"
    ),
}

TURN_DIRECTIVES_INITIAL = {
    "en": (
        "INITIAL TURN: Create the opening scenario AND, for each player where players[pid].pending_join == true, "
        "create their full character kit (cls/ab/inv/cond). Do NOT create characters for players without pending_join.\n"
    ),
    "de": (
        "ERSTE RUNDE: Erzeuge die Eröffnungsszene UND erstelle für jeden Spieler mit players[pid].pending_join == true "
        "das vollständige Charakterpaket (cls/ab/inv/cond). Erstelle KEINE Charaktere für Spieler ohne pending_join.\n"
    ),
}

TURN_DIRECTIVES_ONGOING = {
    "en": (
        "ONGOING TURN: Resolve all submissions. Naturally integrate any players with players[pid].pending_join == true, "
        "and provide narrative closure for any players with players[pid].pending_leave == true so their departure feels organic. "
        "Use only known player IDs and do not invent new players.\n"
    ),
    "de": (
        "LAUFENDE RUNDE: Löse alle eingereichten Aktionen auf. Integriere Spieler mit players[pid].pending_join == true "
        "organisch in die Szene und gib Spielern mit players[pid].pending_leave == true einen erzählerischen Abschied. "
        "Verwende nur bekannte Spieler-IDs und erfinde keine neuen.\n"
    ),
}

THINKING_DIRECTIVES_TEXT = {
    "en": {
        "none": (
            "THINKING MODE: Respond decisively with the minimum internal deliberation. "
            "Avoid lengthy planning and never expose chain-of-thought; return only the required structured outputs.\n"
        ),
        "brief": (
            "THINKING MODE: Take a short internal moment to ensure consistency before responding. "
            "Keep reasoning concise and do not emit hidden deliberations.\n"
        ),
        "balanced": (
            "THINKING MODE: Apply balanced internal reasoning—enough to keep the story coherent and fair—"
            "while keeping the response limited to the structured fields and narrative.\n"
        ),
        "deep": (
            "THINKING MODE: Use thorough internal reasoning to maintain continuity and fairness. "
            "Keep the final response clean JSON + narrative without revealing your chain-of-thought.\n"
        ),
    },
    "de": {
        "none": (
            "DENKMODUS: Antworte entschlossen mit minimalem inneren Abwägen. "
            "Vermeide lange Planungen und gib niemals deine Gedankengänge preis; liefere ausschließlich die geforderten strukturierten Ausgaben.\n"
        ),
        "brief": (
            "DENKMODUS: Nimm dir kurz Zeit, um die Konsistenz zu prüfen, bevor du antwortest. "
            "Halte das Nachdenken knapp und gib keine verborgenen Überlegungen aus.\n"
        ),
        "balanced": (
            "DENKMODUS: Nutze ausgewogenes inneres Nachdenken – genug, um die Geschichte stimmig und fair zu halten – "
            "und beschränke die Ausgabe weiterhin auf die strukturierten Felder und die Erzählung.\n"
        ),
        "deep": (
            "DENKMODUS: Nutze ausführliches inneres Nachdenken, um Kontinuität und Fairness zu bewahren. "
            "Die finale Antwort bleibt reines JSON plus Erzähltext ohne offengelegte Gedankengänge.\n"
        ),
    },
}

USER_PAYLOAD_NOTES = {
    "en": (
        "Players only see their own abilities/inventory/conditions; others see one-word status only. "
        "Update 'pub' with an entry for every current player each turn (word = single lowercase token, no hyphens/punctuation). "
        "For 'upd', include entries for all players where pending_join == true (full cls/ab/inv/cond). "
        "For existing players, include an 'upd' entry only when something changed this turn, and always send full lists, not diffs. "
        "Use only provided player IDs; unknown or invented IDs will be ignored. When pending_leave == true, provide narrative closure this turn and omit that player from future updates. "
        "When history_mode == \"summary\", history and history_summary contain a concise bullet list of prior turns—use it to maintain continuity without expecting full transcripts."
    ),
    "de": (
        "Spieler sehen nur ihre eigenen Fähigkeiten/Inventare/Zustände; andere erhalten lediglich ein einzelnes Statuswort. "
        "Aktualisiere 'pub' in jeder Runde mit einem Eintrag für alle aktuellen Spieler (word = einzelnes kleingeschriebenes Wort ohne Bindestriche oder Satzzeichen). "
        "Für 'upd' Einträge für alle Spieler mit pending_join == true aufnehmen (vollständige cls/ab/inv/cond). "
        "Für bestehende Spieler nur dann einen 'upd'-Eintrag senden, wenn sich in dieser Runde etwas geändert hat, und stets vollständige Listen statt Deltas liefern. "
        "Verwende ausschließlich die bereitgestellten Spieler-IDs; unbekannte oder erfundene IDs werden ignoriert. Ist pending_leave == true, sorge in dieser Runde für einen erzählerischen Abschluss und lasse diesen Spieler in zukünftigen Updates weg. "
        "Wenn history_mode == \"summary\" ist, liefern history und history_summary eine knappe Aufzählung früherer Züge – stütze dich darauf, ohne vollständige Protokolle zu erwarten."
    ),
}

ANNOUNCEMENTS = {
    "new_player": {
        "en": "A new player has joined the party.",
        "de": "Ein neuer Spieler ist der Gruppe beigetreten.",
    }
}

THINKING_MODES = {"none", "brief", "balanced", "deep"}

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a concise, factual bullet summary of a cooperative tabletop RPG session. "
    "Update the chronicle so it stays short, chronological, and captures how the party reached "
    "the present situation, including key discoveries, threats, resources, and character states. "
    "Use plain declarative bullet items without embellishment or duplicate facts."
)

SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "maxItems": MAX_HISTORY_SUMMARY_BULLETS,
        }
    },
    "required": ["summary"],
    "propertyOrdering": ["summary"],
}


PRICING_FILE = APP_DIR / "model_prices.json"


def _load_model_pricing() -> Dict[str, Any]:
    try:
        data = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return {}


MODEL_PRICING_DATA = _load_model_pricing()
TEXT_MODEL_PRICES: Dict[str, Any] = (
    MODEL_PRICING_DATA.get("text", {}).get("models")
    if isinstance(MODEL_PRICING_DATA.get("text", {}).get("models"), dict)
    else {}
)
IMAGE_MODEL_PRICES: Dict[str, Any] = (
    MODEL_PRICING_DATA.get("image", {}).get("models")
    if isinstance(MODEL_PRICING_DATA.get("image", {}).get("models"), dict)
    else {}
)
NARRATION_MODEL_PRICES: Dict[str, Any] = (
    MODEL_PRICING_DATA.get("narration", {}).get("models")
    if isinstance(MODEL_PRICING_DATA.get("narration", {}).get("models"), dict)
    else {}
)


def _lookup_model_pricing(data: Dict[str, Any], model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not model_id or not isinstance(model_id, str):
        return None
    direct = data.get(model_id)
    if isinstance(direct, dict):
        return direct
    lowered = model_id.lower()
    for key, value in data.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
            return value
    return None


def _normalize_model_name(model: Optional[str]) -> str:
    if not model:
        return ""
    return model.split("/")[-1]


def _select_price_tier(price_list: Any, prompt_tokens: Optional[int]) -> Optional[float]:
    if not isinstance(price_list, list) or not price_list:
        return None
    tokens = prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else 0

    def tier_key(tier: Dict[str, Any]) -> float:
        max_tokens = tier.get("max_prompt_tokens")
        return float(max_tokens) if isinstance(max_tokens, (int, float)) else float("inf")

    sorted_tiers = sorted(
        [tier for tier in price_list if isinstance(tier, dict)],
        key=tier_key,
    )
    for tier in sorted_tiers:
        max_tokens = tier.get("max_prompt_tokens")
        price = tier.get("usd_per_million")
        if not isinstance(price, (int, float)):
            continue
        if max_tokens is None:
            return float(price)
        if isinstance(max_tokens, (int, float)) and tokens <= max_tokens:
            return float(price)
    for tier in reversed(sorted_tiers):
        price = tier.get("usd_per_million")
        if isinstance(price, (int, float)):
            return float(price)
    return None


def calculate_turn_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[Dict[str, float]]:
    """Estimate USD costs for a turn given prompt/completion token counts."""
    model_key = _normalize_model_name(model)
    pricing = TEXT_MODEL_PRICES.get(model_key)
    if not isinstance(pricing, dict):
        return None

    prompt_price = _select_price_tier(pricing.get("prompt"), prompt_tokens)
    completion_price = _select_price_tier(pricing.get("completion"), completion_tokens)

    if prompt_price is None and completion_price is None:
        return None

    prompt_tok = prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens > 0 else 0
    completion_tok = completion_tokens if isinstance(completion_tokens, int) and completion_tokens > 0 else 0

    prompt_cost = (prompt_tok / 1_000_000) * prompt_price if prompt_price is not None else 0.0
    completion_cost = (
        (completion_tok / 1_000_000) * completion_price if completion_price is not None else 0.0
    )
    total = prompt_cost + completion_cost

    return {
        "model": model_key,
        "prompt_usd": prompt_cost,
        "completion_usd": completion_cost,
        "total_usd": total,
    }


def calculate_image_cost(
    model: Optional[str],
    *,
    tier: str = "standard",
    images: int = 1,
) -> Optional[Dict[str, Any]]:
    model_key = _normalize_model_name(model)
    pricing = IMAGE_MODEL_PRICES.get(model_key)
    if not isinstance(pricing, dict) or images <= 0:
        return None

    tier_key = tier or "standard"
    tier_info = pricing.get(tier_key)
    if not isinstance(tier_info, dict):
        tier_info = None
        for key, info in pricing.items():
            if isinstance(info, dict) and "usd_per_image" in info:
                tier_info = info
                tier_key = key
                break
    if not isinstance(tier_info, dict):
        return None

    usd_per_image = tier_info.get("usd_per_image")
    if not isinstance(usd_per_image, (int, float)):
        return None
    tokens_per_image = tier_info.get("tokens_per_image")
    cost_usd = float(usd_per_image) * images

    return {
        "model": model_key,
        "tier": tier_key,
        "images": images,
        "usd_per_image": float(usd_per_image),
        "tokens_per_image": tokens_per_image if isinstance(tokens_per_image, int) else None,
        "cost_usd": cost_usd,
    }


def record_image_usage(model: Optional[str], *, purpose: str, tier: str = "standard", images: int = 1) -> None:
    model_key = _normalize_model_name(model)
    STATE.last_image_model = model_key or model
    STATE.last_image_kind = purpose
    STATE.last_image_count = images if images > 0 else 0
    STATE.last_image_turn_index = STATE.turn_index
    cost_info = calculate_image_cost(model, tier=tier, images=images)

    STATE.last_image_tier = cost_info.get("tier") if cost_info else tier
    if cost_info:
        cost_value = cost_info.get("cost_usd")
        per_image_value = cost_info.get("usd_per_image")
        tokens_value = cost_info.get("tokens_per_image")
        STATE.last_image_cost_usd = float(cost_value) if isinstance(cost_value, (int, float)) else None
        STATE.last_image_usd_per = (
            float(per_image_value) if isinstance(per_image_value, (int, float)) else None
        )
        STATE.last_image_tokens = tokens_value if isinstance(tokens_value, int) else None
        if isinstance(cost_value, (int, float)):
            STATE.session_image_cost_usd += float(cost_value)
        if purpose == "scene":
            STATE.last_scene_image_model = STATE.last_image_model
            STATE.last_scene_image_cost_usd = STATE.last_image_cost_usd
            STATE.last_scene_image_usd_per = STATE.last_image_usd_per
            STATE.last_scene_image_turn_index = STATE.turn_index
    else:
        STATE.last_image_cost_usd = None
        STATE.last_image_usd_per = None
        STATE.last_image_tokens = None
        if purpose == "scene":
            STATE.last_scene_image_model = STATE.last_image_model
            STATE.last_scene_image_cost_usd = None
            STATE.last_scene_image_usd_per = None
            STATE.last_scene_image_turn_index = STATE.turn_index

    if images > 0:
        STATE.session_image_requests += images
        STATE.session_image_kind_counts[purpose] = (
            STATE.session_image_kind_counts.get(purpose, 0) + images
        )
    normalized = (purpose or "unknown").strip().lower()
    if normalized in {"scene", "portrait"}:
        key = normalized
    else:
        key = "other"
    STATE.turn_image_kind_counts[key] = STATE.turn_image_kind_counts.get(key, 0) + images

# -------- Text model providers --------
TEXT_PROVIDER_GEMINI = "gemini"
TEXT_PROVIDER_GROK = "grok"
TEXT_PROVIDER_OPENAI = "openai"


def detect_text_provider(model: Optional[str]) -> str:
    value = (model or "").strip().lower()
    if not value:
        return TEXT_PROVIDER_GEMINI
    if value.startswith("models/"):
        value = value.split("/", 1)[1]
    if value.startswith("openai/"):
        value = value.split("/", 1)[1]
    if value.startswith("grok-") or value.startswith("xai-") or "grok" in value:
        return TEXT_PROVIDER_GROK
    openai_markers = (
        "gpt-",
        "gpt4",
        "gpt-4",
        "chatgpt",
        "omni-",
        "omni.",
        "o1-",
        "o3-",
        "o1.",
        "o3.",
        "text-davinci",
        "text-curie",
        "text-babbage",
        "text-ada",
        "davinci-",
        "curie-",
        "babbage-",
        "ada-",
    )
    if any(marker in value for marker in openai_markers):
        return TEXT_PROVIDER_OPENAI
    if "openai" in value:
        return TEXT_PROVIDER_OPENAI
    return TEXT_PROVIDER_GEMINI


def require_text_api_key(provider: str) -> str:
    if provider == TEXT_PROVIDER_GROK:
        api_key = (STATE.settings.get("grok_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Grok API key is not set in settings.")
        return api_key
    if provider == TEXT_PROVIDER_OPENAI:
        api_key = (STATE.settings.get("openai_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is not set in settings.")
        return api_key
    api_key = (STATE.settings.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is not set in settings.")
    return api_key


def _convert_schema_for_jsonschema(schema: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, inner in value.items():
                if key == "type" and isinstance(inner, str):
                    mapping = {
                        "OBJECT": "object",
                        "ARRAY": "array",
                        "STRING": "string",
                        "NUMBER": "number",
                        "INTEGER": "integer",
                        "BOOLEAN": "boolean",
                    }
                    result[key] = mapping.get(inner.upper(), inner.lower())
                elif key == "propertyOrdering":
                    continue
                else:
                    result[key] = _convert(inner)
            return result
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    converted = _convert(copy.deepcopy(schema))

    def _ensure_additional_props(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for child in node.values():
                _ensure_additional_props(child)
        elif isinstance(node, list):
            for item in node:
                _ensure_additional_props(item)

    _ensure_additional_props(converted)
    return converted


# -------- Gemini endpoints (REST) --------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELS_LIST_URL = f"{GEMINI_BASE}/models"
GENERATE_CONTENT_URL = f"{GEMINI_BASE}/models/{{model}}:generateContent"

GROK_BASE = "https://api.x.ai/v1"
GROK_MODELS_URL = f"{GROK_BASE}/models"
GROK_CHAT_COMPLETIONS_URL = f"{GROK_BASE}/chat/completions"

OPENAI_BASE = "https://api.openai.com/v1"
OPENAI_MODELS_URL = f"{OPENAI_BASE}/models"
OPENAI_RESPONSES_URL = f"{OPENAI_BASE}/responses"


# -------------------- Data models (server state) --------------------
class Ability(BaseModel):
    n: str  # name
    x: str  # expertise: novice|apprentice|journeyman|expert|master


class PlayerUpdate(BaseModel):
    pid: str
    cls: str
    ab: List[Ability]
    inv: List[str]
    cond: List[str]


class PublicStatus(BaseModel):
    pid: str  # player id
    word: str  # one-word public status


class TurnStructured(BaseModel):
    """Structured output we expect from the Gemini text model per turn."""
    nar: str  # narrative for the next scenario
    img: str  # image prompt for gemini-2.5-flash-image-preview
    pub: List[PublicStatus]
    upd: List[PlayerUpdate]


class SummaryStructured(BaseModel):
    """Bullet summary returned when history compression is enabled."""
    summary: List[str]


@dataclass
class PlayerPortrait:
    data_url: str
    prompt: str
    updated_at: float


def portrait_payload(portrait: Optional["PlayerPortrait"]) -> Optional[Dict[str, Any]]:
    if not portrait:
        return None
    return {
        "data_url": portrait.data_url,
        "prompt": portrait.prompt,
        "updated_at": portrait.updated_at,
    }


@dataclass
class Player:
    id: str
    name: str
    background: str
    cls: str = ""
    abilities: List[Ability] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    status_word: str = "Unknown"
    connected: bool = True
    pending_join: bool = True  # becomes False after first processed turn
    pending_leave: bool = False  # set when the player disconnects and awaits narrative exit
    token: str = field(default="", repr=False, compare=False)
    # WebSocket connections for private pushes
    sockets: Set[WebSocket] = field(default_factory=set, repr=False, compare=False)
    portrait: Optional[PlayerPortrait] = None


@dataclass
class TurnRecord:
    index: int
    narrative: str
    image_prompt: str
    timestamp: float


@dataclass
class LockState:
    active: bool = False
    reason: str = ""  # "resolving_turn" | "generating_image" | "generating_portrait"


@dataclass
class GameState:
    settings: Dict = field(default_factory=lambda: DEFAULT_SETTINGS.copy())
    players: Dict[str, Player] = field(default_factory=dict)  # player_id -> Player
    submissions: Dict[str, str] = field(default_factory=dict)  # player_id -> text
    current_scenario: str = ""
    turn_index: int = 0
    history: List[TurnRecord] = field(default_factory=list)
    history_summary: List[str] = field(default_factory=list)
    lock: LockState = field(default_factory=LockState)
    language: str = DEFAULT_LANGUAGE
    global_sockets: Set[WebSocket] = field(default_factory=set, repr=False, compare=False)
    # last generated image
    last_image_data_url: Optional[str] = None
    last_image_prompt: Optional[str] = None
    # token usage metadata returned by the last Gemini text call
    last_token_usage: Dict[str, Optional[int]] = field(default_factory=dict)
    last_turn_runtime: Optional[float] = None
    session_token_usage: Dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0, "thinking": 0}
    )
    session_request_count: int = 0
    last_cost_usd: Optional[float] = None
    session_cost_usd: float = 0.0
    last_image_model: Optional[str] = None
    last_image_tier: Optional[str] = None
    last_image_kind: Optional[str] = None
    last_image_cost_usd: Optional[float] = None
    last_image_usd_per: Optional[float] = None
    last_image_tokens: Optional[int] = None
    last_image_count: int = 0
    last_image_turn_index: Optional[int] = None
    session_image_cost_usd: float = 0.0
    session_image_requests: int = 0
    last_scene_image_cost_usd: Optional[float] = None
    last_scene_image_usd_per: Optional[float] = None
    last_scene_image_model: Optional[str] = None
    last_scene_image_turn_index: Optional[int] = None
    session_image_kind_counts: Dict[str, int] = field(default_factory=dict)
    turn_image_kind_counts: Dict[str, int] = field(default_factory=dict)
    last_tts_model: Optional[str] = None
    last_tts_voice_id: Optional[str] = None
    last_tts_characters: Optional[int] = None
    last_tts_character_source: Optional[str] = None
    last_tts_credits: Optional[float] = None
    last_tts_cost_usd: Optional[float] = None
    last_tts_request_id: Optional[str] = None
    last_tts_headers: Dict[str, str] = field(default_factory=dict)
    last_tts_turn_index: Optional[int] = None
    last_tts_total_credits: Optional[int] = None
    last_tts_remaining_credits: Optional[int] = None
    last_tts_next_reset_unix: Optional[int] = None
    session_tts_characters: int = 0
    session_tts_credits: float = 0.0
    session_tts_cost_usd: float = 0.0
    session_tts_requests: int = 0
    auto_image_enabled: bool = False
    auto_tts_enabled: bool = False
    last_text_request: Dict[str, Any] = field(default_factory=dict)
    last_text_response: Dict[str, Any] = field(default_factory=dict)
    last_scenario_request: Dict[str, Any] = field(default_factory=dict)
    last_scenario_response: Dict[str, Any] = field(default_factory=dict)

    def public_snapshot(self) -> Dict:
        """Sanitized state for all players."""
        return {
            "turn_index": self.turn_index,
            "current_scenario": self.current_scenario,
            "world_style": self.settings.get("world_style", "High fantasy"),
            "difficulty": self.settings.get("difficulty", "Normal"),
            "history_mode": self.settings.get("history_mode", HISTORY_MODE_FULL),
            "history_summary": list(self.history_summary),
            "language": self.language,
            "auto_image_enabled": self.auto_image_enabled,
            "auto_tts_enabled": self.auto_tts_enabled,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "cls": p.cls,
                    "status_word": p.status_word,
                    "connected": p.connected,
                    "pending_join": p.pending_join,
                    "pending_leave": p.pending_leave,
                    "portrait": portrait_payload(p.portrait),
                }
                for p in self.players.values()
            ],
            "submissions": [
                {
                    "name": self.players.get(pid).name if pid in self.players else "Unknown",
                    "text": txt,
                }
                for pid, txt in self.submissions.items()
            ],
            "lock": {"active": self.lock.active, "reason": self.lock.reason},
            "image": {
                "data_url": self.last_image_data_url,
                "prompt": self.last_image_prompt,
            },
            "token_usage": self._token_usage_snapshot(),
        }

    def private_snapshot_for(self, player_id: str) -> Dict:
        p = self.players.get(player_id)
        if not p:
            return {}
        return {
            "you": {
                "id": p.id,
                "name": p.name,
                "class": p.cls,
                "pending_join": p.pending_join,
                "pending_leave": p.pending_leave,
                "abilities": [a.model_dump() for a in p.abilities],
                "inventory": p.inventory,
                "conditions": p.conditions,
                "portrait": portrait_payload(p.portrait),
            }
        }
    
    def tokens_per_second(self) -> Optional[float]:
        total_tokens = sum(
            val for val in [
                self.last_token_usage.get("input"),
                self.last_token_usage.get("output"),
                self.last_token_usage.get("thinking"),
            ]
            if isinstance(val, int)
        )
        if not total_tokens or not self.last_turn_runtime or self.last_turn_runtime <= 0:
            return None
        return total_tokens / self.last_turn_runtime

    def _token_usage_snapshot(self) -> Dict:
        last_input = self.last_token_usage.get("input")
        last_output = self.last_token_usage.get("output")
        last_thinking = self.last_token_usage.get("thinking")

        def _sum_tokens(values: List[Optional[int]]) -> Optional[int]:
            ints = [v for v in values if isinstance(v, int)]
            return sum(ints) if ints else None

        last_total = _sum_tokens([last_input, last_output, last_thinking])

        session_input = self.session_token_usage.get("input", 0)
        session_output = self.session_token_usage.get("output", 0)
        session_thinking = self.session_token_usage.get("thinking", 0)
        session_total = session_input + session_output + session_thinking

        last_image = {
            "model": self.last_image_model,
            "tier": self.last_image_tier,
            "kind": self.last_image_kind,
            "images": self.last_image_count or None,
            "usd_per_image": self.last_image_usd_per,
            "tokens_per_image": self.last_image_tokens,
            "cost_usd": self.last_image_cost_usd,
            "turn_index": self.last_image_turn_index,
        }

        session_images = self.session_image_requests
        image_session = {
            "images": session_images,
            "cost_usd": self.session_image_cost_usd,
            "avg_usd_per_image": (
                (self.session_image_cost_usd / session_images) if session_images else None
            ),
            "by_kind": dict(self.session_image_kind_counts),
        }

        image_turn = {
            "by_kind": dict(self.turn_image_kind_counts),
        }

        image_cost_same_turn = 0.0
        if (
            isinstance(self.last_scene_image_cost_usd, (int, float))
            and self.last_scene_image_turn_index == self.turn_index
        ):
            image_cost_same_turn = float(self.last_scene_image_cost_usd)

        narration_price_entry = _lookup_model_pricing(NARRATION_MODEL_PRICES, self.last_tts_model)
        narration_usd_per_million: Optional[float] = None
        if narration_price_entry:
            usd_value = narration_price_entry.get("usd_per_million")
            if isinstance(usd_value, (int, float)):
                narration_usd_per_million = float(usd_value)

        narration_last_headers = {
            k: v
            for k, v in (self.last_tts_headers or {}).items()
            if isinstance(k, str) and isinstance(v, str)
        }

        narration_last = {
            "model": self.last_tts_model,
            "voice_id": self.last_tts_voice_id,
            "characters": self.last_tts_characters,
            "character_source": self.last_tts_character_source,
            "credits": self.last_tts_credits,
            "cost_usd": self.last_tts_cost_usd,
            "usd_per_million": narration_usd_per_million,
            "request_id": self.last_tts_request_id,
            "headers": narration_last_headers,
            "total_credits": self.last_tts_total_credits,
            "remaining_credits": self.last_tts_remaining_credits,
            "next_reset_unix": self.last_tts_next_reset_unix,
        }

        narration_session = {
            "characters": self.session_tts_characters,
            "credits": self.session_tts_credits,
            "cost_usd": self.session_tts_cost_usd,
            "requests": self.session_tts_requests,
            "remaining_credits": self.last_tts_remaining_credits,
        }

        narration_cost_same_turn = 0.0
        if (
            isinstance(self.last_tts_cost_usd, (int, float))
            and self.last_tts_turn_index == self.turn_index
        ):
            narration_cost_same_turn = float(self.last_tts_cost_usd)

        total_last_cost = (
            float(self.last_cost_usd or 0.0)
            + image_cost_same_turn
            + narration_cost_same_turn
        )
        total_session_cost = (
            self.session_cost_usd
            + self.session_image_cost_usd
            + self.session_tts_cost_usd
        )

        return {
            "last_turn": {
                "input": last_input,
                "output": last_output,
                "thinking": last_thinking,
                "total": last_total,
                "tokens_per_sec": self.tokens_per_second(),
                "cost_usd": self.last_cost_usd,
            },
            "session": {
                "input": session_input,
                "output": session_output,
                "thinking": session_thinking,
                "total": session_total,
                "requests": self.session_request_count,
                "cost_usd": self.session_cost_usd,
            },
            "image": {
                "last": last_image,
                "session": image_session,
                "turn": image_turn,
                "scene": {
                    "last_cost_usd": self.last_scene_image_cost_usd,
                    "last_turn_index": self.last_scene_image_turn_index,
                    "last_model": self.last_scene_image_model,
                    "usd_per_image": self.last_scene_image_usd_per,
                },
            },
            "narration": {
                "last": narration_last,
                "session": narration_session,
            },
            "totals": {
                "last_usd": total_last_cost,
                "session_usd": total_session_cost,
                "breakdown": {
                    "text_last_usd": self.last_cost_usd,
                    "image_last_usd": self.last_image_cost_usd,
                    "image_last_scene_usd": self.last_scene_image_cost_usd,
                    "image_last_turn_usd": image_cost_same_turn,
                    "narration_last_usd": self.last_tts_cost_usd,
                    "narration_last_turn_usd": narration_cost_same_turn,
                    "text_session_usd": self.session_cost_usd,
                    "image_session_usd": self.session_image_cost_usd,
                    "narration_session_usd": self.session_tts_cost_usd,
                },
            },
        }

STATE = GameState()
STATE_LOCK = asyncio.Lock()  # coarse lock for turn/image operations
SETTINGS_LOCK = asyncio.Lock()  # for reading/writing settings.json
_RESET_CHECK_TASK: Optional[asyncio.Task] = None


def cancel_pending_reset_task() -> None:
    """Cancel any background task that is waiting to reset the session."""

    global _RESET_CHECK_TASK
    task = _RESET_CHECK_TASK
    if task and not task.done():
        task.cancel()
    _RESET_CHECK_TASK = None


def schedule_session_reset_check(delay: float = 3.0) -> None:
    """Schedule a delayed session reset if no players return within *delay* seconds."""

    global _RESET_CHECK_TASK
    cancel_pending_reset_task()

    async def _runner() -> None:
        global _RESET_CHECK_TASK
        try:
            await asyncio.sleep(delay)
            if maybe_reset_session_if_empty():
                await broadcast_public()
        except asyncio.CancelledError:
            pass
        finally:
            _RESET_CHECK_TASK = None

    _RESET_CHECK_TASK = asyncio.create_task(_runner())


def reset_session_progress() -> None:
    """Reset per-session gameplay state while preserving configuration."""

    global _RESET_CHECK_TASK
    _RESET_CHECK_TASK = None
    STATE.players.clear()
    STATE.submissions.clear()
    STATE.current_scenario = ""
    STATE.turn_index = 0
    STATE.history.clear()
    STATE.history_summary.clear()
    STATE.lock = LockState(False, "")
    STATE.last_image_data_url = None
    STATE.last_image_prompt = None
    STATE.last_image_model = None
    STATE.last_image_tier = None
    STATE.last_image_kind = None
    STATE.last_image_cost_usd = None
    STATE.last_image_usd_per = None
    STATE.last_image_tokens = None
    STATE.last_image_count = 0
    STATE.last_image_turn_index = None
    STATE.last_token_usage = {}
    STATE.last_turn_runtime = None
    STATE.session_token_usage = {"input": 0, "output": 0, "thinking": 0}
    STATE.session_request_count = 0
    STATE.last_cost_usd = None
    STATE.session_cost_usd = 0.0
    STATE.session_image_cost_usd = 0.0
    STATE.session_image_requests = 0
    STATE.session_image_kind_counts = {}
    STATE.last_scene_image_cost_usd = None
    STATE.last_scene_image_usd_per = None
    STATE.last_scene_image_model = None
    STATE.last_scene_image_turn_index = None
    STATE.turn_image_kind_counts = {}
    STATE.last_tts_model = None
    STATE.last_tts_voice_id = None
    STATE.last_tts_characters = None
    STATE.last_tts_character_source = None
    STATE.last_tts_credits = None
    STATE.last_tts_cost_usd = None
    STATE.last_tts_request_id = None
    STATE.last_tts_headers = {}
    STATE.last_tts_turn_index = None
    STATE.last_tts_total_credits = None
    STATE.last_tts_remaining_credits = None
    STATE.last_tts_next_reset_unix = None
    STATE.session_tts_characters = 0
    STATE.session_tts_credits = 0.0
    STATE.session_tts_cost_usd = 0.0
    STATE.session_tts_requests = 0


def maybe_reset_session_if_empty() -> bool:
    """Reset the game if no active players remain.

    Returns True when a reset occurred.
    """

    if not STATE.players:
        reset_session_progress()
        return True

    for player in STATE.players.values():
        if player.connected:
            return False
        if not player.pending_leave:
            return False

    reset_session_progress()
    return True


def apply_language_value(lang: Optional[str]) -> bool:
    if lang is None:
        return False
    normalized = normalize_language(lang)
    if normalized == STATE.language:
        return False
    STATE.language = normalized
    STATE.settings["language"] = normalized
    return True


# -------------------- Helpers: settings I/O --------------------
def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def load_settings() -> Dict:
    ensure_settings_file()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = DEFAULT_SETTINGS.copy()
        # Preserve any forward-compatible keys while ensuring defaults exist.
        merged.update(data)
    except Exception:
        merged = DEFAULT_SETTINGS.copy()

    merged["language"] = normalize_language(merged.get("language"))
    merged["history_mode"] = normalize_history_mode(merged.get("history_mode"))
    return merged


async def save_settings(new_settings: Dict):
    transient_keys = {
        "api_key_preview",
        "api_key_set",
        "elevenlabs_api_key_preview",
        "elevenlabs_api_key_set",
        "grok_api_key_preview",
        "grok_api_key_set",
        "openai_api_key_preview",
        "openai_api_key_set",
    }
    sanitized = {k: v for k, v in new_settings.items() if k not in transient_keys}

    # Preserve existing secrets when callers provide empty placeholders.
    for secret_key in ("api_key", "grok_api_key", "openai_api_key", "elevenlabs_api_key"):
        if secret_key not in sanitized:
            continue
        raw_value = sanitized[secret_key]
        if not isinstance(raw_value, str):
            continue
        stripped_value = raw_value.strip()
        if stripped_value:
            sanitized[secret_key] = stripped_value
        else:
            sanitized.pop(secret_key, None)

    async with SETTINGS_LOCK:
        merged = DEFAULT_SETTINGS.copy()
        if SETTINGS_FILE.exists():
            try:
                existing = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    merged.update(existing)
            except Exception:
                pass

        merged.update(sanitized)
        merged["language"] = normalize_language(merged.get("language"))
        merged["history_mode"] = normalize_history_mode(merged.get("history_mode"))
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")

        if new_settings is STATE.settings:
            STATE.settings.clear()
            STATE.settings.update(merged)


# -------------------- Helpers: sockets --------------------
async def _send_json_to_sockets(sockets: Set[WebSocket], payload: Dict):
    dead = []
    for ws in list(sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets.discard(ws)


def _elevenlabs_library_available() -> bool:
    try:
        import importlib

        importlib.import_module("elevenlabs.client")
        importlib.import_module("elevenlabs.types")
    except ImportError:
        return False
    return True


async def _broadcast_tts_error(message: str, turn_index: int) -> None:
    payload = {
        "event": "tts_error",
        "data": {
            "turn_index": turn_index,
            "message": message,
            "severity": "error",
            "ts": time.time(),
        },
    }
    await _send_json_to_sockets(STATE.global_sockets, payload)


async def elevenlabs_list_models(api_key: str) -> List[Dict[str, Any]]:
    """Return available ElevenLabs narration models for the API key."""
    if not api_key:
        return []

    base = ELEVENLABS_BASE_URL.rstrip("/")
    url = f"{base}/v1/models"
    headers = {"xi-api-key": api_key}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:  # noqa: BLE001
        status = exc.response.status_code if exc.response else None
        if status in {401, 403}:
            print(
                "ElevenLabs model list request unauthorized. Check the configured API key.",
                file=sys.stderr,
                flush=True,
            )
            return []
        print(
            f"ElevenLabs model list request failed with HTTP {status}.",
            file=sys.stderr,
            flush=True,
        )
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch ElevenLabs models: {exc!r}", file=sys.stderr, flush=True)
        return []

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        print("ElevenLabs returned a non-JSON model list response.", file=sys.stderr, flush=True)
        return []

    raw_models: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        raw_models = [m for m in payload if isinstance(m, dict)]
    elif isinstance(payload, dict):
        maybe_models = payload.get("models")
        if isinstance(maybe_models, list):
            raw_models = [m for m in maybe_models if isinstance(m, dict)]

    items: List[Dict[str, Any]] = []
    for entry in raw_models:
        model_id = entry.get("model_id") or entry.get("id") or entry.get("modelId")
        if not model_id:
            continue
        model_name = (
            entry.get("name")
            or entry.get("display_name")
            or entry.get("description")
            or str(model_id)
        )
        languages = entry.get("languages") or entry.get("supported_languages")
        parsed_languages: List[str] = []
        if isinstance(languages, list):
            for lang in languages:
                if isinstance(lang, dict):
                    name_parts = [
                        lang.get("name"),
                        lang.get("display_name"),
                        lang.get("language_name"),
                    ]
                    lang_name = next((str(part) for part in name_parts if part), None)
                    if lang_name:
                        parsed_languages.append(lang_name)
                        continue
                    code = lang.get("language_id") or lang.get("id")
                    if code:
                        parsed_languages.append(str(code))
                        continue
                elif isinstance(lang, str):
                    parsed_languages.append(lang)
                else:
                    parsed_languages.append(str(lang))
        languages = parsed_languages
        language_codes: List[str] = []
        for lang_token in languages:
            normalized = _normalize_language_code(lang_token)
            if normalized and normalized not in language_codes:
                language_codes.append(normalized)
        items.append(
            {
                "id": str(model_id),
                "name": str(model_name),
                "languages": languages,
                "language_codes": language_codes,
            }
        )

    return items


def _elevenlabs_convert_to_base64(
    text: str,
    api_key: str,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    global _ELEVENLABS_IMPORT_ERROR_LOGGED
    if not text or not text.strip():
        return {"audio_base64": None, "metadata": {}}
    if not api_key:
        return {"audio_base64": None, "metadata": {}}
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.types import VoiceSettings
    except ImportError:
        if not _ELEVENLABS_IMPORT_ERROR_LOGGED:
            print("ElevenLabs package not available; skipping narration.", file=sys.stderr, flush=True)
            _ELEVENLABS_IMPORT_ERROR_LOGGED = True
        return {"audio_base64": None, "metadata": {}}

    voice_kwargs = {k: v for k, v in ELEVENLABS_VOICE_SETTINGS_DEFAULTS.items() if v is not None}
    resolved_model = (model_id or ELEVENLABS_MODEL_ID or "").strip() or None
    audio_bytes = b""
    response_headers: Dict[str, str] = {}
    try:
        voice_settings = VoiceSettings(**voice_kwargs) if voice_kwargs else None
        client = ElevenLabs(api_key=api_key, base_url=ELEVENLABS_BASE_URL)
        subscription_info = None
        try:
            subscription_info = client.user.subscription.get()
        except Exception:
            subscription_info = None
        with client.text_to_speech._raw_client.convert(
            ELEVENLABS_VOICE_ID,
            text=text,
            model_id=resolved_model,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            voice_settings=voice_settings,
        ) as response:
            response_headers = {
                str(k): str(v)
                for k, v in (response.headers or {}).items()
                if isinstance(k, str)
            }
            audio_bytes = b"".join(response.data)
    except Exception as exc:  # noqa: BLE001
        details = _format_elevenlabs_exception(exc)
        print(
            f"Failed to generate ElevenLabs narration: {details} ({exc!r})",
            file=sys.stderr,
            flush=True,
        )
        raise ElevenLabsNarrationError(details or "ElevenLabs narration failed.") from exc

    if not audio_bytes:
        message = "ElevenLabs returned no audio data for the narration request."
        print(message, file=sys.stderr, flush=True)
        raise ElevenLabsNarrationError(message)

    normalized_headers = {
        k.lower(): v for k, v in response_headers.items() if isinstance(k, str)
    }

    def _extract_first_int(value: str) -> Optional[int]:
        if not isinstance(value, str):
            return None
        for segment in value.split(";"):
            head = segment.strip().split(",")[0].strip()
            try:
                return int(head)
            except ValueError:
                continue
        return None

    character_source = None
    characters_reported: Optional[int] = None
    preferred_keys = [
        "x-characters-used",
        "x-characters",
        "x-character-count",
        "x-total-characters-used",
        "x-content-characters",
        "x-usage-character-count",
    ]
    for key in preferred_keys:
        if key in normalized_headers:
            maybe = _extract_first_int(normalized_headers[key])
            if maybe is not None:
                characters_reported = maybe
                character_source = key
                break
    if characters_reported is None:
        for key, value in normalized_headers.items():
            if "character" in key:
                maybe = _extract_first_int(value)
                if maybe is not None:
                    characters_reported = maybe
                    character_source = key
                    break

    fallback_characters = len(text)
    final_characters = characters_reported if characters_reported is not None else fallback_characters
    if character_source is None:
        character_source = "text_length"

    price_entry = _lookup_model_pricing(NARRATION_MODEL_PRICES, resolved_model or ELEVENLABS_MODEL_ID)
    usd_per_million = None
    credits_per_character = None
    if price_entry:
        if isinstance(price_entry.get("usd_per_million"), (int, float)):
            usd_per_million = float(price_entry["usd_per_million"])
        if isinstance(price_entry.get("credits_per_character"), (int, float)):
            credits_per_character = float(price_entry["credits_per_character"])

    estimated_credits: Optional[float] = None
    if credits_per_character is not None and final_characters is not None:
        estimated_credits = final_characters * credits_per_character

    estimated_cost: Optional[float] = None
    if usd_per_million is not None and final_characters is not None:
        estimated_cost = (final_characters / 1_000_000.0) * usd_per_million

    subscription_total = None
    subscription_used = None
    subscription_next_reset = None
    if subscription_info is not None:
        total = getattr(subscription_info, "character_limit", None)
        used = getattr(subscription_info, "character_count", None)
        next_reset = getattr(subscription_info, "next_character_count_reset_unix", None)
        if isinstance(total, int):
            subscription_total = total
        if isinstance(used, int):
            subscription_used = used
        if isinstance(next_reset, int):
            subscription_next_reset = next_reset

    remaining_credits: Optional[int] = None
    if isinstance(subscription_total, int) and isinstance(subscription_used, int):
        remaining_credits = max(subscription_total - subscription_used, 0)

    metadata = {
        "model_id": resolved_model or ELEVENLABS_MODEL_ID,
        "voice_id": ELEVENLABS_VOICE_ID,
        "characters_reported": characters_reported,
        "character_source": character_source,
        "characters_final": final_characters,
        "estimated_credits": estimated_credits,
        "estimated_cost_usd": estimated_cost,
        "usd_per_million": usd_per_million,
        "credits_per_character": credits_per_character,
        "request_id": normalized_headers.get("x-request-id") or normalized_headers.get("request-id"),
        "headers": {k: normalized_headers[k] for k in sorted(normalized_headers)},
        "subscription_total_credits": subscription_total,
        "subscription_used_credits": subscription_used,
        "subscription_remaining_credits": remaining_credits,
        "subscription_next_reset_unix": subscription_next_reset,
    }

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "metadata": metadata,
    }


async def maybe_queue_tts(text: str, turn_index: int) -> None:
    if not STATE.auto_tts_enabled:
        return
    if not text or not text.strip():
        return
    api_key = (STATE.settings.get("elevenlabs_api_key") or "").strip()
    if not api_key:
        global _ELEVENLABS_API_KEY_WARNING_LOGGED
        if not _ELEVENLABS_API_KEY_WARNING_LOGGED:
            print(
                "ElevenLabs narration is enabled but the ElevenLabs API key is not configured in settings; skipping audio.",
                file=sys.stderr,
                flush=True,
            )
            _ELEVENLABS_API_KEY_WARNING_LOGGED = True
        return
    if not _elevenlabs_library_available():
        global _ELEVENLABS_LIBRARY_WARNING_LOGGED
        if not _ELEVENLABS_LIBRARY_WARNING_LOGGED:
            print(
                "ElevenLabs narration is enabled but the elevenlabs package is not installed; skipping audio.",
                file=sys.stderr,
                flush=True,
            )
            _ELEVENLABS_LIBRARY_WARNING_LOGGED = True
        return

    configured_model = STATE.settings.get("narration_model")
    model_id = (configured_model or ELEVENLABS_MODEL_ID or "").strip()

    async def _worker() -> None:
        try:
            result = await asyncio.to_thread(
                _elevenlabs_convert_to_base64,
                text,
                api_key,
                model_id or None,
            )
        except ElevenLabsNarrationError as exc:
            await _broadcast_tts_error(str(exc) or "ElevenLabs narration failed.", turn_index)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"ElevenLabs narration worker failed: {exc!r}", file=sys.stderr, flush=True)
            await _broadcast_tts_error(
                "Unexpected ElevenLabs narration failure. Check server logs for details.",
                turn_index,
            )
            return
        if not isinstance(result, dict):
            result = {}
        audio_b64 = result.get("audio_base64")
        metadata = result.get("metadata") or {}
        if not audio_b64:
            await _broadcast_tts_error(
                "ElevenLabs returned no audio data for the narration request.",
                turn_index,
            )
            return
        if not STATE.auto_tts_enabled:
            return

        resolved_model_id = metadata.get("model_id") or model_id or ELEVENLABS_MODEL_ID

        characters_final = metadata.get("characters_final")
        credits_estimated = metadata.get("estimated_credits")
        cost_estimated = metadata.get("estimated_cost_usd")
        headers_payload = metadata.get("headers") if isinstance(metadata.get("headers"), dict) else {}

        STATE.last_tts_model = resolved_model_id
        STATE.last_tts_voice_id = ELEVENLABS_VOICE_ID
        STATE.last_tts_request_id = metadata.get("request_id") if isinstance(metadata.get("request_id"), str) else None
        STATE.last_tts_characters = int(characters_final) if isinstance(characters_final, int) else None
        STATE.last_tts_character_source = metadata.get("character_source")
        STATE.last_tts_credits = float(credits_estimated) if isinstance(credits_estimated, (int, float)) else None
        STATE.last_tts_cost_usd = float(cost_estimated) if isinstance(cost_estimated, (int, float)) else None
        STATE.last_tts_turn_index = turn_index
        STATE.last_tts_headers = {
            str(k): str(v)
            for k, v in headers_payload.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        total_credits_meta = metadata.get("subscription_total_credits")
        used_credits_meta = metadata.get("subscription_used_credits")
        remaining_credits_meta = metadata.get("subscription_remaining_credits")
        next_reset_meta = metadata.get("subscription_next_reset_unix")
        STATE.last_tts_total_credits = int(total_credits_meta) if isinstance(total_credits_meta, int) else None
        STATE.last_tts_remaining_credits = int(remaining_credits_meta) if isinstance(remaining_credits_meta, int) else None
        STATE.last_tts_next_reset_unix = int(next_reset_meta) if isinstance(next_reset_meta, int) else None
        if (
            STATE.last_tts_total_credits is None
            and isinstance(total_credits_meta, (float, str))
            and str(total_credits_meta).isdigit()
        ):
            STATE.last_tts_total_credits = int(float(total_credits_meta))
        if (
            STATE.last_tts_remaining_credits is None
            and isinstance(remaining_credits_meta, (float, str))
            and str(remaining_credits_meta).isdigit()
        ):
            STATE.last_tts_remaining_credits = int(float(remaining_credits_meta))

        if isinstance(characters_final, int):
            STATE.session_tts_characters += characters_final
        if isinstance(credits_estimated, (int, float)):
            STATE.session_tts_credits += float(credits_estimated)
        if isinstance(cost_estimated, (int, float)):
            STATE.session_tts_cost_usd += float(cost_estimated)
        STATE.session_tts_requests += 1

        payload = {
            "event": "tts_audio",
            "data": {
                "turn_index": turn_index,
                "audio_base64": audio_b64,
                "format": ELEVENLABS_OUTPUT_FORMAT,
                "voice_id": ELEVENLABS_VOICE_ID,
                "model_id": resolved_model_id,
                "usage": {
                    "characters": characters_final,
                    "character_source": metadata.get("character_source"),
                    "estimated_cost_usd": cost_estimated,
                    "estimated_credits": credits_estimated,
                    "usd_per_million": metadata.get("usd_per_million"),
                },
            },
        }
        await _send_json_to_sockets(STATE.global_sockets, payload)

    asyncio.create_task(_worker())


async def maybe_queue_scene_image(prompt: Optional[str], turn_index: int, *, force: bool = False) -> None:
    if not STATE.auto_image_enabled:
        return
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return
    if not force:
        last_scene_turn = STATE.last_scene_image_turn_index
        if isinstance(last_scene_turn, int) and last_scene_turn == turn_index:
            return

    async def _worker() -> None:
        attempts = 0
        max_attempts = 20
        while True:
            if not STATE.auto_image_enabled:
                return
            if attempts >= max_attempts:
                print(
                    "Auto image generation aborted after repeated retries while the game was busy.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if STATE.lock.active:
                attempts += 1
                await asyncio.sleep(0.5)
                continue

            async with STATE_LOCK:
                if STATE.lock.active:
                    attempts += 1
                    await asyncio.sleep(0.5)
                    continue
                current_prompt = (STATE.last_image_prompt or prompt_text).strip()
                if not current_prompt:
                    return
                if not force:
                    last_turn = STATE.last_scene_image_turn_index
                    if isinstance(last_turn, int) and last_turn == turn_index:
                        return
                STATE.lock = LockState(True, "generating_image")
                await broadcast_public()

            try:
                img_model = STATE.settings.get("image_model") or "gemini-2.5-flash-image-preview"
                data_url = await gemini_generate_image(
                    img_model,
                    current_prompt,
                    purpose="scene",
                )
                STATE.last_image_data_url = data_url
                STATE.last_image_prompt = current_prompt
                await announce("Image generated.")
                await broadcast_public()
            except HTTPException as exc:
                print(f"Auto image generation failed: {exc.detail}", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Auto image generation error: {exc!r}", file=sys.stderr, flush=True)
            finally:
                STATE.lock = LockState(False, "")
                await broadcast_public()
            return

    asyncio.create_task(_worker())


async def broadcast_public():
    payload = {"event": "state", "data": STATE.public_snapshot()}
    await _send_json_to_sockets(STATE.global_sockets, payload)


async def send_private(player_id: str):
    payload = {"event": "private", "data": STATE.private_snapshot_for(player_id)}
    p = STATE.players.get(player_id)
    if not p:
        return
    await _send_json_to_sockets(p.sockets, payload)


async def announce(message: str):
    payload = {"event": "announce", "data": {"message": message, "ts": time.time()}}
    await _send_json_to_sockets(STATE.global_sockets, payload)


def authenticate_player(player_id: str, token: str) -> Player:
    if not player_id or player_id not in STATE.players:
        raise HTTPException(status_code=404, detail="Unknown player.")
    player = STATE.players[player_id]
    if not token or not secrets.compare_digest(token, player.token):
        raise HTTPException(status_code=403, detail="Invalid player token.")
    return player


# -------------------- Helpers: Gemini REST --------------------
def check_api_key() -> str:
    return require_text_api_key(TEXT_PROVIDER_GEMINI)


async def gemini_list_models() -> List[Dict]:
    api_key = check_api_key()
    url = f"{MODELS_LIST_URL}?key={api_key}"
    headers = {"x-goog-api-key": api_key}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Model list failed: {r.text}")
        data = r.json()
        return data.get("models", [])


async def grok_list_models(api_key: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(GROK_MODELS_URL, headers=headers)
    if r.status_code != 200:
        detail_text = r.text or f"HTTP {r.status_code}"
        raise HTTPException(status_code=502, detail=f"Grok model list failed: {detail_text}")
    try:
        data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Malformed Grok model list: {exc}") from exc

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("models"), list):
            return data["models"]
    if isinstance(data, list):
        return data
    return []


async def openai_list_models(api_key: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(OPENAI_MODELS_URL, headers=headers)
    if r.status_code != 200:
        detail_text = r.text or f"HTTP {r.status_code}"
        raise HTTPException(status_code=502, detail=f"OpenAI model list failed: {detail_text}")
    try:
        data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Malformed OpenAI model list: {exc}") from exc

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, list):
        return data
    return []


def _sanitize_request_headers(headers: Dict[str, str]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"x-goog-api-key", "authorization"}:
            sanitized[key] = "***"
        else:
            sanitized[key] = value
    return sanitized


async def _gemini_generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict,
    schema: Dict,
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict:
    """Call Gemini with a JSON schema and return the parsed response."""
    api_key = check_api_key()
    url = GENERATE_CONTENT_URL.format(model=model)
    mode = (STATE.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    effective_temperature = temperature
    if effective_temperature is None:
        effective_temperature = {
            "none": 0.55,
            "brief": 0.7,
            "balanced": 0.9,
            "deep": 1.0,
        }.get(mode, 0.9)

    thinking_budget = None
    if include_thinking_budget:
        thinking_budget = compute_thinking_budget(model, mode)

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}]},
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": effective_temperature,
        },
    }
    if thinking_budget is not None:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    request_snapshot = {
        "timestamp": time.time(),
        "url": url,
        "model": model,
        "provider": TEXT_PROVIDER_GEMINI,
        "thinking_mode": mode,
        "temperature": effective_temperature,
        "thinking_budget": thinking_budget,
        "record_usage": record_usage,
        "headers": _sanitize_request_headers(headers),
        "body": body,
    }
    STATE.last_text_request = request_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_request = copy.deepcopy(request_snapshot)
    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
    elapsed = time.perf_counter() - start_time
    if record_usage:
        STATE.last_turn_runtime = elapsed
    raw_text = getattr(r, "text", "")
    if callable(raw_text):
        try:
            raw_text = raw_text()
        except Exception:
            raw_text = ""
    if raw_text is None:
        raw_text = ""
    try:
        response_json = r.json()
    except Exception:
        response_json = None
    headers_source = getattr(r, "headers", {}) or {}
    if hasattr(headers_source, "items"):
        header_items = headers_source.items()
    else:
        header_items = []
    response_headers = {str(k): str(v) for k, v in header_items}
    response_snapshot = {
        "timestamp": time.time(),
        "status_code": r.status_code,
        "elapsed_seconds": elapsed,
        "headers": response_headers,
        "json": response_json,
        "text": raw_text,
        "provider": TEXT_PROVIDER_GEMINI,
    }
    STATE.last_text_response = response_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_response = copy.deepcopy(response_snapshot)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Text generation failed: {raw_text}")
    if not isinstance(response_json, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")
    data = response_json
    usage_meta = data.get("usageMetadata") or {}
    if record_usage:
        # Capture token usage for UI display; missing values remain None.
        STATE.last_token_usage = {
            "input": usage_meta.get("promptTokenCount"),
            "output": usage_meta.get("candidatesTokenCount"),
            "thinking": usage_meta.get("thoughtsTokenCount") or usage_meta.get("thinkingTokenCount"),
        }
        for key in ("input", "output", "thinking"):
            val = STATE.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                STATE.session_token_usage[key] = STATE.session_token_usage.get(key, 0) + val
        STATE.session_request_count += 1

        prompt_tokens = STATE.last_token_usage.get("input")
        output_tokens = STATE.last_token_usage.get("output")
        thinking_tokens = STATE.last_token_usage.get("thinking")
        combined_output_tokens = sum(
            val for val in [output_tokens, thinking_tokens] if isinstance(val, int) and val > 0
        )
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            STATE.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if STATE.last_cost_usd is not None:
                STATE.session_cost_usd += STATE.last_cost_usd
        else:
            STATE.last_cost_usd = None

    # Text is returned in candidates[0].content.parts[0].text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        txt = ""
        for prt in parts:
            if "text" in prt and prt["text"]:
                txt += prt["text"]
        parsed = json.loads(txt)
    except Exception:
        raise HTTPException(status_code=502, detail="Malformed response from model.")
    return parsed


async def _grok_generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict,
    schema: Dict,
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict:
    api_key = require_text_api_key(TEXT_PROVIDER_GROK)
    url = GROK_CHAT_COMPLETIONS_URL
    mode = (STATE.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    effective_temperature = temperature
    if effective_temperature is None:
        effective_temperature = {
            "none": 0.55,
            "brief": 0.7,
            "balanced": 0.9,
            "deep": 1.0,
        }.get(mode, 0.9)

    json_schema = _convert_schema_for_jsonschema(schema) if schema else None
    response_format: Optional[Dict[str, Any]] = None
    if json_schema and isinstance(json_schema, dict) and json_schema.get("type"):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name or "payload",
                "schema": json_schema,
            },
        }
    else:
        response_format = {"type": "json_object"}

    system_prompt_augmented = system_prompt.rstrip() + (
        "\n\nRespond strictly with a single JSON object matching the schema. "
        "Do not include markdown or any explanatory text."
    )
    payload_text = json.dumps(user_payload, ensure_ascii=False)
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_augmented},
            {
                "role": "user",
                "content": (
                    "Use the following JSON input to determine the response. "
                    "Echo nothing from it except as needed in the output.\n" + payload_text
                ),
            },
        ],
        "temperature": effective_temperature,
        "stream": False,
    }
    if response_format:
        body["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    request_snapshot = {
        "timestamp": time.time(),
        "url": url,
        "model": model,
        "provider": TEXT_PROVIDER_GROK,
        "thinking_mode": mode,
        "temperature": effective_temperature,
        "record_usage": record_usage,
        "headers": _sanitize_request_headers(headers),
        "body": body,
    }
    STATE.last_text_request = request_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_request = copy.deepcopy(request_snapshot)

    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
    elapsed = time.perf_counter() - start_time
    if record_usage:
        STATE.last_turn_runtime = elapsed

    raw_text = getattr(r, "text", "")
    if callable(raw_text):
        try:
            raw_text = raw_text()
        except Exception:
            raw_text = ""
    if raw_text is None:
        raw_text = ""
    try:
        response_json = r.json()
    except Exception:
        response_json = None

    headers_source = getattr(r, "headers", {}) or {}
    header_items = headers_source.items() if hasattr(headers_source, "items") else []
    response_headers = {str(k): str(v) for k, v in header_items}
    response_snapshot = {
        "timestamp": time.time(),
        "status_code": r.status_code,
        "elapsed_seconds": elapsed,
        "headers": response_headers,
        "json": response_json,
        "text": raw_text,
        "provider": TEXT_PROVIDER_GROK,
    }
    STATE.last_text_response = response_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_response = copy.deepcopy(response_snapshot)

    if r.status_code != 200:
        detail = raw_text or "Text generation failed."
        raise HTTPException(status_code=502, detail=detail)

    if not isinstance(response_json, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    usage_meta = response_json.get("usage") or {}
    prompt_tokens = usage_meta.get("prompt_tokens")
    completion_tokens = usage_meta.get("completion_tokens")
    reasoning_tokens = None

    def _coerce_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    prompt_tokens = _coerce_int(prompt_tokens)
    completion_tokens = _coerce_int(completion_tokens)

    completion_details = usage_meta.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        for key in ("reasoning_tokens", "reasoning", "thinking_tokens"):
            candidate = _coerce_int(completion_details.get(key))
            if candidate is not None:
                reasoning_tokens = candidate
                break

    if record_usage:
        STATE.last_token_usage = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "thinking": reasoning_tokens,
        }
        for key in ("input", "output", "thinking"):
            val = STATE.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                STATE.session_token_usage[key] = STATE.session_token_usage.get(key, 0) + val
        STATE.session_request_count += 1

        combined_output_tokens = 0
        for val in (completion_tokens, reasoning_tokens):
            if isinstance(val, int) and val > 0:
                combined_output_tokens += val
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens or None)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            STATE.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if isinstance(STATE.last_cost_usd, float):
                STATE.session_cost_usd += STATE.last_cost_usd
        else:
            STATE.last_cost_usd = None

    choices = response_json.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    first_choice = choices[0] if isinstance(choices, list) else None
    if not isinstance(first_choice, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else first_choice.get("message")
    content = None
    if isinstance(message, dict):
        content = message.get("content")
    elif isinstance(first_choice.get("content"), (str, list)):
        content = first_choice.get("content")

    if isinstance(content, list):
        fragments: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text_val = part.get("text") or part.get("value")
                if isinstance(text_val, str):
                    fragments.append(text_val)
            elif isinstance(part, str):
                fragments.append(part)
        content_text = "".join(fragments)
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = ""

    content_text = content_text.strip()
    if not content_text:
        raise HTTPException(status_code=502, detail="Empty response from model.")

    def _clean_json_text(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`\n")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        return cleaned

    try:
        parsed = json.loads(content_text)
    except Exception:
        try:
            parsed = json.loads(_clean_json_text(content_text))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Malformed response from model: {exc}") from exc

    return parsed

def _normalize_openai_model_name(model: str) -> str:
    normalized = (model or "").strip().lower()
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def _openai_requires_reasoning_header(model: str) -> bool:
    normalized = _normalize_openai_model_name(model)
    prefixes = ("o1", "o3", "o4")
    if any(normalized.startswith(prefix) for prefix in prefixes):
        return True
    if "gpt-4.1" in normalized or normalized.startswith("gpt4.1"):
        return True
    return False


def _openai_supports_temperature(model: str) -> bool:
    normalized = _normalize_openai_model_name(model)
    if normalized.startswith("gpt-5"):
        return False
    return True


def _openai_reasoning_for_mode(model: str, mode: str) -> Optional[Dict[str, str]]:
    normalized = _normalize_openai_model_name(model)
    if not (normalized.startswith("gpt-5") or normalized.startswith("o1") or normalized.startswith("o3")):
        return None
    effort_map = {
        "none": "minimal",
        "brief": "low",
        "balanced": "medium",
        "deep": "high",
    }
    effort = effort_map.get(mode)
    if not effort:
        return None
    return {"effort": effort}


async def _openai_generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict,
    schema: Dict,
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,  # ignored for OpenAI but kept for signature parity
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict:
    api_key = require_text_api_key(TEXT_PROVIDER_OPENAI)
    url = OPENAI_RESPONSES_URL
    mode = (STATE.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    effective_temperature = temperature
    if effective_temperature is None:
        effective_temperature = {
            "none": 0.6,
            "brief": 0.75,
            "balanced": 0.9,
            "deep": 1.0,
        }.get(mode, 0.9)

    json_schema = _convert_schema_for_jsonschema(schema) if schema else None
    if json_schema and isinstance(json_schema, dict) and json_schema.get("type"):
        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name or "payload",
                "schema": json_schema,
            },
        }
    else:
        response_format = {"type": "json_object"}

    system_prompt_augmented = system_prompt.rstrip() + (
        "\n\nRespond strictly with a single JSON object matching the schema. "
        "Do not include markdown or any explanatory text."
    )
    payload_text = json.dumps(user_payload, ensure_ascii=False)

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt_augmented}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Use the following JSON input to determine the response. "
                            "Echo nothing from it except as needed in the output.\n" + payload_text
                        ),
                    }
                ],
            },
        ],
    }
    if effective_temperature is not None and _openai_supports_temperature(model):
        body["temperature"] = effective_temperature
    reasoning_cfg = None
    if include_thinking_budget:
        reasoning_cfg = _openai_reasoning_for_mode(model, mode)
    if reasoning_cfg:
        body["reasoning"] = reasoning_cfg
    if response_format.get("type") == "json_schema":
        json_schema_cfg = response_format.get("json_schema")
        text_format: Dict[str, Any] = {"type": "json_schema"}
        if isinstance(json_schema_cfg, dict):
            text_format.update(json_schema_cfg)
        else:
            text_format.update({
                "name": schema_name or "payload",
                "schema": json_schema or {},
            })
    else:
        text_format = response_format

    body["text"] = {"format": text_format}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if _openai_requires_reasoning_header(model):
        headers["OpenAI-Beta"] = "reasoning"

    request_snapshot = {
        "timestamp": time.time(),
        "url": url,
        "model": model,
        "provider": TEXT_PROVIDER_OPENAI,
        "thinking_mode": mode,
        "temperature": effective_temperature,
        "record_usage": record_usage,
        "headers": _sanitize_request_headers(headers),
        "body": body,
    }
    STATE.last_text_request = request_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_request = copy.deepcopy(request_snapshot)

    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
    elapsed = time.perf_counter() - start_time
    if record_usage:
        STATE.last_turn_runtime = elapsed

    raw_text = getattr(r, "text", "")
    if callable(raw_text):
        try:
            raw_text = raw_text()
        except Exception:
            raw_text = ""
    if raw_text is None:
        raw_text = ""
    try:
        response_json = r.json()
    except Exception:
        response_json = None

    headers_source = getattr(r, "headers", {}) or {}
    header_items = headers_source.items() if hasattr(headers_source, "items") else []
    response_headers = {str(k): str(v) for k, v in header_items}
    response_snapshot = {
        "timestamp": time.time(),
        "status_code": r.status_code,
        "elapsed_seconds": elapsed,
        "headers": response_headers,
        "json": response_json,
        "text": raw_text,
        "provider": TEXT_PROVIDER_OPENAI,
    }
    STATE.last_text_response = response_snapshot
    if dev_snapshot == "scenario":
        STATE.last_scenario_response = copy.deepcopy(response_snapshot)

    if r.status_code != 200:
        detail = raw_text or "Text generation failed."
        raise HTTPException(status_code=502, detail=detail)

    if not isinstance(response_json, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    usage_meta = response_json.get("usage") or {}
    if record_usage:
        reasoning_tokens = (
            usage_meta.get("reasoning_tokens")
            or usage_meta.get("reasoning_tokens_total")
            or usage_meta.get("reasoning_tokens_used")
        )
        if not isinstance(reasoning_tokens, int):
            details = usage_meta.get("output_tokens_details")
            if isinstance(details, dict):
                for key in ("reasoning_tokens", "reasoning", "thinking_tokens"):
                    value = details.get(key)
                    if isinstance(value, int):
                        reasoning_tokens = value
                        break
        STATE.last_token_usage = {
            "input": usage_meta.get("input_tokens"),
            "output": usage_meta.get("output_tokens"),
            "thinking": reasoning_tokens if isinstance(reasoning_tokens, int) else None,
        }
        for key in ("input", "output", "thinking"):
            val = STATE.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                STATE.session_token_usage[key] = STATE.session_token_usage.get(key, 0) + val
        STATE.session_request_count += 1

        prompt_tokens = STATE.last_token_usage.get("input")
        output_tokens = STATE.last_token_usage.get("output")
        thinking_tokens = STATE.last_token_usage.get("thinking")
        combined_output_tokens = sum(
            val for val in [output_tokens, thinking_tokens] if isinstance(val, int) and val > 0
        )
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            STATE.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if STATE.last_cost_usd is not None:
                STATE.session_cost_usd += STATE.last_cost_usd
        else:
            STATE.last_cost_usd = None

    output_entries: List[Any] = []
    raw_output = response_json.get("output")
    if isinstance(raw_output, list):
        output_entries = raw_output
    elif isinstance(raw_output, dict):
        output_entries = [raw_output]
    elif isinstance(response_json.get("data"), list):
        output_entries = response_json["data"]

    text_parts: List[str] = []

    def _collect_text(entry: Any):
        if isinstance(entry, dict):
            entry_type = entry.get("type")
            if entry_type == "message":
                for content in entry.get("content") or []:
                    _collect_text(content)
            elif entry_type in {"output_text", "text"}:
                text_val = entry.get("text") or entry.get("value")
                if text_val:
                    text_parts.append(str(text_val))
            elif "text" in entry and isinstance(entry.get("text"), str):
                text_parts.append(str(entry["text"]))
        elif isinstance(entry, str):
            text_parts.append(entry)

    for entry in output_entries:
        _collect_text(entry)

    if not text_parts:
        fallback_text = response_json.get("output_text") or response_json.get("text")
        if isinstance(fallback_text, str) and fallback_text.strip():
            text_parts.append(fallback_text)

    assembled = "".join(text_parts)
    if not assembled.strip():
        raise HTTPException(status_code=502, detail="Empty response from model.")

    try:
        parsed = json.loads(assembled)
    except Exception:
        cleaned = assembled.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`\n")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Malformed response from model: {exc}") from exc
    return parsed


async def _generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict,
    schema: Dict,
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict:
    provider = detect_text_provider(model)
    if provider == TEXT_PROVIDER_GROK:
        return await _grok_generate_structured(
            model=model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            schema=schema,
            temperature=temperature,
            record_usage=record_usage,
            include_thinking_budget=include_thinking_budget,
            dev_snapshot=dev_snapshot,
            schema_name=schema_name,
        )
    if provider == TEXT_PROVIDER_OPENAI:
        return await _openai_generate_structured(
            model=model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            schema=schema,
            temperature=temperature,
            record_usage=record_usage,
            include_thinking_budget=include_thinking_budget,
            dev_snapshot=dev_snapshot,
            schema_name=schema_name,
        )
    return await _gemini_generate_structured(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        temperature=temperature,
        record_usage=record_usage,
        include_thinking_budget=include_thinking_budget,
        dev_snapshot=dev_snapshot,
        schema_name=schema_name,
    )


async def gemini_generate_json(model: str, system_prompt: str, user_payload: Dict, schema: Dict) -> TurnStructured:
    """Calls generateContent with forced JSON schema; returns parsed TurnStructured."""
    parsed = await _generate_structured(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        temperature=None,
        record_usage=True,
        include_thinking_budget=True,
        dev_snapshot="scenario",
        schema_name="turn_payload",
    )
    try:
        return TurnStructured.model_validate(parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Schema validation error: {e}")


async def gemini_generate_summary(model: str, system_prompt: str, user_payload: Dict, schema: Dict) -> SummaryStructured:
    """Helper for generating concise history summaries without mutating token metrics."""
    parsed = await _generate_structured(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        temperature=0.35,
        record_usage=False,
        include_thinking_budget=False,
        dev_snapshot="summary",
        schema_name="summary_payload",
    )
    try:
        return SummaryStructured.model_validate(parsed)
    except Exception as exc:
        raise ValueError(f"Summary schema validation error: {exc}") from exc


def _parse_data_url(data_url: str) -> Optional[Tuple[str, str]]:
    """Return (mime_type, base64_data) extracted from a data URL."""
    if not isinstance(data_url, str):
        return None
    data_url = data_url.strip()
    if not data_url or not data_url.startswith("data:"):
        return None
    header, sep, b64_data = data_url.partition(",")
    if not sep or not b64_data:
        return None
    header = header.strip()
    if ";" in header:
        header = header.split(";", 1)[0]
    mime = "application/octet-stream"
    if header.startswith("data:"):
        candidate = header[5:].strip()
        if candidate:
            mime = candidate
    cleaned_data = "".join(b64_data.split())
    if not cleaned_data:
        return None
    return mime or "application/octet-stream", cleaned_data


def _build_scene_image_parts(prompt: str) -> Optional[List[Dict[str, Any]]]:
    """Assemble user parts for scene images, including portrait references when available."""
    references: List[Tuple[Player, str, str]] = []
    for player_id in sorted(STATE.players.keys()):
        player = STATE.players[player_id]
        portrait = player.portrait
        if not portrait or not portrait.data_url:
            continue
        parsed = _parse_data_url(portrait.data_url)
        if not parsed:
            continue
        mime, data = parsed
        references.append((player, mime, data))
    if not references:
        return None
    parts: List[Dict[str, Any]] = []
    for player, mime, data in references:
        parts.append({"inlineData": {"mimeType": mime, "data": data}})
    directive_lines: List[str] = [
        "Use the provided player portraits to keep each adventurer's appearance consistent across this scene.",
        "Match faces, hair, and distinctive accessories from the references even if lighting or outfits change.",
    ]
    for player, _, _ in references:
        descriptors: List[str] = []
        cls = (player.cls or "").strip()
        background = (player.background or "").strip()
        if cls:
            descriptors.append(cls)
        if background and background.lower() != cls.lower():
            descriptors.append(background)
        status = (player.status_word or "").strip().lower()
        if status and status not in {"", "unknown"}:
            descriptors.append(f"mood: {status}")
        descriptor_text = ", ".join(descriptors)
        if descriptor_text:
            directive_lines.append(f"{player.name}: match the portrait reference ({descriptor_text}).")
        else:
            directive_lines.append(f"{player.name}: match the portrait reference.")
    prompt_text = (prompt or "").strip()
    if prompt_text:
        directive_lines.extend(["", f"Scene prompt: {prompt_text}"])
    else:
        directive_lines.extend(["", "Scene prompt: (none provided)"])
    parts.append({"text": "\n".join(directive_lines)})
    return parts


async def gemini_generate_image(model: str, prompt: str, *, purpose: str = "scene") -> str:
    """Returns a data URL (base64 image) from gemini-2.5-flash-image-preview."""
    api_key = check_api_key()
    url = GENERATE_CONTENT_URL.format(model=model)
    prompt_text = prompt if isinstance(prompt, str) else str(prompt)
    request_parts: Optional[List[Dict[str, Any]]] = None
    if purpose == "scene":
        request_parts = _build_scene_image_parts(prompt_text)
    if not request_parts:
        request_parts = [{"text": prompt_text}]
    body = {
        "contents": [
            {"role": "user", "parts": request_parts}
        ]
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Image generation failed: {r.text}")
        data = r.json()

    # Extract first inline image part
    response_parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for prt in response_parts:
        inline = prt.get("inlineData") or prt.get("inline_data")
        if inline and inline.get("data"):
            b64 = inline["data"]
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            record_image_usage(model, purpose=purpose, images=1)
            return f"data:{mime};base64,{b64}"
    # Some generations also include text; if no image found, raise
    raise HTTPException(status_code=502, detail="No image data returned by model.")


# -------------------- Turn engine --------------------
def build_turn_schema() -> Dict:
    # Keep schema compact (avoid 400 for complexity); mirrors TurnStructured
    return {
        "type": "OBJECT",
        "properties": {
            "nar": {"type": "STRING"},
            "img": {"type": "STRING"},
            "pub": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "pid": {"type": "STRING"},
                        "word": {"type": "STRING"},
                    },
                    "required": ["pid", "word"],
                    "propertyOrdering": ["pid", "word"],
                },
            },
            "upd": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "pid": {"type": "STRING"},
                        "cls": {"type": "STRING"},
                        "ab": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "n": {"type": "STRING"},
                                    "x": {"type": "STRING"},
                                },
                                "required": ["n", "x"],
                                "propertyOrdering": ["n", "x"],
                            },
                        },
                        "inv": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "cond": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["pid", "cls", "ab", "inv", "cond"],
                    "propertyOrdering": ["pid", "cls", "ab", "inv", "cond"],
                },
            },
        },
        "required": ["nar", "img", "pub", "upd"],
        "propertyOrdering": ["nar", "img", "pub", "upd"],
    }


def build_thinking_directive() -> str:
    mode = (STATE.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"
    lang = STATE.language if STATE.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    lang_map = THINKING_DIRECTIVES_TEXT.get(lang) or THINKING_DIRECTIVES_TEXT[DEFAULT_LANGUAGE]
    return lang_map.get(mode) or lang_map["none"]


def compute_thinking_budget(model: str, mode: str) -> Optional[int]:
    if detect_text_provider(model) != TEXT_PROVIDER_GEMINI:
        return None
    normalized = (model or "").strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    model_lc = normalized
    mode = mode if mode in THINKING_MODES else "none"

    if "2.5" not in model_lc:
        return None

    if "flash-lite" in model_lc:
        mapping = {"none": 0, "brief": 512, "balanced": 2048, "deep": -1}
    elif "flash" in model_lc:
        mapping = {"none": 0, "brief": 1024, "balanced": 4096, "deep": -1}
    elif "pro" in model_lc:
        mapping = {"none": 128, "brief": 1024, "balanced": 4096, "deep": -1}
    else:
        return None

    return mapping.get(mode)


def make_gm_instruction(is_initial: bool) -> str:
    lang = STATE.language if STATE.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    template = GM_PROMPT_TEMPLATE if lang == DEFAULT_LANGUAGE else load_gm_prompt(lang)
    language_rule = LANGUAGE_RULES.get(lang) or LANGUAGE_RULES[DEFAULT_LANGUAGE]
    turn_rule_map = TURN_DIRECTIVES_INITIAL if is_initial else TURN_DIRECTIVES_ONGOING
    turn_rule = turn_rule_map.get(lang) or turn_rule_map[DEFAULT_LANGUAGE]
    directive_parts = [language_rule, turn_rule, build_thinking_directive()]
    directive = "".join(directive_parts)

    if TURN_DIRECTIVE_TOKEN in template:
        return template.replace(TURN_DIRECTIVE_TOKEN, directive)

    base = template.rstrip("\n")
    separator = "\n" if base else ""
    return f"{base}{separator}{directive}"


_SUMMARY_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _ensure_summary_bullet(text: str) -> str:
    cleaned = sanitize_narrative(text).strip()
    if not cleaned:
        cleaned = "No events recorded."
    if len(cleaned) > MAX_HISTORY_SUMMARY_CHARS:
        cleaned = cleaned[: MAX_HISTORY_SUMMARY_CHARS - 3].rstrip(" ,;:-") + "..."
    if cleaned.startswith("- "):
        return cleaned
    if cleaned.startswith("* "):
        cleaned = cleaned[2:].lstrip()
    if cleaned.startswith("-"):
        return cleaned
    return f"- {cleaned}"


def fallback_summarize_turn(rec: TurnRecord) -> str:
    text = sanitize_narrative(rec.narrative)
    if text:
        parts = _SUMMARY_SENTENCE_SPLIT.split(text, 1)
        snippet = parts[0].strip() if parts else text.strip()
        if not snippet:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            snippet = lines[0] if lines else "No narrative recorded."
    else:
        snippet = "No narrative recorded."
    line = f"Turn {rec.index}: {snippet}"
    return _ensure_summary_bullet(line)


def _fallback_summary_from_history() -> List[str]:
    if not STATE.history:
        return []
    recent = STATE.history[-MAX_HISTORY_SUMMARY_BULLETS:]
    return [fallback_summarize_turn(rec) for rec in recent]


async def update_history_summary(latest_turn: TurnRecord) -> None:
    """Refresh the cached bullet summary after each processed turn."""
    summary_before = list(STATE.history_summary[-MAX_HISTORY_SUMMARY_BULLETS:])

    history_mode = normalize_history_mode(STATE.settings.get("history_mode"))
    if history_mode != HISTORY_MODE_SUMMARY:
        STATE.history_summary = _fallback_summary_from_history()
        return

    api_key_present = bool((STATE.settings.get("api_key") or "").strip())
    if not api_key_present:
        STATE.history_summary = _fallback_summary_from_history()
        return

    players_snapshot = []
    for player in STATE.players.values():
        players_snapshot.append(
            {
                "id": player.id,
                "name": player.name,
                "class": player.cls,
                "status_word": player.status_word,
                "pending_join": player.pending_join,
                "pending_leave": player.pending_leave,
                "conditions": list(player.conditions),
                "inventory": list(player.inventory),
            }
        )

    payload = {
        "world_style": STATE.settings.get("world_style", "High fantasy"),
        "difficulty": STATE.settings.get("difficulty", "Normal"),
        "language": STATE.language,
        "turn_index": STATE.turn_index,
        "latest_turn": {
            "turn": latest_turn.index,
            "narrative": latest_turn.narrative,
            "image_prompt": latest_turn.image_prompt,
        },
        "previous_summary": summary_before,
        "players": players_snapshot,
        "max_bullets": MAX_HISTORY_SUMMARY_BULLETS,
    }

    model = STATE.settings.get("text_model") or DEFAULT_SETTINGS["text_model"]

    try:
        result = await gemini_generate_summary(
            model=model,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_payload=payload,
            schema=SUMMARY_SCHEMA,
        )
        lines: List[str] = []
        for item in result.summary:
            if not isinstance(item, str):
                continue
            ensured = _ensure_summary_bullet(item)
            if ensured:
                lines.append(ensured)
            if len(lines) >= MAX_HISTORY_SUMMARY_BULLETS:
                break
        if not lines:
            lines = _fallback_summary_from_history()
        STATE.history_summary = lines[:MAX_HISTORY_SUMMARY_BULLETS]
    except Exception as exc:  # noqa: BLE001
        print(f"History summary update failed: {exc}", file=sys.stderr, flush=True)
        STATE.history_summary = _fallback_summary_from_history()


def compile_user_payload() -> Dict:
    history_mode = normalize_history_mode(STATE.settings.get("history_mode"))
    use_summary = history_mode == HISTORY_MODE_SUMMARY

    full_history = [
        {
            "turn": rec.index,
            "narrative": rec.narrative,
            "image_prompt": rec.image_prompt,
        }
        for rec in STATE.history
    ]

    summary_lines = list(STATE.history_summary)

    if use_summary:
        if not summary_lines and full_history:
            summary_lines = [
                fallback_summarize_turn(rec)
                for rec in STATE.history[-MAX_HISTORY_SUMMARY_BULLETS:]
            ]
        history_payload: Any = summary_lines
    else:
        history_payload = full_history

    players = {
        pid: {
            "name": p.name,
            "background": p.background,
            "cls": p.cls,
            # Use Pydantic's serializer so we don't crash once abilities are populated.
            "ab": [
                a.model_dump() if isinstance(a, Ability) else Ability.model_validate(a).model_dump()
                for a in p.abilities
            ],
            "inv": p.inventory,
            "cond": p.conditions,
            "status_word": p.status_word,
            "pending_join": p.pending_join,
            "pending_leave": p.pending_leave,
        }
        for pid, p in STATE.players.items()
    }

    lang = STATE.language if STATE.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    payload = {
        "world_style": STATE.settings.get("world_style", "High fantasy"),
        "difficulty": STATE.settings.get("difficulty", "Normal"),
        "turn_index": STATE.turn_index,
        "history": history_payload,
        "history_mode": history_mode,
        "history_summary": summary_lines,
        "players": players,
        "submissions": STATE.submissions,  # {player_id: "action text"}
        "language": lang,
        "note": USER_PAYLOAD_NOTES.get(lang, USER_PAYLOAD_NOTES[DEFAULT_LANGUAGE]),
    }
    return payload


def build_portrait_prompt(player: Player) -> str:
    """Create a deterministic prompt for a player's portrait using the configured world style."""
    descriptors: List[str] = []
    cls = player.cls.strip() if player.cls else ""
    background = player.background.strip() if player.background else ""

    if cls:
        descriptors.append(cls)
    if background and background.lower() != cls.lower():
        descriptors.append(background)

    world_style = STATE.settings.get("world_style", "High fantasy")
    tone_bits: List[str] = []
    status_word = player.status_word.strip().lower() if player.status_word else ""
    if status_word and status_word not in {"unknown", ""}:
        tone_bits.append(f"mood of {status_word}")
    if player.pending_join:
        tone_bits.append("freshly joined hero")

    tone_text = ", ".join(tone_bits)
    descriptor_text = ", ".join(descriptors) if descriptors else "adventurer"

    prompt = (
        f"Highly detailed bust portrait of {player.name}, a {descriptor_text} from a {world_style} setting. "
        "Centered head and shoulders, cohesive lighting, clean backdrop, ready for use as a small game avatar. "
        "Painterly concept art finish, crisp readable details, square format."
    )

    if tone_text:
        prompt += f" Convey a {tone_text}."

    return prompt


async def resolve_turn(initial: bool = False):
    pending_before: Set[str] = set()
    leaving_before: Set[str] = set()
    async with STATE_LOCK:
        if STATE.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        STATE.lock = LockState(True, "resolving_turn")
        await broadcast_public()
        pending_before = {pid for pid, p in STATE.players.items() if p.pending_join}
        leaving_before = {pid for pid, p in STATE.players.items() if p.pending_leave}
        STATE.last_token_usage = {}
        STATE.last_turn_runtime = None
        STATE.last_cost_usd = None

    try:
        # Build prompt + schema and call Gemini once
        schema = build_turn_schema()
        system_text = make_gm_instruction(is_initial=initial)
        payload = compile_user_payload()
        model = STATE.settings.get("text_model") or "gemini-2.5-flash"

        result: TurnStructured = await gemini_generate_json(
            model=model,
            system_prompt=system_text,
            user_payload=payload,
            schema=schema
        )

        # Apply updates
        for upd in result.upd:
            pid = upd.pid
            if pid not in STATE.players:
                # Ignore unknown ids; model must stick to provided ids
                continue
            p = STATE.players[pid]
            p.cls = upd.cls
            p.abilities = [Ability(**a.model_dump()) if isinstance(a, Ability) else Ability(**a) for a in upd.ab]
            p.inventory = upd.inv
            p.conditions = upd.cond
            p.pending_join = False

        # Update public statuses
        for status in result.pub or []:
            pid = status.pid
            if pid in STATE.players:
                raw = (status.word or "").strip()
                first_word = raw.split()[0] if raw else "unknown"
                STATE.players[pid].status_word = first_word.lower()

        # Commit scenario + history
        narrative_text = sanitize_narrative(result.nar)
        image_prompt = result.img
        current_turn_index = STATE.turn_index
        STATE.current_scenario = narrative_text
        STATE.last_image_prompt = image_prompt
        STATE.turn_image_kind_counts = {}

        await maybe_queue_tts(narrative_text or "", current_turn_index)

        rec = TurnRecord(
            index=current_turn_index,
            narrative=narrative_text,
            image_prompt=image_prompt,
            timestamp=time.time(),
        )
        STATE.history.append(rec)

        await update_history_summary(rec)

        # Clear submissions for next turn
        STATE.submissions.clear()

        # Remove players who were marked for departure before the turn and
        # remain pending_leave after narrative send-off.
        departed_now: List[str] = []
        for pid in list(STATE.players.keys()):
            player = STATE.players[pid]
            if player.pending_leave and pid in leaving_before:
                departed_now.append(pid)
                STATE.players.pop(pid, None)

        reset_occurred = maybe_reset_session_if_empty()

        joined_now: List[str] = []
        if not reset_occurred:
            # Turn advances AFTER applying
            STATE.turn_index += 1

            # Inform everyone about joiners
            joined_now = [
                pid
                for pid in pending_before
                if pid in STATE.players and not STATE.players[pid].pending_join
            ]
            if not initial and joined_now:
                lang = STATE.language if STATE.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
                message = ANNOUNCEMENTS["new_player"].get(lang) or ANNOUNCEMENTS["new_player"][DEFAULT_LANGUAGE]
                await announce(message)

        # Clean up any lingering submissions for removed players (if turn advanced before clear)
        for pid in departed_now:
            STATE.submissions.pop(pid, None)

        # Push updated states
        await broadcast_public()
        # Private slices
        if not reset_occurred:
            for pid in list(STATE.players.keys()):
                await send_private(pid)
            await maybe_queue_scene_image(image_prompt, STATE.turn_index, force=False)

    finally:
        STATE.lock = LockState(False, "")
        await broadcast_public()


# -------------------- FastAPI app --------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.settings = load_settings()
    ensure_settings_file()
    STATE.language = normalize_language(STATE.settings.get("language"))
    yield


app = FastAPI(title="Nils' RPG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
if RESOURCES_DIR.exists():
    app.mount("/resources", StaticFiles(directory=str(RESOURCES_DIR)), name="resources")


# --------- Static root ---------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(APP_DIR / "static" / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if not FAVICON_FILE.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(str(FAVICON_FILE))


# --------- Settings ---------
@app.get("/api/settings")
async def get_settings():
    return STATE.settings.copy()


@app.get("/api/public_url")
async def get_public_url(request: Request):
    base = httpx.URL(str(request.base_url))
    scheme = base.scheme or "http"
    host = base.host or "localhost"
    port = base.port

    def format_host(host_value: str) -> str:
        if ":" in host_value and not host_value.startswith("["):
            return f"[{host_value}]"
        return host_value

    def build_url(host_value: str) -> str:
        host_display = format_host(host_value)
        if port in (None, 0):
            return f"{scheme}://{host_display}/"
        if scheme == "http" and port == 80:
            return f"{scheme}://{host_display}/"
        if scheme == "https" and port == 443:
            return f"{scheme}://{host_display}/"
        return f"{scheme}://{host_display}:{port}/"

    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if host not in loopback_hosts:
        return {"url": build_url(host), "source": "inferred_host"}

    public_ip = await fetch_public_ip()
    if public_ip:
        return {"url": build_url(public_ip), "source": "public_ip"}

    placeholder = "<your-public-ip-or-domain>"
    return {"url": build_url(placeholder), "source": "placeholder"}


@app.get("/api/dev/text_inspect")
async def get_dev_text_inspect():
    async with STATE_LOCK:
        request_payload = copy.deepcopy(STATE.last_text_request)
        response_payload = copy.deepcopy(STATE.last_text_response)
        scenario_request = copy.deepcopy(STATE.last_scenario_request)
        scenario_response = copy.deepcopy(STATE.last_scenario_response)
        history_mode = normalize_history_mode(STATE.settings.get("history_mode"))
    return {
        "request": request_payload,
        "response": response_payload,
        "scenario_request": scenario_request,
        "scenario_response": scenario_response,
        "history_mode": history_mode,
    }


class SettingsUpdate(BaseModel):
    api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    world_style: Optional[str] = None
    difficulty: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    narration_model: Optional[str] = None
    thinking_mode: Optional[str] = None
    history_mode: Optional[str] = None


@app.put("/api/settings")
async def update_settings(body: SettingsUpdate):
    global _ELEVENLABS_API_KEY_WARNING_LOGGED
    changed = False
    for k in [
        "world_style",
        "difficulty",
        "text_model",
        "image_model",
        "narration_model",
        "thinking_mode",
        "history_mode",
    ]:
        v = getattr(body, k)
        if v is not None:
            if k == "thinking_mode":
                mode = str(v).strip().lower()
                if mode not in THINKING_MODES:
                    raise HTTPException(status_code=400, detail="Unsupported thinking_mode value.")
                STATE.settings[k] = mode
            elif k == "narration_model":
                model_val = str(v).strip()
                if not model_val:
                    continue
                STATE.settings[k] = model_val
            elif k == "history_mode":
                STATE.settings[k] = normalize_history_mode(v)
            else:
                STATE.settings[k] = v
            changed = True
    # API key is optional but if provided we save immediately
    if body.api_key is not None:
        STATE.settings["api_key"] = body.api_key.strip()
        changed = True
    if body.grok_api_key is not None:
        STATE.settings["grok_api_key"] = body.grok_api_key.strip()
        changed = True
    if body.openai_api_key is not None:
        STATE.settings["openai_api_key"] = body.openai_api_key.strip()
        changed = True
    if body.elevenlabs_api_key is not None:
        STATE.settings["elevenlabs_api_key"] = body.elevenlabs_api_key.strip()
        _ELEVENLABS_API_KEY_WARNING_LOGGED = False
        changed = True
    if changed:
        await save_settings(STATE.settings)
    return {"ok": True}


# --------- Models list ---------
@app.get("/api/models")
async def api_models():
    items: List[Dict[str, Any]] = []
    errors: List[HTTPException] = []

    gemini_key = (STATE.settings.get("api_key") or "").strip()
    if gemini_key:
        try:
            models = await gemini_list_models()
        except HTTPException as exc:  # propagate later if nothing else succeeds
            errors.append(exc)
        else:
            for m in models:
                items.append(
                    {
                        "name": m.get("name", ""),
                        "displayName": m.get("displayName") or m.get("description") or m.get("name", ""),
                        "supported": m.get("supportedGenerationMethods")
                        or m.get("supported_actions")
                        or [],
                        "provider": TEXT_PROVIDER_GEMINI,
                    }
                )

    grok_key = (STATE.settings.get("grok_api_key") or "").strip()
    if grok_key:
        try:
            grok_models = await grok_list_models(grok_key)
        except HTTPException as exc:
            errors.append(exc)
        else:
            for gm in grok_models:
                identifier = (
                    gm.get("id")
                    or gm.get("name")
                    or gm.get("model")
                    or gm.get("slug")
                    or ""
                )
                identifier = str(identifier).strip()
                if not identifier:
                    continue
                display = (
                    gm.get("display_name")
                    or gm.get("displayName")
                    or gm.get("description")
                    or gm.get("title")
                    or identifier
                )
                supported_raw = (
                    gm.get("capabilities")
                    or gm.get("modalities")
                    or gm.get("endpoints")
                    or gm.get("interfaces")
                    or []
                )
                if isinstance(supported_raw, dict):
                    supported = [str(k) for k, v in supported_raw.items() if v]
                elif isinstance(supported_raw, (list, tuple)):
                    supported = [str(x) for x in supported_raw]
                elif supported_raw:
                    supported = [str(supported_raw)]
                else:
                    supported = []
                if not any("chat" in x.lower() for x in supported):
                    supported.append("chat.completions")
                items.append(
                    {
                        "name": identifier,
                        "displayName": str(display),
                        "supported": supported,
                        "provider": TEXT_PROVIDER_GROK,
                    }
                )

    openai_key = (STATE.settings.get("openai_api_key") or "").strip()
    if openai_key:
        try:
            openai_models = await openai_list_models(openai_key)
        except HTTPException as exc:
            errors.append(exc)
        else:
            for om in openai_models:
                identifier = (
                    om.get("id")
                    or om.get("model")
                    or om.get("name")
                    or om.get("slug")
                    or ""
                )
                identifier = str(identifier).strip()
                if not identifier:
                    continue
                display = (
                    om.get("display_name")
                    or om.get("displayName")
                    or om.get("description")
                    or identifier
                )
                supported_raw = (
                    om.get("capabilities")
                    or om.get("modalities")
                    or om.get("interfaces")
                    or []
                )
                if isinstance(supported_raw, dict):
                    supported = [str(k) for k, v in supported_raw.items() if v]
                elif isinstance(supported_raw, (list, tuple)):
                    supported = [str(x) for x in supported_raw]
                elif supported_raw:
                    supported = [str(supported_raw)]
                else:
                    supported = []
                if not supported:
                    supported = ["responses"]
                items.append(
                    {
                        "name": identifier,
                        "displayName": str(display),
                        "supported": supported,
                        "provider": TEXT_PROVIDER_OPENAI,
                    }
                )

    if not items and errors:
        raise errors[0]

    items.sort(
        key=lambda entry: (
            str(entry.get("displayName") or entry.get("name") or "").lower(),
            str(entry.get("name") or "").lower(),
        )
    )

    narration_models: List[Dict[str, Any]] = []
    api_key = (STATE.settings.get("elevenlabs_api_key") or "").strip()
    if api_key:
        narration_models = await elevenlabs_list_models(api_key)
    return {"models": items, "narration_models": narration_models}


# --------- Join / Leave / State ---------
class LanguageBody(BaseModel):
    language: str
    player_id: Optional[str] = None
    token: Optional[str] = None


@app.post("/api/language")
async def set_language(body: LanguageBody):
    if body.player_id and body.token:
        authenticate_player(body.player_id, body.token)
    changed = apply_language_value(body.language)
    if changed:
        await save_settings(STATE.settings)
        await broadcast_public()
    return {"language": STATE.language}


class JoinBody(BaseModel):
    name: Optional[str] = "Hephaest"
    background: Optional[str] = "Wizard"
    language: Optional[str] = None


@app.post("/api/join")
async def join_game(body: JoinBody):
    cancel_pending_reset_task()
    if STATE.players:
        maybe_reset_session_if_empty()
    apply_language_value(body.language)
    pid = secrets.token_hex(8)
    name = (body.name or "Hephaest").strip()[:40]
    background = (body.background or "Wizard").strip()[:200]
    token = secrets.token_hex(16)
    p = Player(id=pid, name=name, background=background, pending_join=True, connected=True, token=token)
    STATE.players[pid] = p

    # If this is the very first player and world not started -> run initial world-gen immediately
    if STATE.turn_index == 0 and not STATE.current_scenario:
        try:
            await announce(f"{name} is starting a new world…")
            await resolve_turn(initial=True)
        except Exception:
            # Undo the player registration so a failed turn doesn't leave ghosts around.
            STATE.players.pop(pid, None)
            await broadcast_public()
            raise

    await broadcast_public()
    return {"player_id": pid, "auth_token": token}


@app.get("/api/state")
async def get_state():
    return STATE.public_snapshot()


class SubmitBody(BaseModel):
    player_id: str
    token: str
    text: str
    language: Optional[str] = None


@app.post("/api/submit")
async def submit_action(body: SubmitBody):
    if STATE.lock.active:
        raise HTTPException(status_code=409, detail="Game is busy. Try again in a moment.")
    player = authenticate_player(body.player_id, body.token)
    apply_language_value(body.language)
    STATE.submissions[player.id] = body.text.strip()[:1000]
    await broadcast_public()
    return {"ok": True}


class NextTurnBody(BaseModel):
    player_id: str
    token: str
    language: Optional[str] = None


@app.post("/api/next_turn")
async def next_turn(body: NextTurnBody):
    # Anyone can advance the turn per spec
    authenticate_player(body.player_id, body.token)
    apply_language_value(body.language)
    await resolve_turn(initial=False)
    return {"ok": True}


class ToggleTtsBody(BaseModel):
    player_id: str
    token: str
    enabled: bool


class ToggleSceneImageBody(BaseModel):
    player_id: str
    token: str
    enabled: bool


@app.post("/api/tts_toggle")
async def toggle_tts(body: ToggleTtsBody):
    authenticate_player(body.player_id, body.token)
    if body.enabled:
        api_key = (STATE.settings.get("elevenlabs_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Set the ElevenLabs API key in Settings before enabling narration.")
        if not _elevenlabs_library_available():
            raise HTTPException(status_code=500, detail="elevenlabs package is not installed on the server.")
    STATE.auto_tts_enabled = bool(body.enabled)
    await broadcast_public()
    if STATE.auto_tts_enabled:
        await maybe_queue_tts(STATE.current_scenario or "", STATE.turn_index)
    return {"auto_tts_enabled": STATE.auto_tts_enabled}


@app.post("/api/image_toggle")
async def toggle_scene_image(body: ToggleSceneImageBody):
    authenticate_player(body.player_id, body.token)
    STATE.auto_image_enabled = bool(body.enabled)
    await broadcast_public()
    if STATE.auto_image_enabled:
        await maybe_queue_scene_image(STATE.last_image_prompt, STATE.turn_index, force=True)
    return {"auto_image_enabled": STATE.auto_image_enabled}


class CreateImageBody(BaseModel):
    player_id: str
    token: str


@app.post("/api/create_image")
async def create_image(body: CreateImageBody):
    async with STATE_LOCK:
        if STATE.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        authenticate_player(body.player_id, body.token)
        # Need an image prompt from the latest turn
        if not STATE.last_image_prompt:
            raise HTTPException(status_code=400, detail="No image prompt available yet.")
        STATE.lock = LockState(True, "generating_image")
        await broadcast_public()

    try:
        img_model = STATE.settings.get("image_model") or "gemini-2.5-flash-image-preview"
        data_url = await gemini_generate_image(
            img_model,
            STATE.last_image_prompt,
            purpose="scene",
        )
        STATE.last_image_data_url = data_url
        await announce("Image generated.")
        await broadcast_public()
        return {"ok": True}
    finally:
        STATE.lock = LockState(False, "")
        await broadcast_public()


class CreatePortraitBody(BaseModel):
    player_id: str
    token: str


@app.post("/api/create_portrait")
async def create_portrait(body: CreatePortraitBody):
    async with STATE_LOCK:
        if STATE.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        player = authenticate_player(body.player_id, body.token)
        portrait_prompt = build_portrait_prompt(player)
        STATE.lock = LockState(True, "generating_portrait")
        await broadcast_public()

    try:
        img_model = STATE.settings.get("image_model") or "gemini-2.5-flash-image-preview"
        data_url = await gemini_generate_image(
            img_model,
            portrait_prompt,
            purpose="portrait",
        )
        player.portrait = PlayerPortrait(
            data_url=data_url,
            prompt=portrait_prompt,
            updated_at=time.time(),
        )
        await announce(f"{player.name}'s portrait updated.")
        await broadcast_public()
        await send_private(player.id)
        return {"ok": True, "portrait": portrait_payload(player.portrait)}
    finally:
        STATE.lock = LockState(False, "")
        await broadcast_public()


# --------- WebSockets ---------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Optional player_id supplied as query for private channel
    player_id = ws.query_params.get("player_id")
    auth_token = ws.query_params.get("auth_token")
    STATE.global_sockets.add(ws)
    authed_player = None
    if player_id and auth_token:
        candidate = STATE.players.get(player_id)
        if candidate and candidate.token == auth_token:
            authed_player = candidate
            authed_player.sockets.add(ws)
            authed_player.connected = True
            authed_player.pending_leave = False
            cancel_pending_reset_task()
    # Send initial snapshots
    await ws.send_json({"event": "state", "data": STATE.public_snapshot()})
    if authed_player:
        await ws.send_json({"event": "private", "data": STATE.private_snapshot_for(authed_player.id)})

    try:
        while True:
            # We don't need to receive anything; just keep connection alive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup
        STATE.global_sockets.discard(ws)
        leave_player: Optional[Player] = None
        if authed_player and authed_player.id in STATE.players:
            authed_player.sockets.discard(ws)
            if not authed_player.sockets:
                authed_player.connected = False
                if not authed_player.pending_leave:
                    authed_player.pending_leave = True
                    leave_player = authed_player
        await broadcast_public()
        if leave_player:
            await announce(f"{leave_player.name} has left the party.")
            schedule_session_reset_check()


if __name__ == "__main__":
    # Provide a convenient CLI entry point for local running.
    import logging
    import socket

    import uvicorn
    from uvicorn.config import Config
    from uvicorn.server import Server
    from uvicorn.supervisors import ChangeReload, Multiprocess

    def _create_listeners(port: int, backlog: int) -> list[socket.socket]:
        """Return sockets that cover IPv6 and IPv4 where possible."""

        def _mark_socket(sock: socket.socket) -> socket.socket:
            sock.set_inheritable(True)
            return sock

        errors: list[BaseException] = []

        # Prefer a single dual-stack socket when supported by the platform.
        try:
            sock = socket.create_server(
                ("::", port),
                family=socket.AF_INET6,
                backlog=backlog,
                reuse_port=False,
                dualstack_ipv6=True,
            )
        except (OSError, ValueError, AttributeError) as exc:
            errors.append(exc)
        else:
            return [_mark_socket(sock)]

        sockets: list[socket.socket] = []
        bind_port = port

        # Bind a dedicated IPv6 socket if possible.
        try:
            sock6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "IPV6_V6ONLY"):
                sock6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock6.bind(("::", bind_port))
            sock6.listen(backlog)
            sockets.append(_mark_socket(sock6))
            if port == 0:
                bind_port = sock6.getsockname()[1]
        except OSError as exc:
            errors.append(exc)

        # Fall back to IPv4 so localhost keeps working even if IPv6-only binding succeeds.
        try:
            sock4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock4.bind(("0.0.0.0", bind_port))
            sock4.listen(backlog)
            sockets.append(_mark_socket(sock4))
        except OSError as exc:
            errors.append(exc)

        if sockets:
            return sockets

        raise RuntimeError("Failed to bind IPv4 or IPv6 listener") from (errors[-1] if errors else None)

    def _log_bindings(sockets: list[socket.socket], is_ssl: bool) -> None:
        protocol = "https" if is_ssl else "http"
        logger = logging.getLogger("uvicorn.error")
        for sock in sockets:
            address = sock.getsockname()
            if sock.family == socket.AF_INET6:
                host = f"[{address[0]}]"
            else:
                host = address[0]
            logger.info("Uvicorn running on %s://%s:%d (Press CTRL+C to quit)", protocol, host, address[1])

    host_override = os.environ.get("ORPG_HOST")
    port = int(os.environ.get("ORPG_PORT", "8000"))
    reload_enabled = os.environ.get("ORPG_RELOAD") == "1"

    if host_override:
        uvicorn.run(
            "rpg:app",
            host=host_override,
            port=port,
            reload=reload_enabled,
        )
    else:
        config = Config(
            "rpg:app",
            host="::",
            port=port,
            reload=reload_enabled,
        )
        server = Server(config=config)

        sockets = _create_listeners(port=config.port, backlog=config.backlog)
        _log_bindings(sockets, config.is_ssl)

        if config.should_reload:
            ChangeReload(config, target=server.run, sockets=sockets).run()
        elif config.workers > 1:
            Multiprocess(config, target=server.run, sockets=sockets).run()
        else:
            try:
                server.run(sockets=sockets)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Allow graceful shutdown without a traceback when interrupted locally.
                pass
