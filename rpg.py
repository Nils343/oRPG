# rpg.py
# Multiplayer text RPG server using FastAPI + WebSockets
# - settings.json (same folder) stores API key, world style, difficulty, and model choices
# - Structured turn calls support Gemini, Grok, and OpenAI providers over REST (httpx)
# - Each turn makes one text-generation request; summary mode may add a follow-up structured call
# - Optional image generation per turn via gemini-2.5-flash-image-preview
# - __main__ entry-point wraps uvicorn with IPv4/IPv6 helpers for local hosting

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import secrets
import sys
import tempfile
import time
import subprocess
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    BinaryIO,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    cast,
)

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field

try:  # google-genai is optional at import time for environments without video support
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:  # pragma: no cover - handled at runtime
    _genai = None
    _genai_types = None

genai: Any = _genai
genai_types: Any = _genai_types

APP_DIR = Path(__file__).parent
SETTINGS_FILE = APP_DIR / "settings.json"
PROMPT_FILE = APP_DIR / "gm_prompt.txt"
PROMPT_FILES = {
    "en": PROMPT_FILE,
    "de": APP_DIR / "gm_prompt.de.txt",
}
_GM_PROMPT_CACHE: Dict[str, str] = {}
GM_PROMPT_TEMPLATE: Optional[str] = None
# Marker used in prompts; not a credential.
TURN_DIRECTIVE_TOKEN = "<<TURN_DIRECTIVE>>"  # nosec B105

GENERATED_MEDIA_DIR = APP_DIR / "generated_media"

WEBSOCKET_IDLE_TIMEOUT = 30.0

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[float, List[object]]] = {}
_MODEL_CACHE_LOCK = asyncio.Lock()


def _models_cache_key(provider: str, api_key: Optional[str]) -> Tuple[str, str]:
    """Return a stable cache key using a provider name and obfuscated API key."""
    key_material = (api_key or "").strip()
    if not key_material:
        return provider, ""
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return provider, digest


ModelT = TypeVar("ModelT")


async def _get_cached_models(
    provider: str,
    api_key: Optional[str],
    loader: Callable[[], Awaitable[List[ModelT]]],
) -> List[ModelT]:
    cache_key = _models_cache_key(provider, api_key)
    now = time.time()
    async with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < MODEL_CACHE_TTL_SECONDS:
            return copy.deepcopy(cast(List[ModelT], cached[1]))

    models = await loader()
    snapshot = copy.deepcopy(models)
    async with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (time.time(), cast(List[object], snapshot))
    return models

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {"en", "de"}

DEFAULT_TEXT_TEMPERATURE = 0.2

DEFAULT_VIDEO_MODEL = "veo-3.0-generate-001"
FRAMEPACK_MODEL_ID = "FramePack"
FRAMEPACK_STATIC_MODEL_ID = "Static FramePack"
FRAMEPACK_STATIC_END_INFLUENCE = 0.8
PARALLAX_MODEL_ID = "Parallax"
PARALLAX_DEFAULT_LAYERS = 4
FRAMEPACK_DEFAULT_DURATION_SECONDS = 8.0
FRAMEPACK_DEFAULT_VARIANT = "Original"
FRAMEPACK_MIN_DURATION_SECONDS = 1
FRAMEPACK_MAX_DURATION_SECONDS = 120

HISTORY_MODE_FULL = "full"
HISTORY_MODE_SUMMARY = "summary"
HISTORY_MODE_OPTIONS = {HISTORY_MODE_FULL, HISTORY_MODE_SUMMARY}
MAX_HISTORY_SUMMARY_BULLETS = 12
MAX_HISTORY_SUMMARY_CHARS = 280


class ProviderModelInfo(TypedDict, total=False):
    name: str
    displayName: str
    supported: List[str]
    provider: Literal["gemini", "grok", "openai"]
    category: Literal["text", "video", "other"]
    modelId: str
    family: str


class ElevenLabsModelInfo(TypedDict):
    id: str
    name: str
    languages: List[str]
    language_codes: List[str]


class ElevenLabsNarrationMetadata(TypedDict, total=False):
    model_id: Optional[str]
    voice_id: Optional[str]
    characters_reported: Optional[int]
    character_source: Optional[str]
    characters_final: Optional[int]
    estimated_credits: Optional[float]
    estimated_cost_usd: Optional[float]
    usd_per_million: Optional[float]
    credits_per_character: Optional[float]
    request_id: Optional[str]
    headers: Dict[str, str]
    subscription_total_credits: Optional[int]
    subscription_used_credits: Optional[int]
    subscription_remaining_credits: Optional[int]
    subscription_next_reset_unix: Optional[int]


class ElevenLabsNarrationResult(TypedDict):
    audio_base64: Optional[str]
    metadata: ElevenLabsNarrationMetadata


def _normalize_supported_list(raw: Any, *, fallback: Optional[Iterable[str]] = None) -> List[str]:
    result: List[str] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value:
                result.append(str(key))
    elif isinstance(raw, (list, tuple, set)):
        result.extend(str(item) for item in raw if item is not None)
    elif raw not in (None, ""):
        result.append(str(raw))

    if not result and fallback:
        result.extend(str(item) for item in fallback if item is not None)

    seen: Set[str] = set()
    unique: List[str] = []
    for entry in result:
        lowered = entry.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(entry)
    return unique


def _sanitize_media_prefix(prefix: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prefix.lower())
    safe = safe.strip("_")
    return safe or "media"


def _suffix_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return ".bin"
    normalized = mime.strip().lower()
    common = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if normalized in common:
        return common[normalized]
    if "/" in normalized:
        subtype = normalized.split("/", 1)[1].split(";", 1)[0]
        subtype = re.sub(r"[^a-z0-9.+-]", "", subtype)
        if subtype:
            return f".{subtype}"
    return ".bin"


def _unique_media_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    directory = path.parent
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _is_framepack_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return str(model).strip().lower() == FRAMEPACK_MODEL_ID.lower()


def _is_framepack_static_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return str(model).strip().lower() == FRAMEPACK_STATIC_MODEL_ID.lower()


def _is_parallax_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return str(model).strip().lower() == PARALLAX_MODEL_ID.lower()


def _load_framepack_module():
    try:
        return importlib.import_module("Animate.animate_framepack")
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise HTTPException(
            status_code=500,
            detail=(
                "FramePack integration is unavailable because its dependencies are missing. "
                "Install the optional requirements to enable FramePack video generation."
            ),
        ) from exc


def _resolve_framepack_image(image_data_url: Optional[str]) -> Optional[Tuple[bytes, str]]:
    if not image_data_url:
        return None
    parsed = _parse_data_url(image_data_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="FramePack reference image data URL is invalid.")
    mime, b64_data = parsed
    image_bytes = _decode_base64_data(b64_data)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="FramePack reference image could not be decoded.")
    suffix = _suffix_from_mime(mime or "image/png") or ".png"
    if suffix == ".bin":
        suffix = ".png"
    return image_bytes, suffix


def _resolve_parallax_image(image_data_url: Optional[str]) -> Tuple[bytes, str]:
    if not image_data_url:
        raise HTTPException(status_code=400, detail="Parallax animation requires an existing image.")
    parsed = _parse_data_url(image_data_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Parallax reference image data URL is invalid.")
    mime, b64_data = parsed
    image_bytes = _decode_base64_data(b64_data)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Parallax reference image could not be decoded.")
    suffix = _suffix_from_mime(mime or "image/png") or ".png"
    if suffix == ".bin":
        suffix = ".png"
    return image_bytes, suffix


def _normalize_video_duration_seconds(raw: Any) -> int:
    if raw is None:
        return int(FRAMEPACK_DEFAULT_DURATION_SECONDS)
    if isinstance(raw, bool):  # guard against bool being subclass of int
        raise ValueError("Invalid duration value")
    if isinstance(raw, (int, float)):
        numeric = float(raw)
    elif isinstance(raw, str):
        candidate = raw.strip()
        if not candidate:
            return int(FRAMEPACK_DEFAULT_DURATION_SECONDS)
        try:
            numeric = float(candidate)
        except ValueError as exc:
            raise ValueError("Invalid duration value") from exc
    else:
        raise ValueError("Invalid duration value")
    if not math.isfinite(numeric):
        raise ValueError("Invalid duration value")
    rounded = int(round(numeric))
    if rounded < FRAMEPACK_MIN_DURATION_SECONDS:
        return FRAMEPACK_MIN_DURATION_SECONDS
    if rounded > FRAMEPACK_MAX_DURATION_SECONDS:
        return FRAMEPACK_MAX_DURATION_SECONDS
    return rounded


def _decode_base64_data(b64_data: str) -> Optional[bytes]:
    if not b64_data:
        return None
    try:
        return base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError):
        pass
    try:
        padding = "=" * (-len(b64_data) % 4)
        return base64.b64decode(b64_data + padding, validate=False)
    except (binascii.Error, ValueError):
        return None


def _archive_generated_media(content: bytes, *, prefix: str, suffix: str) -> Path:
    GENERATED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename = f"{_sanitize_media_prefix(prefix)}_{int(time.time())}_{secrets.token_hex(4)}{normalized_suffix}"
    output_path = GENERATED_MEDIA_DIR / filename
    output_path.write_bytes(content)
    return output_path


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


def normalize_language(lang: Any) -> str:
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


def _get_setting_str(
    settings: Mapping[str, Any],
    key: str,
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return a sanitized string from settings or *default* when absent."""

    value = settings.get(key)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned or cleaned == "":
            return cleaned
    return default


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
_ASCII_FALLBACKS = {
    "ß": "ss",
    "ẞ": "SS",
}


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
    for src, replacement in _ASCII_FALLBACKS.items():
        if src in cleaned:
            cleaned = cleaned.replace(src, replacement)
    return cleaned.strip()


def normalize_player_name(name: Optional[str]) -> str:
    """Return a canonical representation for comparing player names."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKC", str(name)).strip()
    return normalized.casefold()


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

    def __init__(self, message: str) -> None:
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


PUBLIC_IP_CACHE: Dict[str, float | str | None] = {
    "ip": None,
    "cached_at": 0.0,
    "last_failure_at": 0.0,
}
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


def get_default_gm_prompt() -> str:
    if GM_PROMPT_TEMPLATE is not None:
        return GM_PROMPT_TEMPLATE
    return load_gm_prompt(DEFAULT_LANGUAGE)


async def fetch_public_ip() -> Optional[str]:
    """Best-effort lookup of the host's public IP for sharing with remote players."""
    now = time.time()
    cached_value = PUBLIC_IP_CACHE.get("ip")
    cached = cached_value if isinstance(cached_value, str) and cached_value else None
    timestamp_raw = PUBLIC_IP_CACHE.get("cached_at", 0.0)
    timestamp = float(timestamp_raw) if isinstance(timestamp_raw, (int, float)) else 0.0
    if cached and (now - timestamp) < PUBLIC_IP_CACHE_TTL:
        return cached

    failure_raw = PUBLIC_IP_CACHE.get("last_failure_at", 0.0)
    last_failure = float(failure_raw) if isinstance(failure_raw, (int, float)) else 0.0
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
                except Exception:  # nosec B112
                    continue

                if not ip:
                    continue
                if ":" in ip:
                    preferred_ip = ip
                    break
                if fallback_ip is None:
                    fallback_ip = ip

            ip_candidate = preferred_ip or fallback_ip
            if isinstance(ip_candidate, str) and ip_candidate:
                PUBLIC_IP_CACHE["ip"] = ip_candidate
                PUBLIC_IP_CACHE["cached_at"] = now
                PUBLIC_IP_CACHE["last_failure_at"] = 0.0
                return ip_candidate
    except Exception:
        PUBLIC_IP_CACHE["last_failure_at"] = now
    return None


# -------- Defaults --------
DEFAULT_SETTINGS: Dict[str, str | int] = {
    "gemini_api_key": "",
    "grok_api_key": "",
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "text_model": "grok-4-fast-non-reasoning",
    "image_model": "gemini-2.5-flash-image-preview",
    "video_model": DEFAULT_VIDEO_MODEL,
    "video_duration_seconds": int(FRAMEPACK_DEFAULT_DURATION_SECONDS),
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
        "INITIAL TURN: Create the opening scenario AND, for each player where "
        "players[pid].pending_join == true, create their full character kit "
        "(cls/ab/inv/cond). Do NOT create characters for players without pending_join.\n"
    ),
    "de": (
        "ERSTE RUNDE: Erzeuge die Eröffnungsszene UND erstelle für jeden Spieler mit "
        "players[pid].pending_join == true das vollständige Charakterpaket (cls/ab/inv/cond). "
        "Erstelle KEINE Charaktere für Spieler ohne pending_join.\n"
    ),
}

TURN_DIRECTIVES_ONGOING = {
    "en": (
        "ONGOING TURN: Resolve all submissions. Naturally integrate any players with "
        "players[pid].pending_join == true, and provide narrative closure for any players "
        "with players[pid].pending_leave == true so their departure feels organic. Use only "
        "known player IDs and do not invent new players. Names listed in departed_players "
        "are gone for now—do not revive or reference them unless they return via pending_join == true.\n"
    ),
    "de": (
        "LAUFENDE RUNDE: Löse alle eingereichten Aktionen auf. Integriere Spieler mit "
        "players[pid].pending_join == true organisch in die Szene und gib Spielern mit "
        "players[pid].pending_leave == true einen erzählerischen Abschied. Verwende nur "
        "bekannte Spieler-IDs und erfinde keine neuen. Namen in departed_players gelten "
        "als ausgeschieden—bringe sie nur zurück, wenn sie erneut mit pending_join == true auftauchen.\n"
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
            ""
        ),
        "deep": (
            "THINKING MODE: Use thorough internal reasoning to maintain continuity and fairness. "
            "Keep the final response clean JSON + narrative without revealing your chain-of-thought.\n"
        ),
    },
    "de": {
        "none": (
            "DENKMODUS: Antworte entschlossen mit minimalem inneren Abwägen. "
            "Vermeide lange Planungen und gib niemals deine Gedankengänge preis; "
            "liefere ausschließlich die geforderten strukturierten Ausgaben.\n"
        ),
        "brief": (
            "DENKMODUS: Nimm dir kurz Zeit, um die Konsistenz zu prüfen, bevor du antwortest. "
            "Halte das Nachdenken knapp und gib keine verborgenen Überlegungen aus.\n"
        ),
        "balanced": (
            ""
        ),
        "deep": (
            "DENKMODUS: Nutze ausführliches inneres Nachdenken, um Kontinuität und Fairness zu bewahren. "
            "Die finale Antwort bleibt reines JSON plus Erzähltext ohne offengelegte Gedankengänge.\n"
        ),
    },
}

USER_PAYLOAD_NOTES = {
    "en": (
        "Update 'pub' with an entry for every current player each turn (word = single lowercase token, "
        "no hyphens/punctuation). For 'upd', include entries for all players where pending_join == true "
        "(full cls/ab/inv/cond). For existing players, include an 'upd' entry only when something changed "
        "this turn, and always send full lists, not diffs. Use only provided player IDs; unknown or "
        "invented IDs will be ignored. When pending_leave == true, provide narrative closure this turn and "
        "omit that player from future updates."
    ),
    "de": (
        "Aktualisiere 'pub' in jeder Runde mit einem Eintrag für alle aktuellen Spieler (word = einzelnes "
        "kleingeschriebenes Wort ohne Bindestriche oder Satzzeichen). Für 'upd' Einträge für alle Spieler mit "
        "pending_join == true aufnehmen (vollständige cls/ab/inv/cond). Für bestehende Spieler nur dann einen "
        "'upd'-Eintrag senden, wenn sich in dieser Runde etwas geändert hat, und stets vollständige Listen statt "
        "Deltas liefern. Verwende ausschließlich die bereitgestellten Spieler-IDs; unbekannte oder erfundene IDs "
        "werden ignoriert. Ist pending_leave == true, sorge in dieser Runde für einen erzählerischen Abschluss und "
        "lasse diesen Spieler in zukünftigen Updates weg."
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
VIDEO_MODEL_PRICES: Dict[str, Any] = (
    MODEL_PRICING_DATA.get("video", {}).get("models")
    if isinstance(MODEL_PRICING_DATA.get("video", {}).get("models"), dict)
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
    return None


def calculate_turn_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[Dict[str, Any]]:
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

    def _compute_cost(tokens: int, price: Optional[float]) -> Optional[float]:
        if tokens <= 0:
            return 0.0
        if price is None:
            return None
        return (tokens / 1_000_000) * float(price)

    prompt_cost = _compute_cost(prompt_tok, prompt_price)
    completion_cost = _compute_cost(completion_tok, completion_price)

    unknown_cost = (
        (prompt_tok > 0 and prompt_cost is None)
        or (completion_tok > 0 and completion_cost is None)
    )

    if unknown_cost:
        total: Optional[float] = None
    else:
        total = 0.0
        if isinstance(prompt_cost, (int, float)):
            total += float(prompt_cost)
        if isinstance(completion_cost, (int, float)):
            total += float(completion_cost)

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


def calculate_video_cost(
    model: Optional[str],
    *,
    seconds: Optional[float],
    tier: str = "default",
) -> Optional[Dict[str, Any]]:
    model_key = _normalize_model_name(model)
    pricing = VIDEO_MODEL_PRICES.get(model_key)
    if not isinstance(pricing, dict):
        return None

    tier_key = tier or "default"
    tier_info = pricing.get(tier_key)
    if not isinstance(tier_info, dict):
        tier_info = None
        for key, info in pricing.items():
            if isinstance(info, dict) and "usd_per_second" in info:
                tier_info = info
                tier_key = key
                break
    if not isinstance(tier_info, dict):
        return None

    usd_per_second = tier_info.get("usd_per_second")
    if not isinstance(usd_per_second, (int, float)):
        return None
    seconds_value = float(seconds) if isinstance(seconds, (int, float)) and seconds >= 0 else None
    cost_usd = None
    if seconds_value is not None:
        cost_usd = float(usd_per_second) * seconds_value

    return {
        "model": model_key,
        "tier": tier_key,
        "seconds": seconds_value,
        "usd_per_second": float(usd_per_second),
        "cost_usd": cost_usd,
    }


def _probe_mp4_duration_seconds(path: Path) -> Optional[float]:
    """Return media duration in seconds for a generated MP4, if readable."""

    try:
        with path.open('rb') as fh:
            return _read_mp4_duration_from_stream(fh)
    except OSError:
        return None



def _read_mp4_duration_from_stream(fh: BinaryIO) -> Optional[float]:
    import io
    import struct

    def _read_atom(stream: BinaryIO, size: int) -> bytes:
        data = stream.read(size)
        if len(data) != size:
            raise EOFError
        return data

    try:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return None
            atom_size, atom_type = struct.unpack('>I4s', header)
            if atom_size == 1:
                largesize_bytes = fh.read(8)
                if len(largesize_bytes) < 8:
                    return None
                atom_size = struct.unpack('>Q', largesize_bytes)[0]
                header_size = 16
            else:
                header_size = 8
            if atom_size < header_size:
                return None
            payload_size = atom_size - header_size
            if atom_type == b'moov':
                moov_data = _read_atom(fh, payload_size)
                return _extract_mvhd_duration(io.BytesIO(moov_data))
            fh.seek(payload_size, 1)
    except (OSError, EOFError, struct.error):
        return None



def _extract_mvhd_duration(stream: Any) -> Optional[float]:
    import struct

    data = stream.read()
    length = len(data)
    offset = 0
    while offset + 8 <= length:
        atom_size = struct.unpack('>I', data[offset:offset + 4])[0]
        atom_type = data[offset + 4:offset + 8]
        if atom_size == 0:
            atom_size = length - offset
        if atom_size < 8:
            return None
        if atom_type == b'mvhd':
            content_offset = offset + 8
            if content_offset >= length:
                return None
            version = data[content_offset]
            if version == 0:
                needed = content_offset + 20
                if needed > length:
                    return None
                timescale = struct.unpack('>I', data[content_offset + 12:content_offset + 16])[0]
                duration = struct.unpack('>I', data[content_offset + 16:content_offset + 20])[0]
            elif version == 1:
                needed = content_offset + 36
                if needed > length:
                    return None
                timescale = struct.unpack('>I', data[content_offset + 24:content_offset + 28])[0]
                duration = struct.unpack('>Q', data[content_offset + 28:content_offset + 36])[0]
            else:
                return None
            if timescale and duration is not None:
                return float(duration) / float(timescale)
            return None
        offset += atom_size
    return None


def _bind_turn_image_bucket(turn_index: int) -> Dict[str, int]:
    """Ensure the mutable image counter bucket for a turn exists and is active."""
    bucket = game_state.image_counts_by_turn.setdefault(turn_index, {})
    game_state.current_turn_image_counts = bucket
    game_state.current_turn_index_for_image_counts = turn_index
    return bucket


def record_image_usage(
    model: Optional[str],
    *,
    purpose: str,
    tier: str = "standard",
    images: int = 1,
    turn_index: Optional[int] = None,
) -> None:
    model_key = _normalize_model_name(model)
    game_state.last_image_model = model_key or model
    game_state.last_image_kind = purpose
    game_state.last_image_count = images if images > 0 else 0
    record_turn = turn_index if isinstance(turn_index, int) else None
    if record_turn is None and isinstance(game_state.turn_index, int):
        record_turn = game_state.turn_index
    game_state.last_image_turn_index = record_turn
    cost_info = calculate_image_cost(model, tier=tier, images=images)

    game_state.last_image_tier = cost_info.get("tier") if cost_info else tier
    if cost_info:
        cost_value = cost_info.get("cost_usd")
        per_image_value = cost_info.get("usd_per_image")
        tokens_value = cost_info.get("tokens_per_image")
        game_state.last_image_cost_usd = float(cost_value) if isinstance(cost_value, (int, float)) else None
        game_state.last_image_usd_per_image = (
            float(per_image_value) if isinstance(per_image_value, (int, float)) else None
        )
        game_state.last_image_tokens = tokens_value if isinstance(tokens_value, int) else None
        if isinstance(cost_value, (int, float)):
            game_state.session_image_cost_usd += float(cost_value)
        if purpose == "scene":
            game_state.last_scene_image_model = game_state.last_image_model
            game_state.last_scene_image_cost_usd = game_state.last_image_cost_usd
            game_state.last_scene_image_usd_per_image = game_state.last_image_usd_per_image
            game_state.last_scene_image_turn_index = record_turn
    else:
        game_state.last_image_cost_usd = None
        game_state.last_image_usd_per_image = None
        game_state.last_image_tokens = None
        if purpose == "scene":
            game_state.last_scene_image_model = game_state.last_image_model
            game_state.last_scene_image_cost_usd = None
            game_state.last_scene_image_usd_per_image = None
            game_state.last_scene_image_turn_index = record_turn

    if images > 0:
        game_state.session_image_requests += images
        game_state.session_image_kind_counts[purpose] = (
            game_state.session_image_kind_counts.get(purpose, 0) + images
        )
    normalized = (purpose or "unknown").strip().lower()
    if normalized in {"scene", "portrait"}:
        key = normalized
    else:
        key = "other"

    if isinstance(record_turn, int):
        turn_counts = _bind_turn_image_bucket(record_turn)
        turn_counts[key] = turn_counts.get(key, 0) + images
    else:
        game_state.current_turn_image_counts[key] = (
            game_state.current_turn_image_counts.get(key, 0) + images
        )
        game_state.current_turn_index_for_image_counts = None


def record_video_usage(
    model: Optional[str],
    *,
    seconds: Optional[float],
    turn_index: Optional[int] = None,
    tier: str = "default",
) -> None:
    model_key = _normalize_model_name(model)
    game_state.last_video_model = model_key or model
    duration_value = float(seconds) if isinstance(seconds, (int, float)) and seconds >= 0 else None
    game_state.last_video_seconds = duration_value
    record_turn = turn_index if isinstance(turn_index, int) else None
    if record_turn is None and isinstance(game_state.turn_index, int):
        record_turn = game_state.turn_index
    game_state.last_video_turn_index = record_turn

    cost_info = calculate_video_cost(model, seconds=duration_value, tier=tier)
    if cost_info:
        cost_value = cost_info.get("cost_usd")
        per_second = cost_info.get("usd_per_second")
        game_state.last_video_tier = cost_info.get("tier") or tier
        game_state.last_video_cost_usd = float(cost_value) if isinstance(cost_value, (int, float)) else None
        game_state.last_video_usd_per_second = (
            float(per_second) if isinstance(per_second, (int, float)) else None
        )
        if isinstance(cost_value, (int, float)):
            game_state.session_video_cost_usd += float(cost_value)
    else:
        game_state.last_video_tier = tier
        game_state.last_video_cost_usd = None
        game_state.last_video_usd_per_second = None

    if duration_value is not None:
        game_state.session_video_seconds += duration_value
    game_state.session_video_requests += 1

# -------- Text model providers --------
TEXT_PROVIDER_GEMINI: Final[Literal["gemini"]] = "gemini"
TEXT_PROVIDER_GROK: Final[Literal["grok"]] = "grok"
TEXT_PROVIDER_OPENAI: Final[Literal["openai"]] = "openai"
NARRATION_PROVIDER_ELEVENLABS = "elevenlabs"


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
        api_key = (game_state.settings.get("grok_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Grok API key is not set in settings.")
        return api_key
    if provider == TEXT_PROVIDER_OPENAI:
        api_key = (game_state.settings.get("openai_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is not set in settings.")
        return api_key
    api_key = (
        game_state.settings.get("gemini_api_key")
        or game_state.settings.get("api_key")
        or ""
    ).strip()
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

    converted = cast(Dict[str, Any], _convert(copy.deepcopy(schema)))

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
    name: str = Field(
        validation_alias=AliasChoices("n", "name"),
        serialization_alias="n",
    )
    expertise: str = Field(
        validation_alias=AliasChoices("x", "expertise"),
        serialization_alias="x",
    )  # novice|apprentice|journeyman|expert|master


class PlayerUpdate(BaseModel):
    player_id: str = Field(
        validation_alias=AliasChoices("pid", "player_id"),
        serialization_alias="pid",
    )
    character_class: str = Field(
        validation_alias=AliasChoices("cls", "character_class", "player_class", "class"),
        serialization_alias="cls",
    )
    abilities: List[Ability] = Field(
        validation_alias=AliasChoices("ab", "abilities"),
        serialization_alias="ab",
    )
    inventory: List[str] = Field(
        validation_alias=AliasChoices("inv", "inventory"),
        serialization_alias="inv",
    )
    conditions: List[str] = Field(
        validation_alias=AliasChoices("cond", "conditions"),
        serialization_alias="cond",
    )


class PublicStatus(BaseModel):
    player_id: str = Field(
        validation_alias=AliasChoices("pid", "player_id"),
        serialization_alias="pid",
    )  # player id
    status_word: str = Field(
        validation_alias=AliasChoices("word", "status_word", "status"),
        serialization_alias="word",
    )  # one-word public status


class VideoPromptStructured(BaseModel):
    """Structured video prompt bundle returned by the GM model."""

    prompt: str
    negative_prompt: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("negative_prompt", "negativePrompt"),
        serialization_alias="negative_prompt",
    )


class TurnStructured(BaseModel):
    """Structured output we expect from the active text model per turn."""

    narrative: str = Field(
        validation_alias=AliasChoices("nar", "narrative"),
        serialization_alias="nar",
    )  # narrative for the next turn
    image_prompt: str = Field(
        validation_alias=AliasChoices("img", "image_prompt"),
        serialization_alias="img",
    )  # image prompt for gemini-2.5-flash-image-preview
    video: Optional[VideoPromptStructured | str] = Field(
        default=None,
        validation_alias=AliasChoices("vid", "video"),
        serialization_alias="vid",
    )  # Video prompt bundle (legacy string allowed)
    public_statuses: List[PublicStatus] = Field(
        default_factory=list,
        validation_alias=AliasChoices("pub", "public_statuses"),
        serialization_alias="pub",
    )
    updates: List[PlayerUpdate] = Field(
        default_factory=list,
        validation_alias=AliasChoices("upd", "updates"),
        serialization_alias="upd",
    )


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
class SceneVideo:
    url: str
    prompt: str
    negative_prompt: Optional[str]
    model: str
    updated_at: float
    file_path: str


def scene_video_payload(video: Optional["SceneVideo"]) -> Optional[Dict[str, Any]]:
    if not video:
        return None
    return {
        "url": video.url,
        "prompt": video.prompt,
        "negative_prompt": video.negative_prompt,
        "model": video.model,
        "updated_at": video.updated_at,
    }


@dataclass
class Player:
    id: str
    name: str
    background: str
    character_class: str = ""
    abilities: List[Ability] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    status_word: str = "Unknown"
    connected: bool = False
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
    reason: str = ""  # "resolving_turn" | "generating_image" | "generating_portrait" | "generating_video"


@dataclass
class GameState:
    settings: Dict[str, Any] = field(default_factory=lambda: DEFAULT_SETTINGS.copy())
    players: Dict[str, Player] = field(default_factory=dict)  # player_id -> Player
    departed_players: Dict[str, str] = field(default_factory=dict)  # normalized_name -> display name
    submissions: Dict[str, str] = field(default_factory=dict)  # player_id -> text
    current_narrative: str = ""
    turn_index: int = 0
    history: List[TurnRecord] = field(default_factory=list)
    history_summary: List[str] = field(default_factory=list)
    lock: LockState = field(default_factory=LockState)
    language: str = DEFAULT_LANGUAGE
    global_sockets: Set[WebSocket] = field(default_factory=set, repr=False, compare=False)
    # last generated image
    last_image_data_url: Optional[str] = None
    last_image_prompt: Optional[str] = None
    # last generated video prompt (if provided by the model)
    last_video_prompt: Optional[str] = None
    last_video_negative_prompt: Optional[str] = None
    scene_video: Optional[SceneVideo] = None
    last_scene_video_turn_index: Optional[int] = None
    last_video_model: Optional[str] = None
    last_video_tier: Optional[str] = None
    last_video_cost_usd: Optional[float] = None
    last_video_usd_per_second: Optional[float] = None
    last_video_seconds: Optional[float] = None
    last_video_turn_index: Optional[int] = None
    session_video_cost_usd: float = 0.0
    session_video_seconds: float = 0.0
    session_video_requests: int = 0
    # token usage metadata returned by the last text model call
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
    last_image_usd_per_image: Optional[float] = None
    last_image_tokens: Optional[int] = None
    last_image_count: int = 0
    last_image_turn_index: Optional[int] = None
    session_image_cost_usd: float = 0.0
    session_image_requests: int = 0
    last_scene_image_cost_usd: Optional[float] = None
    last_scene_image_usd_per_image: Optional[float] = None
    last_scene_image_model: Optional[str] = None
    last_scene_image_turn_index: Optional[int] = None
    last_manual_scene_image_turn_index: Optional[int] = None
    session_image_kind_counts: Dict[str, int] = field(default_factory=dict)
    current_turn_image_counts: Dict[str, int] = field(default_factory=dict)
    current_turn_index_for_image_counts: Optional[int] = None
    image_counts_by_turn: Dict[int, Dict[str, int]] = field(default_factory=dict)
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
    auto_video_enabled: bool = False
    last_text_request: Dict[str, Any] = field(default_factory=dict)
    last_text_response: Dict[str, Any] = field(default_factory=dict)
    last_turn_request: Dict[str, Any] = field(default_factory=dict)
    last_turn_response: Dict[str, Any] = field(default_factory=dict)

    def public_snapshot(self) -> Dict:
        """Sanitized state for all players."""
        return {
            "turn_index": self.turn_index,
            "current_narrative": self.current_narrative,
            "world_style": self.settings.get("world_style", "High fantasy"),
            "difficulty": self.settings.get("difficulty", "Normal"),
            "history_mode": self.settings.get("history_mode", HISTORY_MODE_FULL),
            "history": [
                {
                    "turn": rec.index,
                    "narrative": rec.narrative,
                    "image_prompt": rec.image_prompt,
                    "timestamp": rec.timestamp,
                }
                for rec in self.history
            ],
            "history_summary": list(self.history_summary),
            "language": self.language,
            "auto_image_enabled": self.auto_image_enabled,
            "auto_tts_enabled": self.auto_tts_enabled,
            "auto_video_enabled": self.auto_video_enabled,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "cls": p.character_class,
                    "status_word": p.status_word,
                    "connected": p.connected,
                    "pending_join": p.pending_join,
                    "pending_leave": p.pending_leave,
                    "portrait": portrait_payload(p.portrait),
                    "submission": self.submissions.get(p.id),
                }
                for p in self.players.values()
            ],
            "departed_players": sorted(
                self.departed_players.values(),
                key=lambda value: value.casefold(),
            ),
            "submissions": [
                {
                    "name": player.name if player else "Unknown",
                    "text": txt,
                }
                for pid, txt in self.submissions.items()
                for player in [self.players.get(pid)]
            ],
            "lock": {"active": self.lock.active, "reason": self.lock.reason},
            "image": {
                "data_url": self.last_image_data_url,
                "prompt": self.last_image_prompt,
            },
            "video": scene_video_payload(self.scene_video),
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
                "cls": p.character_class,
                "pending_join": p.pending_join,
                "pending_leave": p.pending_leave,
                "abilities": [a.model_dump(by_alias=True) for a in p.abilities],
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
            "usd_per_image": self.last_image_usd_per_image,
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

        if isinstance(self.current_turn_index_for_image_counts, int):
            image_turn_index = self.current_turn_index_for_image_counts
        elif isinstance(self.turn_index, int):
            image_turn_index = self.turn_index
        else:
            image_turn_index = None

        image_turn: Dict[str, Any] = {
            "turn_index": image_turn_index,
            "by_kind": dict(self.current_turn_image_counts),
        }

        by_turn_entries: List[Tuple[int, Dict[str, int]]] = []
        for turn, counts in self.image_counts_by_turn.items():
            if isinstance(turn, int):
                by_turn_entries.append((turn, dict(counts)))
        if by_turn_entries:
            image_turn["by_turn"] = {turn: data for turn, data in sorted(by_turn_entries)}

        image_cost_same_turn = 0.0
        if (
            isinstance(self.last_scene_image_cost_usd, (int, float))
            and self.last_scene_image_turn_index == self.turn_index
        ):
            image_cost_same_turn = float(self.last_scene_image_cost_usd)

        video_last = {
            "model": self.last_video_model,
            "tier": self.last_video_tier,
            "seconds": self.last_video_seconds,
            "usd_per_second": self.last_video_usd_per_second,
            "cost_usd": self.last_video_cost_usd,
            "turn_index": self.last_video_turn_index,
        }
        session_video_seconds = self.session_video_seconds
        video_session = {
            "seconds": session_video_seconds,
            "cost_usd": self.session_video_cost_usd,
            "requests": self.session_video_requests,
            "avg_usd_per_second": (
                (self.session_video_cost_usd / session_video_seconds)
                if session_video_seconds
                else None
            ),
            "avg_seconds_per_request": (
                (session_video_seconds / self.session_video_requests)
                if self.session_video_requests
                else None
            ),
        }
        video_cost_same_turn = 0.0
        if (
            isinstance(self.last_video_cost_usd, (int, float))
            and self.last_video_turn_index == self.turn_index
        ):
            video_cost_same_turn = float(self.last_video_cost_usd)

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
            + video_cost_same_turn
            + narration_cost_same_turn
        )
        total_session_cost = (
            self.session_cost_usd
            + self.session_image_cost_usd
            + self.session_video_cost_usd
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
                    "usd_per_image": self.last_scene_image_usd_per_image,
                },
            },
            "video": {
                "last": video_last,
                "session": video_session,
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
                    "video_last_usd": self.last_video_cost_usd,
                    "video_last_turn_usd": video_cost_same_turn,
                    "narration_last_usd": self.last_tts_cost_usd,
                    "narration_last_turn_usd": narration_cost_same_turn,
                    "text_session_usd": self.session_cost_usd,
                    "image_session_usd": self.session_image_cost_usd,
                    "video_session_usd": self.session_video_cost_usd,
                    "narration_session_usd": self.session_tts_cost_usd,
                },
            },
        }

game_state = GameState()
STATE_LOCK = asyncio.Lock()  # coarse lock for turn/image operations
SETTINGS_LOCK = asyncio.Lock()  # for reading/writing settings.json
_RESET_CHECK_TASK: Optional[asyncio.Task] = None


def _clear_scene_video(remove_file: bool = True) -> None:
    """Remove the stored scene video, optionally deleting the backing file."""

    video = game_state.scene_video
    if not video:
        game_state.last_scene_video_turn_index = None
        return
    if remove_file and video.file_path:
        try:
            video_path = Path(video.file_path)
            if video_path.exists():
                video_path.unlink()
        except Exception as exc:  # noqa: BLE001 - best effort cleanup
            print(f"Failed to remove scene video: {exc!r}", file=sys.stderr, flush=True)
    game_state.scene_video = None
    game_state.last_scene_video_turn_index = None


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
            if reset_session_if_inactive():
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
    game_state.players.clear()
    game_state.departed_players.clear()
    game_state.submissions.clear()
    game_state.current_narrative = ""
    game_state.turn_index = 0
    game_state.history.clear()
    game_state.history_summary.clear()
    game_state.lock = LockState(active=False, reason="")
    _clear_scene_video()
    game_state.last_image_data_url = None
    game_state.last_video_prompt = None
    game_state.last_video_negative_prompt = None
    game_state.last_image_prompt = None
    game_state.last_scene_video_turn_index = None
    game_state.last_video_model = None
    game_state.last_video_tier = None
    game_state.last_video_cost_usd = None
    game_state.last_video_usd_per_second = None
    game_state.last_video_seconds = None
    game_state.last_video_turn_index = None
    game_state.session_video_cost_usd = 0.0
    game_state.session_video_seconds = 0.0
    game_state.session_video_requests = 0
    game_state.last_image_model = None
    game_state.last_image_tier = None
    game_state.last_image_kind = None
    game_state.last_image_cost_usd = None
    game_state.last_image_usd_per_image = None
    game_state.last_image_tokens = None
    game_state.last_image_count = 0
    game_state.last_image_turn_index = None
    game_state.last_token_usage = {}
    game_state.last_turn_runtime = None
    game_state.session_token_usage = {"input": 0, "output": 0, "thinking": 0}
    game_state.session_request_count = 0
    game_state.last_cost_usd = None
    game_state.session_cost_usd = 0.0
    game_state.session_image_cost_usd = 0.0
    game_state.session_image_requests = 0
    game_state.session_image_kind_counts = {}
    game_state.last_scene_image_cost_usd = None
    game_state.last_scene_image_usd_per_image = None
    game_state.last_scene_image_model = None
    game_state.last_scene_image_turn_index = None
    game_state.last_manual_scene_image_turn_index = None
    game_state.current_turn_image_counts = {}
    game_state.current_turn_index_for_image_counts = None
    game_state.image_counts_by_turn = {}
    game_state.last_tts_model = None
    game_state.last_tts_voice_id = None
    game_state.last_tts_characters = None
    game_state.last_tts_character_source = None
    game_state.last_tts_credits = None
    game_state.last_tts_cost_usd = None
    game_state.last_tts_request_id = None
    game_state.last_tts_headers = {}
    game_state.last_tts_turn_index = None
    game_state.last_tts_total_credits = None
    game_state.last_tts_remaining_credits = None
    game_state.last_tts_next_reset_unix = None
    game_state.session_tts_characters = 0
    game_state.session_tts_credits = 0.0
    game_state.session_tts_cost_usd = 0.0
    game_state.session_tts_requests = 0
    game_state.auto_image_enabled = False
    game_state.auto_video_enabled = False


def reset_session_if_inactive() -> bool:
    """Reset the game if no active players remain.

    Returns True when a reset occurred.
    """

    if not game_state.players:
        reset_session_progress()
        return True

    for player in game_state.players.values():
        if player.connected:
            return False
        if not player.pending_leave:
            return False

    reset_session_progress()
    return True


def set_language_if_changed(lang: Optional[str]) -> bool:
    if lang is None:
        return False
    normalized = normalize_language(lang)
    if normalized == game_state.language:
        return False
    game_state.language = normalized
    game_state.settings["language"] = normalized
    return True


# -------------------- Helpers: settings I/O --------------------
def ensure_settings_file() -> None:
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def load_settings() -> Dict:
    ensure_settings_file()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = DEFAULT_SETTINGS.copy()
        # Preserve any forward-compatible keys while ensuring defaults exist.
        merged.update(data)
        if "gemini_api_key" not in merged and isinstance(data, dict):
            legacy = data.get("api_key")
            if isinstance(legacy, str) and legacy.strip():
                merged["gemini_api_key"] = legacy
        merged.pop("api_key", None)
        if isinstance(data, dict) and not data.get("video_model") and data.get("veo_model"):
            merged["video_model"] = str(data["veo_model"])
    except Exception:
        merged = DEFAULT_SETTINGS.copy()

    merged["language"] = normalize_language(merged.get("language"))
    merged["history_mode"] = normalize_history_mode(merged.get("history_mode"))
    return merged


async def save_settings(new_settings: Dict[str, Any]) -> None:
    transient_keys = {
        "gemini_api_key_preview",
        "gemini_api_key_set",
        "elevenlabs_api_key_preview",
        "elevenlabs_api_key_set",
        "grok_api_key_preview",
        "grok_api_key_set",
        "openai_api_key_preview",
        "openai_api_key_set",
        # Legacy fields preserved for compatibility cleanup
        "api_key_preview",
        "api_key_set",
    }
    sanitized = {k: v for k, v in new_settings.items() if k not in transient_keys}

    # Preserve existing secrets when callers provide empty placeholders.
    if "api_key" in sanitized and "gemini_api_key" not in sanitized:
        sanitized["gemini_api_key"] = sanitized.pop("api_key")
    else:
        sanitized.pop("api_key", None)

    for secret_key in ("gemini_api_key", "grok_api_key", "openai_api_key", "elevenlabs_api_key"):
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
                    if not existing.get("video_model") and existing.get("veo_model"):
                        merged["video_model"] = str(existing["veo_model"])
            except Exception:  # nosec B110
                pass

        merged.update(sanitized)
        merged.pop("api_key", None)
        merged["language"] = normalize_language(merged.get("language"))
        merged["history_mode"] = normalize_history_mode(merged.get("history_mode"))
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")

        if new_settings is game_state.settings:
            game_state.settings.clear()
            game_state.settings.update(merged)


# -------------------- Helpers: sockets --------------------
async def _send_json_to_sockets(sockets: Set[WebSocket], payload: Dict[str, Any]) -> None:
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
    await _send_json_to_sockets(game_state.global_sockets, payload)


async def elevenlabs_list_models(api_key: str) -> List[ElevenLabsModelInfo]:
    """Return available ElevenLabs narration models for the API key."""
    normalized_key = (api_key or "").strip()
    if not normalized_key:
        return []

    async def loader() -> List[ElevenLabsModelInfo]:
        base = ELEVENLABS_BASE_URL.rstrip("/")
        url = f"{base}/v1/models"
        headers = {"xi-api-key": normalized_key}

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

        items: List[ElevenLabsModelInfo] = []
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
            languages_raw = entry.get("languages") or entry.get("supported_languages")
            parsed_languages: List[str] = []
            if isinstance(languages_raw, list):
                for lang in languages_raw:
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
            elif isinstance(languages_raw, str):
                parsed_languages.append(languages_raw)
            elif languages_raw not in (None, ""):
                parsed_languages.append(str(languages_raw))

            languages: List[str] = parsed_languages
            language_codes: List[str] = []
            for lang_token in languages:
                normalized = _normalize_language_code(lang_token)
                if normalized and normalized not in language_codes:
                    language_codes.append(normalized)
            info: ElevenLabsModelInfo = {
                "id": str(model_id),
                "name": str(model_name),
                "languages": languages,
                "language_codes": language_codes,
            }
            items.append(info)

        return items

    return await _get_cached_models(NARRATION_PROVIDER_ELEVENLABS, normalized_key, loader)


def _elevenlabs_convert_to_base64(
    text: str,
    api_key: str,
    model_id: Optional[str] = None,
) -> ElevenLabsNarrationResult:
    global _ELEVENLABS_IMPORT_ERROR_LOGGED
    empty_result: ElevenLabsNarrationResult = {"audio_base64": None, "metadata": {}}
    if not text or not text.strip():
        return empty_result
    if not api_key:
        return empty_result
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.types import VoiceSettings
    except ImportError:
        if not _ELEVENLABS_IMPORT_ERROR_LOGGED:
            print("ElevenLabs package not available; skipping narration.", file=sys.stderr, flush=True)
            _ELEVENLABS_IMPORT_ERROR_LOGGED = True
        return empty_result

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
        convert_context = client.text_to_speech.with_raw_response.convert(
            ELEVENLABS_VOICE_ID,
            text=text,
            model_id=resolved_model,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            voice_settings=voice_settings,
        )
        with convert_context as response:
            raw_headers: Dict[Any, Any] = {}
            try:
                raw_headers = response.headers or {}
            except Exception:
                raw_headers = {}
            response_headers = {
                str(k): str(v)
                for k, v in raw_headers.items()
                if isinstance(k, str)
            }
            data_stream = response.data if hasattr(response, "data") else ()
            audio_bytes = b"".join(data_stream)
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

    metadata: ElevenLabsNarrationMetadata = {
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


async def schedule_auto_tts(text: str, turn_index: int) -> None:
    if not game_state.auto_tts_enabled:
        return
    if not text or not text.strip():
        return
    api_key = (game_state.settings.get("elevenlabs_api_key") or "").strip()
    if not api_key:
        global _ELEVENLABS_API_KEY_WARNING_LOGGED
        if not _ELEVENLABS_API_KEY_WARNING_LOGGED:
            print(
                "ElevenLabs narration is enabled but the ElevenLabs API key is not configured in settings; "
                "skipping audio.",
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

    configured_model = game_state.settings.get("narration_model")
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
        result_dict: ElevenLabsNarrationResult
        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"audio_base64": None, "metadata": {}}
        audio_b64 = result_dict.get("audio_base64")
        metadata = result_dict["metadata"]
        if not audio_b64:
            await _broadcast_tts_error(
                "ElevenLabs returned no audio data for the narration request.",
                turn_index,
            )
            return
        if not game_state.auto_tts_enabled:
            return

        resolved_model_id = metadata.get("model_id") or model_id or ELEVENLABS_MODEL_ID

        characters_final = metadata.get("characters_final")
        credits_estimated = metadata.get("estimated_credits")
        cost_estimated = metadata.get("estimated_cost_usd")
        headers_payload = cast(Dict[str, str], metadata.get("headers") or {})

        game_state.last_tts_model = resolved_model_id
        game_state.last_tts_voice_id = ELEVENLABS_VOICE_ID
        request_id_meta = metadata.get("request_id")
        game_state.last_tts_request_id = request_id_meta if isinstance(request_id_meta, str) else None
        game_state.last_tts_characters = int(characters_final) if isinstance(characters_final, int) else None
        game_state.last_tts_character_source = metadata.get("character_source")
        game_state.last_tts_credits = float(credits_estimated) if isinstance(credits_estimated, (int, float)) else None
        game_state.last_tts_cost_usd = float(cost_estimated) if isinstance(cost_estimated, (int, float)) else None
        game_state.last_tts_turn_index = turn_index
        game_state.last_tts_headers = {
            str(k): str(v)
            for k, v in headers_payload.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        total_credits_meta = metadata.get("subscription_total_credits")
        remaining_credits_meta = metadata.get("subscription_remaining_credits")
        next_reset_meta = metadata.get("subscription_next_reset_unix")
        game_state.last_tts_total_credits = (
            int(total_credits_meta) if isinstance(total_credits_meta, int) else None
        )
        game_state.last_tts_remaining_credits = (
            int(remaining_credits_meta) if isinstance(remaining_credits_meta, int) else None
        )
        game_state.last_tts_next_reset_unix = int(next_reset_meta) if isinstance(next_reset_meta, int) else None
        if (
            game_state.last_tts_total_credits is None
            and isinstance(total_credits_meta, (float, str))
            and str(total_credits_meta).isdigit()
        ):
            game_state.last_tts_total_credits = int(float(total_credits_meta))
        if (
            game_state.last_tts_remaining_credits is None
            and isinstance(remaining_credits_meta, (float, str))
            and str(remaining_credits_meta).isdigit()
        ):
            game_state.last_tts_remaining_credits = int(float(remaining_credits_meta))

        if isinstance(characters_final, int):
            game_state.session_tts_characters += characters_final
        if isinstance(credits_estimated, (int, float)):
            game_state.session_tts_credits += float(credits_estimated)
        if isinstance(cost_estimated, (int, float)):
            game_state.session_tts_cost_usd += float(cost_estimated)
        game_state.session_tts_requests += 1

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
        await _send_json_to_sockets(game_state.global_sockets, payload)

    asyncio.create_task(_worker())


async def schedule_auto_scene_image(prompt: Optional[str], turn_index: int, *, force: bool = False) -> None:
    if not game_state.auto_image_enabled:
        return
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return
    if not force:
        last_scene_turn = game_state.last_scene_image_turn_index
        if isinstance(last_scene_turn, int) and last_scene_turn == turn_index:
            return

    async def _worker() -> None:
        attempts = 0
        max_attempts = 20
        while True:
            if not game_state.auto_image_enabled:
                return
            if attempts >= max_attempts:
                print(
                    "Auto image generation aborted after repeated retries while the game was busy.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if game_state.lock.active:
                attempts += 1
                await asyncio.sleep(0.5)
                continue

            async with STATE_LOCK:
                if game_state.lock.active:
                    attempts += 1
                    await asyncio.sleep(0.5)
                    continue
                current_prompt = (game_state.last_image_prompt or prompt_text).strip()
                if not current_prompt:
                    return
                if not force:
                    last_turn = game_state.last_scene_image_turn_index
                    if isinstance(last_turn, int) and last_turn == turn_index:
                        return
                game_state.lock = LockState(active=True, reason="generating_image")
                await broadcast_public()

            try:
                img_model = game_state.settings.get("image_model") or "gemini-2.5-flash-image-preview"
                data_url = await gemini_generate_image(
                    img_model,
                    current_prompt,
                    purpose="scene",
                    turn_index=turn_index,
                )
                _clear_scene_video()
                game_state.last_image_data_url = data_url
                game_state.last_manual_scene_image_turn_index = None
                game_state.last_image_prompt = current_prompt
                await announce("Image generated.")
                await broadcast_public()
            except HTTPException as exc:
                print(f"Auto image generation failed: {exc.detail}", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Auto image generation error: {exc!r}", file=sys.stderr, flush=True)
            finally:
                game_state.lock = LockState(active=False, reason="")
                await broadcast_public()
            return

    asyncio.create_task(_worker())


async def schedule_auto_scene_video(
    prompt: Optional[str],
    turn_index: int,
    *,
    force: bool = False,
    negative_prompt: Optional[str] = None,
) -> None:
    if not game_state.auto_video_enabled:
        return

    model_setting = game_state.settings.get("video_model")

    prompt_text = (prompt or game_state.last_video_prompt or game_state.last_image_prompt or "").strip()
    if not prompt_text:
        return

    negative_text = negative_prompt
    if isinstance(negative_text, str):
        negative_text = negative_text.strip()
        if not negative_text:
            negative_text = None
    if negative_text is None:
        stored_negative = game_state.last_video_negative_prompt or None
        if isinstance(stored_negative, str):
            stored_negative = stored_negative.strip()
        negative_text = stored_negative or None

    if not force:
        last_turn = game_state.last_scene_video_turn_index
        if isinstance(last_turn, int) and last_turn == turn_index:
            return

    async def _worker() -> None:
        attempts = 0
        max_attempts = 20
        while True:
            if not game_state.auto_video_enabled:
                return
            if attempts >= max_attempts:
                print(
                    "Auto video generation aborted after repeated retries while the game was busy.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if game_state.lock.active:
                attempts += 1
                await asyncio.sleep(0.5)
                continue

            async with STATE_LOCK:
                if game_state.lock.active:
                    attempts += 1
                    await asyncio.sleep(0.5)
                    continue
                current_prompt = ((game_state.last_video_prompt or game_state.last_image_prompt) or prompt_text).strip()
                if not current_prompt:
                    return
                current_negative = negative_text
                if not force:
                    last_turn = game_state.last_scene_video_turn_index
                    if isinstance(last_turn, int) and last_turn == turn_index:
                        return
                game_state.lock = LockState(active=True, reason="generating_video")
                await broadcast_public()

            try:
                image_for_video: Optional[str] = None
                if _is_parallax_model(model_setting):
                    source_image = game_state.last_image_data_url
                    if not source_image:
                        print(
                            "Auto video generation skipped: Parallax requires an existing image.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return
                    image_for_video = source_image
                elif _is_framepack_static_model(model_setting):
                    source_image = game_state.last_image_data_url
                    if not source_image:
                        print(
                            "Auto video generation skipped: Static FramePack requires an existing image.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return
                    image_for_video = source_image
                video_kwargs = {
                    "negative_prompt": current_negative,
                    "turn_index": turn_index,
                }
                if image_for_video is not None:
                    video_kwargs["image_data_url"] = image_for_video
                new_video = await generate_scene_video(
                    current_prompt,
                    model_setting,
                    **video_kwargs,
                )
                _clear_scene_video()
                game_state.scene_video = new_video
                game_state.last_video_prompt = current_prompt
                game_state.last_video_negative_prompt = current_negative
                if isinstance(turn_index, int):
                    game_state.last_scene_video_turn_index = turn_index
                else:
                    last_hist_turn = game_state.history[-1].index if game_state.history else game_state.turn_index
                    game_state.last_scene_video_turn_index = last_hist_turn if isinstance(last_hist_turn, int) else None
                await announce("Video generated.")
                await broadcast_public()
            except HTTPException as exc:
                print(f"Auto video generation failed: {exc.detail}", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Auto video generation error: {exc!r}", file=sys.stderr, flush=True)
            finally:
                game_state.lock = LockState(active=False, reason="")
                await broadcast_public()
            return

    asyncio.create_task(_worker())


async def broadcast_public() -> None:
    payload = {"event": "state", "data": game_state.public_snapshot()}
    await _send_json_to_sockets(game_state.global_sockets, payload)


async def send_private(player_id: str) -> None:
    payload = {"event": "private", "data": game_state.private_snapshot_for(player_id)}
    p = game_state.players.get(player_id)
    if not p:
        return
    await _send_json_to_sockets(p.sockets, payload)


async def announce(message: str) -> None:
    payload = {"event": "announce", "data": {"message": message, "ts": time.time()}}
    await _send_json_to_sockets(game_state.global_sockets, payload)


def authenticate_player(player_id: str, token: str) -> Player:
    if not player_id or player_id not in game_state.players:
        raise HTTPException(status_code=404, detail="Unknown player.")
    player = game_state.players[player_id]
    if not token or not secrets.compare_digest(token, player.token):
        raise HTTPException(status_code=403, detail="Invalid player token.")
    return player


# -------------------- Helpers: Gemini REST --------------------
def require_gemini_api_key() -> str:
    """Return the configured Gemini API key or raise if missing."""

    return require_text_api_key(TEXT_PROVIDER_GEMINI)


async def gemini_list_models() -> List[ProviderModelInfo]:
    api_key = require_gemini_api_key()

    async def loader() -> List[ProviderModelInfo]:
        headers = {"x-goog-api-key": api_key}
        models: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {"key": api_key, "pageSize": 200}
                if page_token:
                    params["pageToken"] = page_token
                r = await client.get(MODELS_LIST_URL, headers=headers, params=params)
                if r.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"Model list failed: {r.text}")
                data = r.json()
                chunk = data.get("models") or []
                if isinstance(chunk, list):
                    models.extend([m for m in chunk if isinstance(m, dict)])
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        entries: List[ProviderModelInfo] = []
        for raw in models:
            name_value = raw.get("name") or raw.get("id")
            name = str(name_value).strip() if name_value else ""
            if not name:
                continue
            display_raw = raw.get("displayName") or raw.get("description") or name
            supported_raw = raw.get("supportedGenerationMethods") or raw.get("supported_actions")
            supported = _normalize_supported_list(supported_raw)
            supported_lower = {item.lower() for item in supported}
            is_veo = name.lower().startswith("models/veo")
            is_video = is_veo or "predictlongrunning" in supported_lower
            is_text = bool({"generatecontent", "responses"}.intersection(supported_lower))
            if is_video:
                category: Literal["text", "video", "other"] = "video"
            elif is_text:
                category = "text"
            else:
                category = "other"
            model_id = name.replace("models/", "", 1) if name.startswith("models/") else None
            entry: ProviderModelInfo = {
                "name": name,
                "displayName": str(display_raw),
                "supported": supported,
                "provider": TEXT_PROVIDER_GEMINI,
                "category": category,
            }
            if model_id and model_id != name:
                entry["modelId"] = model_id
            if is_veo:
                entry["family"] = "veo"
            entries.append(entry)
        return entries

    return await _get_cached_models(TEXT_PROVIDER_GEMINI, api_key, loader)


async def grok_list_models(api_key: str) -> List[ProviderModelInfo]:
    normalized_key = (api_key or "").strip()
    if not normalized_key:
        return []

    async def loader() -> List[ProviderModelInfo]:
        headers = {"Authorization": f"Bearer {normalized_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(GROK_MODELS_URL, headers=headers)
        if r.status_code != 200:
            detail_text = r.text or f"HTTP {r.status_code}"
            raise HTTPException(status_code=502, detail=f"Grok model list failed: {detail_text}")
        try:
            data = r.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Malformed Grok model list: {exc}") from exc

        raw_entries: List[Dict[str, Any]]
        if isinstance(data, dict):
            candidates = data.get("data") or data.get("models")
            if isinstance(candidates, list):
                raw_entries = [entry for entry in candidates if isinstance(entry, dict)]
            else:
                raw_entries = []
        elif isinstance(data, list):
            raw_entries = [entry for entry in data if isinstance(entry, dict)]
        else:
            raw_entries = []

        entries: List[ProviderModelInfo] = []
        for raw in raw_entries:
            identifier_source = (
                raw.get("id")
                or raw.get("name")
                or raw.get("model")
                or raw.get("slug")
                or ""
            )
            identifier = str(identifier_source).strip()
            if not identifier:
                continue
            display = (
                raw.get("display_name")
                or raw.get("displayName")
                or raw.get("description")
                or raw.get("title")
                or identifier
            )
            supported_raw = (
                raw.get("capabilities")
                or raw.get("modalities")
                or raw.get("endpoints")
                or raw.get("interfaces")
            )
            supported = _normalize_supported_list(supported_raw)
            if not any("chat" in entry.lower() for entry in supported):
                supported.append("chat.completions")
            entry: ProviderModelInfo = {
                "name": identifier,
                "displayName": str(display),
                "supported": supported,
                "provider": TEXT_PROVIDER_GROK,
                "category": "text",
            }
            entries.append(entry)
        return entries

    return await _get_cached_models(TEXT_PROVIDER_GROK, normalized_key, loader)


async def openai_list_models(api_key: str) -> List[ProviderModelInfo]:
    normalized_key = (api_key or "").strip()
    if not normalized_key:
        return []

    async def loader() -> List[ProviderModelInfo]:
        headers = {"Authorization": f"Bearer {normalized_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(OPENAI_MODELS_URL, headers=headers)
        if r.status_code != 200:
            detail_text = r.text or f"HTTP {r.status_code}"
            raise HTTPException(status_code=502, detail=f"OpenAI model list failed: {detail_text}")
        try:
            data = r.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Malformed OpenAI model list: {exc}") from exc

        raw_entries: List[Dict[str, Any]] = []
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            raw_entries = [entry for entry in data["data"] if isinstance(entry, dict)]
        elif isinstance(data, list):
            raw_entries = [entry for entry in data if isinstance(entry, dict)]

        entries: List[ProviderModelInfo] = []
        for raw in raw_entries:
            identifier_source = raw.get("id") or raw.get("model") or raw.get("name") or raw.get("slug")
            identifier = str(identifier_source).strip() if identifier_source else ""
            if not identifier:
                continue
            display = (
                raw.get("display_name")
                or raw.get("displayName")
                or raw.get("description")
                or identifier
            )
            supported_raw = (
                raw.get("capabilities")
                or raw.get("modalities")
                or raw.get("interfaces")
            )
            supported = _normalize_supported_list(supported_raw, fallback=["responses"])
            entry: ProviderModelInfo = {
                "name": identifier,
                "displayName": str(display),
                "supported": supported,
                "provider": TEXT_PROVIDER_OPENAI,
                "category": "text",
            }
            entries.append(entry)
        return entries

    return await _get_cached_models(TEXT_PROVIDER_OPENAI, normalized_key, loader)


def _sanitize_request_headers(headers: Dict[str, str]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"x-goog-api-key", "authorization"}:
            sanitized[key] = "***"
        else:
            sanitized[key] = value
    return sanitized


async def _post_structured_request(
    *,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    provider: str,
    record_usage: bool,
    dev_snapshot: str,
    request_meta: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Tuple[httpx.Response, str, Any]:
    meta = dict(request_meta or {})
    meta.setdefault("record_usage", record_usage)
    request_snapshot = {
        "timestamp": time.time(),
        "url": url,
        "provider": provider,
        "headers": _sanitize_request_headers(headers),
        "body": body,
    }
    request_snapshot.update(meta)
    game_state.last_text_request = request_snapshot
    if dev_snapshot == "turn":
        game_state.last_turn_request = copy.deepcopy(request_snapshot)

    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=body)
    elapsed = time.perf_counter() - start_time
    if record_usage:
        game_state.last_turn_runtime = elapsed

    raw_text = getattr(response, "text", "")
    if callable(raw_text):
        try:
            raw_text = raw_text()
        except Exception:
            raw_text = ""
    if raw_text is None:
        raw_text = ""
    try:
        response_json = response.json()
    except Exception:
        response_json = None

    headers_source = getattr(response, "headers", {}) or {}
    header_items = headers_source.items() if hasattr(headers_source, "items") else []
    response_headers = {str(k): str(v) for k, v in header_items}
    response_snapshot = {
        "timestamp": time.time(),
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "headers": response_headers,
        "json": response_json,
        "text": raw_text,
        "provider": provider,
    }
    game_state.last_text_response = response_snapshot
    if dev_snapshot == "turn":
        game_state.last_turn_response = copy.deepcopy(response_snapshot)
    return response, raw_text, response_json


async def _gemini_generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict[str, Any]:
    """Call Gemini with a JSON schema and return the parsed response."""
    api_key = require_gemini_api_key()
    url = GENERATE_CONTENT_URL.format(model=model)
    mode = (game_state.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    effective_temperature = DEFAULT_TEXT_TEMPERATURE

    thinking_budget = None
    if include_thinking_budget:
        thinking_budget = compute_thinking_budget(model, mode)

    body: Dict[str, Any] = {
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
    response, raw_text, response_json = await _post_structured_request(
        url=url,
        headers=headers,
        body=body,
        provider=TEXT_PROVIDER_GEMINI,
        record_usage=record_usage,
        dev_snapshot=dev_snapshot,
        request_meta={
            "model": model,
            "thinking_mode": mode,
            "temperature": effective_temperature,
            "thinking_budget": thinking_budget,
        },
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Text generation failed: {raw_text}")
    if not isinstance(response_json, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")
    data = response_json
    usage_meta = data.get("usageMetadata") or {}
    if record_usage:
        # Capture token usage for UI display; missing values remain None.
        game_state.last_token_usage = {
            "input": usage_meta.get("promptTokenCount"),
            "output": usage_meta.get("candidatesTokenCount"),
            "thinking": usage_meta.get("thoughtsTokenCount") or usage_meta.get("thinkingTokenCount"),
        }
        for key in ("input", "output", "thinking"):
            val = game_state.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                game_state.session_token_usage[key] = game_state.session_token_usage.get(key, 0) + val
        game_state.session_request_count += 1

        prompt_tokens = game_state.last_token_usage.get("input")
        output_tokens = game_state.last_token_usage.get("output")
        thinking_tokens = game_state.last_token_usage.get("thinking")
        combined_output_tokens = sum(
            val for val in [output_tokens, thinking_tokens] if isinstance(val, int) and val > 0
        )
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            game_state.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if game_state.last_cost_usd is not None:
                game_state.session_cost_usd += game_state.last_cost_usd
        else:
            game_state.last_cost_usd = None

    # Text is returned in candidates[0].content.parts[0].text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        txt = ""
        for prt in parts:
            if "text" in prt and prt["text"]:
                txt += prt["text"]
        parsed_obj = json.loads(txt)
        if not isinstance(parsed_obj, dict):
            raise HTTPException(status_code=502, detail="Malformed response from model.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Malformed response from model.") from exc
    return parsed_obj


async def _grok_generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict[str, Any]:
    api_key = require_text_api_key(TEXT_PROVIDER_GROK)
    url = GROK_CHAT_COMPLETIONS_URL
    mode = (game_state.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    effective_temperature = DEFAULT_TEXT_TEMPERATURE

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
    response, raw_text, response_json = await _post_structured_request(
        url=url,
        headers=headers,
        body=body,
        provider=TEXT_PROVIDER_GROK,
        record_usage=record_usage,
        dev_snapshot=dev_snapshot,
        request_meta={
            "model": model,
            "thinking_mode": mode,
            "temperature": effective_temperature,
        },
    )

    if response.status_code != 200:
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
        game_state.last_token_usage = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "thinking": reasoning_tokens,
        }
        for key in ("input", "output", "thinking"):
            val = game_state.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                game_state.session_token_usage[key] = game_state.session_token_usage.get(key, 0) + val
        game_state.session_request_count += 1

        combined_output_tokens = 0
        for val in (completion_tokens, reasoning_tokens):
            if isinstance(val, int) and val > 0:
                combined_output_tokens += val
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens or None)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            game_state.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if isinstance(game_state.last_cost_usd, float):
                game_state.session_cost_usd += game_state.last_cost_usd
        else:
            game_state.last_cost_usd = None

    choices = response_json.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    first_choice = choices[0] if isinstance(choices, list) else None
    if not isinstance(first_choice, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")
    message = first_choice.get("message")
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
        parsed_obj = json.loads(content_text)
    except Exception:
        try:
            parsed_obj = json.loads(_clean_json_text(content_text))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Malformed response from model: {exc}") from exc

    if not isinstance(parsed_obj, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")

    return parsed_obj

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
    mode = (game_state.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"

    supports_temperature = _openai_supports_temperature(model)
    effective_temperature = DEFAULT_TEXT_TEMPERATURE if supports_temperature else None

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
    if effective_temperature is not None:
        body["temperature"] = effective_temperature
    reasoning_cfg = None
    if include_thinking_budget:
        reasoning_cfg = _openai_reasoning_for_mode(model, mode)
    if reasoning_cfg:
        body["reasoning"] = reasoning_cfg
    if response_format.get("type") == "json_schema":
        json_schema_cfg = response_format.get("json_schema")
        normalized_schema_cfg: Dict[str, Any] = {}
        if isinstance(json_schema_cfg, dict):
            normalized_schema_cfg.update(json_schema_cfg)
        if "name" not in normalized_schema_cfg:
            normalized_schema_cfg["name"] = schema_name or "payload"
        if "schema" not in normalized_schema_cfg:
            normalized_schema_cfg["schema"] = json_schema or {}
        response_format = {"type": "json_schema", "json_schema": normalized_schema_cfg}

    body["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if _openai_requires_reasoning_header(model):
        headers["OpenAI-Beta"] = "reasoning"

    response, raw_text, response_json = await _post_structured_request(
        url=url,
        headers=headers,
        body=body,
        provider=TEXT_PROVIDER_OPENAI,
        record_usage=record_usage,
        dev_snapshot=dev_snapshot,
        request_meta={
            "model": model,
            "thinking_mode": mode,
            "temperature": effective_temperature,
        },
    )

    if response.status_code != 200:
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
        game_state.last_token_usage = {
            "input": usage_meta.get("input_tokens"),
            "output": usage_meta.get("output_tokens"),
            "thinking": reasoning_tokens if isinstance(reasoning_tokens, int) else None,
        }
        for key in ("input", "output", "thinking"):
            val = game_state.last_token_usage.get(key)
            if isinstance(val, int) and val >= 0:
                game_state.session_token_usage[key] = game_state.session_token_usage.get(key, 0) + val
        game_state.session_request_count += 1

        prompt_tokens = game_state.last_token_usage.get("input")
        output_tokens = game_state.last_token_usage.get("output")
        thinking_tokens = game_state.last_token_usage.get("thinking")
        combined_output_tokens = sum(
            val for val in [output_tokens, thinking_tokens] if isinstance(val, int) and val > 0
        )
        cost_info = calculate_turn_cost(model, prompt_tokens, combined_output_tokens)
        if cost_info is not None:
            total_cost = cost_info.get("total_usd")
            game_state.last_cost_usd = float(total_cost) if isinstance(total_cost, (int, float)) else None
            if game_state.last_cost_usd is not None:
                game_state.session_cost_usd += game_state.last_cost_usd
        else:
            game_state.last_cost_usd = None

    output_entries: List[Any] = []
    raw_output = response_json.get("output")
    if isinstance(raw_output, list):
        output_entries = raw_output
    elif isinstance(raw_output, dict):
        output_entries = [raw_output]
    elif isinstance(response_json.get("data"), list):
        output_entries = response_json["data"]

    text_parts: List[str] = []

    def _collect_text(entry: Any) -> None:
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
        parsed_obj = json.loads(assembled)
    except Exception:
        cleaned = assembled.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`\n")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            parsed_obj = json.loads(cleaned)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Malformed response from model: {exc}") from exc
    if not isinstance(parsed_obj, dict):
        raise HTTPException(status_code=502, detail="Malformed response from model.")
    return parsed_obj


async def _generate_structured(
    *,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    record_usage: bool = True,
    include_thinking_budget: bool = True,
    dev_snapshot: str = "generic",
    schema_name: str = "payload",
) -> Dict[str, Any]:
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


async def request_turn_payload(
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> TurnStructured:
    """Call the configured text provider with a forced JSON schema and parse the turn payload."""
    parsed = await _generate_structured(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        temperature=None,
        record_usage=True,
        include_thinking_budget=True,
        dev_snapshot="turn",
        schema_name="turn_payload",
    )
    try:
        return cast(TurnStructured, TurnStructured.model_validate(parsed))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Schema validation error: {e}")


async def request_summary_payload(
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> SummaryStructured:
    """Helper for generating concise history summaries without mutating token metrics."""
    parsed = await _generate_structured(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        schema=schema,
        record_usage=False,
        include_thinking_budget=False,
        dev_snapshot="summary",
        schema_name="summary_payload",
    )
    try:
        return cast(SummaryStructured, SummaryStructured.model_validate(parsed))
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
    """Assemble user parts for scene images.

    - Inlines player portrait references so the image model can keep faces consistent.
    - Adds concise directives per player to reflect their current inventory and conditions.
    """
    references: List[Tuple[Player, str, str]] = []
    for player_id in sorted(game_state.players.keys()):
        player = game_state.players[player_id]
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
    world_style = game_state.settings.get("world_style", "High fantasy")
    directive_lines: List[str] = [
        "Use the provided player portraits to keep each adventurer's appearance consistent across this scene.",
        "Match faces, hair, and distinctive accessories from the references even if lighting or outfits change.",
        f"World style: {world_style}. Keep the setting aesthetics consistent.",
    ]
    for player, _, _ in references:
        descriptors: List[str] = []
        cls = (player.character_class or "").strip()
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

        # Inventory and conditions guide the depiction of equipment/props and visible state.
        inv_items = [item.strip() for item in (player.inventory or []) if isinstance(item, str) and item.strip()]
        if inv_items:
            directive_lines.append(
                f"{player.name}: include visible gear from inventory ({', '.join(inv_items)})."
            )
        cond_items = [c.strip() for c in (player.conditions or []) if isinstance(c, str) and c.strip()]
        if cond_items:
            directive_lines.append(
                f"{player.name}: reflect current conditions ({', '.join(cond_items)})."
            )
    prompt_text = (prompt or "").strip()
    if prompt_text:
        directive_lines.extend(["", f"Scene prompt: {prompt_text}"])
    else:
        directive_lines.extend(["", "Scene prompt: (none provided)"])
    parts.append({"text": "\n".join(directive_lines)})
    return parts


async def gemini_generate_image(
    model: str,
    prompt: str,
    *,
    purpose: str = "scene",
    turn_index: Optional[int] = None,
) -> str:
    """Returns a data URL (base64 image) from gemini-2.5-flash-image-preview."""
    api_key = require_gemini_api_key()
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
            record_image_usage(model, purpose=purpose, images=1, turn_index=turn_index)
            image_bytes = _decode_base64_data(b64)
            if image_bytes:
                suffix = _suffix_from_mime(mime)
                prefix = f"image_{purpose or 'unknown'}"
                try:
                    _archive_generated_media(image_bytes, prefix=prefix, suffix=suffix)
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
            return f"data:{mime};base64,{b64}"
    # Some generations also include text; if no image found, raise
    raise HTTPException(status_code=502, detail="No image data returned by model.")


async def _generate_framepack_video(
    prompt_text: str,
    *,
    output_path: Path,
    image_data_url: Optional[str],
    negative_prompt: Optional[str],
    requested_duration: float,
    prompt_override: Optional[str] = None,
    require_image: bool = False,
    require_image_error: Optional[str] = None,
    use_image_as_end_frame: bool = False,
    end_frame_influence: Optional[float] = None,
) -> None:
    module = _load_framepack_module()
    negative_value = (negative_prompt or "").strip()
    framepack_prompt = prompt_override if prompt_override is not None else prompt_text
    if framepack_prompt is None:
        framepack_prompt = ""

    min_duration = float(getattr(module, "MIN_DURATION_SECONDS", FRAMEPACK_DEFAULT_DURATION_SECONDS))
    max_duration = float(getattr(module, "MAX_DURATION_SECONDS", FRAMEPACK_DEFAULT_DURATION_SECONDS))
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = FRAMEPACK_DEFAULT_DURATION_SECONDS
    duration = max(min_duration, min(duration, max_duration))
    model_variant = getattr(module, "FRAMEPACK_DEFAULT_VARIANT", FRAMEPACK_DEFAULT_VARIANT)
    if not isinstance(model_variant, str) or not model_variant:
        model_variant = FRAMEPACK_DEFAULT_VARIANT

    resolved_image = _resolve_framepack_image(image_data_url)
    if require_image and resolved_image is None:
        detail = require_image_error or "FramePack animation requires an existing image."
        raise HTTPException(status_code=400, detail=detail)

    def _run() -> None:
        with tempfile.TemporaryDirectory(prefix="framepack_src_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            source_path: Optional[Path] = None
            if resolved_image is not None:
                image_bytes, suffix = resolved_image
                source_path = temp_dir / f"framepack_source{suffix}"
                source_path.write_bytes(image_bytes)
            end_frame_path: Optional[Path] = None
            if use_image_as_end_frame and source_path is not None:
                end_frame_path = source_path

            auto_download = os.name != "nt"
            download_setting = temp_dir if auto_download else False

            framepack_url = getattr(module, "FRAMEPACK_URL", None)
            if not isinstance(framepack_url, str) or not framepack_url.strip():
                raise RuntimeError("FramePack helper missing FRAMEPACK_URL setting")

            client = module.Client(framepack_url, download_files=download_setting)
            try:
                submit = getattr(module, "submit_job", None)
                if not callable(submit):
                    raise RuntimeError("FramePack helper missing submit_job function")

                submit_args = [client, source_path, framepack_prompt, duration, model_variant]
                submit_kwargs: Dict[str, Any] = {}
                try:
                    signature = inspect.signature(submit)
                    if "negative_prompt" in signature.parameters:
                        submit_kwargs["negative_prompt"] = negative_value
                    if "end_frame" in signature.parameters and end_frame_path is not None:
                        submit_kwargs["end_frame"] = end_frame_path
                    if (
                        "end_frame_influence" in signature.parameters
                        and end_frame_influence is not None
                    ):
                        submit_kwargs["end_frame_influence"] = float(end_frame_influence)
                except (TypeError, ValueError):
                    pass
                job_id = submit(*submit_args, **submit_kwargs)

                monitor = getattr(module, "wait_for_completion", None)
                if not callable(monitor):
                    raise RuntimeError("FramePack helper missing wait_for_completion function")
                monitor_kwargs: Dict[str, Any] = {}
                try:
                    monitor_sig = inspect.signature(monitor)
                    if "verbose" in monitor_sig.parameters:
                        monitor_kwargs["verbose"] = False
                except (TypeError, ValueError):
                    pass
                video_path = monitor(client, job_id, **monitor_kwargs)
                ensure_local = getattr(module, "_ensure_local_video", None)
                if not callable(ensure_local):
                    raise RuntimeError("FramePack helper missing _ensure_local_video helper")
                local_video = ensure_local(
                    client,
                    video_path,
                    temp_dir,
                    job_id,
                    auto_download=auto_download,
                )
                if not local_video:
                    raise RuntimeError("FramePack job did not return a video path")
                final_path = Path(local_video)
                if not final_path.exists():
                    raise FileNotFoundError(
                        f"FramePack reported video at {final_path} but it was not found"
                    )
                output_path.write_bytes(final_path.read_bytes())
            finally:
                close_method = getattr(client, "close", None)
                if callable(close_method):
                    close_method()
                executor = getattr(client, "executor", None)
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

    try:
        await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or "FramePack generation failed"
        raise HTTPException(status_code=502, detail=f"FramePack generation failed: {detail}") from exc


async def generate_scene_video(
    prompt: str,
    model: Optional[str] = None,
    *,
    image_data_url: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    turn_index: Optional[int] = None,
) -> SceneVideo:
    """Create a scene animation video using the provided prompt and optional reference image."""
    requested_model = model or game_state.settings.get("video_model") or DEFAULT_VIDEO_MODEL
    normalized_model = str(requested_model).strip() or DEFAULT_VIDEO_MODEL
    prompt_text = (prompt or "").strip()
    if not prompt_text and not _is_framepack_static_model(normalized_model):
        raise HTTPException(status_code=400, detail="No image available to animate.")
    negative_text = (negative_prompt or "").strip()

    GENERATED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"scene_{int(time.time())}_{secrets.token_hex(4)}.mp4"
    output_path = GENERATED_MEDIA_DIR / filename
    output_path = _unique_media_path(output_path)
    result_model = normalized_model

    if _is_framepack_model(normalized_model) or _is_framepack_static_model(normalized_model):
        is_static_framepack = _is_framepack_static_model(normalized_model)
        try:
            duration_setting = _normalize_video_duration_seconds(
                game_state.settings.get("video_duration_seconds")
            )
        except ValueError:
            duration_setting = int(FRAMEPACK_DEFAULT_DURATION_SECONDS)
        if is_static_framepack:
            source_data_url = image_data_url or game_state.last_image_data_url
            await _generate_framepack_video(
                prompt_text,
                output_path=output_path,
                image_data_url=source_data_url,
                negative_prompt=negative_text,
                requested_duration=duration_setting,
                prompt_override="",
                require_image=True,
                require_image_error="Static FramePack requires an existing image.",
                use_image_as_end_frame=True,
                end_frame_influence=FRAMEPACK_STATIC_END_INFLUENCE,
            )
            result_model = FRAMEPACK_STATIC_MODEL_ID
        else:
            await _generate_framepack_video(
                prompt_text,
                output_path=output_path,
                image_data_url=image_data_url,
                negative_prompt=negative_text,
                requested_duration=duration_setting,
            )
            result_model = FRAMEPACK_MODEL_ID
    elif _is_parallax_model(normalized_model):
        source_data_url = image_data_url or game_state.last_image_data_url
        image_payload = _resolve_parallax_image(source_data_url)
        script_path = APP_DIR / "Animate" / "parallax.py"
        if not script_path.exists():
            raise HTTPException(status_code=500, detail="Parallax animator is unavailable on the server.")

        try:
            duration_setting = _normalize_video_duration_seconds(
                game_state.settings.get("video_duration_seconds")
            )
        except ValueError:
            duration_setting = int(FRAMEPACK_DEFAULT_DURATION_SECONDS)
        duration_seconds = float(duration_setting)

        device_preference = "cuda"
        try:  # torch is optional; fall back to CPU when unavailable
            import torch  # type: ignore
        except Exception:  # pragma: no cover - optional dependency detection only
            device_preference = "cpu"
        else:  # pragma: no branch - simple availability check
            if not torch.cuda.is_available():
                device_preference = "cpu"

        def _run_parallax() -> None:
            image_bytes, suffix = image_payload
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir = Path(tmpdir)
                input_path = tmp_dir / f"parallax_input{suffix}"
                input_path.write_bytes(image_bytes)
                temp_output = tmp_dir / "parallax_output.mp4"

                cmd = [
                    sys.executable,
                    str(script_path),
                    str(input_path),
                    str(temp_output),
                    "--seconds",
                    str(duration_seconds),
                    "--device",
                    device_preference,
                    "--layers",
                    str(PARALLAX_DEFAULT_LAYERS),
                ]

                completed = subprocess.run(
                    cmd,
                    check=False,
                    cwd=str(APP_DIR),
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    stderr = (completed.stderr or "").strip()
                    stdout = (completed.stdout or "").strip()
                    message = stderr or stdout or "Unknown parallax error"
                    snippet = message[:400]
                    raise RuntimeError(snippet)
                if not temp_output.exists():
                    raise RuntimeError("Parallax animation did not produce a video file.")
                output_path.write_bytes(temp_output.read_bytes())

        try:
            await asyncio.to_thread(_run_parallax)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Parallax animation failed: {exc}") from exc
        result_model = PARALLAX_MODEL_ID
    else:
        if genai is None:
            raise HTTPException(status_code=500, detail="google-genai package is not installed on the server.")
        if genai_types is None:
            raise HTTPException(status_code=500, detail="google-genai package is missing type definitions.")

        api_key = require_gemini_api_key()

        reference_image = None
        if image_data_url:
            parsed = _parse_data_url(image_data_url)
            if parsed:
                mime, b64_data = parsed
                try:
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except (binascii.Error, ValueError):  # noqa: PERF203 - decoded path is tiny
                    image_bytes = None
                if image_bytes:
                    reference_image = genai_types.Image(image_bytes=image_bytes, mime_type=mime)

        def _run_generation() -> None:
            """Invoke google.genai synchronously; runs in a worker thread."""

            client = genai.Client(api_key=api_key)
            request_payload: Dict[str, Any] = {
                "model": normalized_model,
                "prompt": prompt_text,
            }
            config: Dict[str, Any] = {}
            if negative_text:
                config["negative_prompt"] = negative_text
            if config:
                request_payload["config"] = config
            if reference_image is not None:
                request_payload["image"] = reference_image
            operation = client.models.generate_videos(**request_payload)

            while not getattr(operation, "done", False):
                time.sleep(10)
                operation = client.operations.get(operation)

            if getattr(operation, "error", None):
                message = getattr(operation.error, "message", None) or "Video generation failed."
                raise RuntimeError(message)

            response = getattr(operation, "response", None)
            generated_videos = getattr(response, "generated_videos", None) if response else None
            if not generated_videos:
                raise RuntimeError("No video returned by the model.")

            generated_video = generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(str(output_path))

        try:
            await asyncio.to_thread(_run_generation)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Video generation failed: {exc}") from exc

    static_root = APP_DIR / "static"
    try:
        relative_path = output_path.relative_to(static_root)
    except ValueError:
        try:
            relative_path = output_path.relative_to(GENERATED_MEDIA_DIR)
        except ValueError:
            url_path = f"/generated_media/{output_path.name}"
        else:
            url_path = f"/generated_media/{relative_path.as_posix()}"
    else:
        url_path = f"/static/{relative_path.as_posix()}"

    duration_seconds = _probe_mp4_duration_seconds(output_path)
    record_video_usage(result_model, seconds=duration_seconds, turn_index=turn_index)

    return SceneVideo(
        url=url_path,
        prompt=prompt_text,
        negative_prompt=negative_text or None,
        model=result_model,
        updated_at=time.time(),
        file_path=str(output_path),
    )


# -------------------- Turn engine --------------------
def build_turn_schema() -> Dict:
    # Keep schema compact (avoid 400 for complexity); mirrors TurnStructured
    return {
        "type": "OBJECT",
        "properties": {
            "nar": {"type": "STRING"},
            "img": {"type": "STRING"},
            "vid": {
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": "STRING"},
                    "negative_prompt": {"type": "STRING"},
                },
                "required": ["prompt"],
                "propertyOrdering": ["prompt", "negative_prompt"],
            },
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
        "propertyOrdering": ["nar", "img", "vid", "pub", "upd"],
    }


def build_thinking_directive() -> str:
    mode = (game_state.settings.get("thinking_mode") or "none").lower()
    if mode not in THINKING_MODES:
        mode = "none"
    lang = game_state.language if game_state.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
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
    lang = game_state.language if game_state.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    template = get_default_gm_prompt() if lang == DEFAULT_LANGUAGE else load_gm_prompt(lang)
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
    if not game_state.history:
        return []
    recent = game_state.history[-MAX_HISTORY_SUMMARY_BULLETS:]
    return [fallback_summarize_turn(rec) for rec in recent]


async def update_history_summary(latest_turn: TurnRecord) -> None:
    """Refresh the cached bullet summary after each processed turn."""
    summary_before = list(game_state.history_summary[-MAX_HISTORY_SUMMARY_BULLETS:])

    history_mode = normalize_history_mode(game_state.settings.get("history_mode"))
    if history_mode != HISTORY_MODE_SUMMARY:
        game_state.history_summary = _fallback_summary_from_history()
        return

    default_text_model = cast(str, DEFAULT_SETTINGS["text_model"])
    model = _get_setting_str(game_state.settings, "text_model", default=default_text_model) or default_text_model
    provider = detect_text_provider(model)
    try:
        require_text_api_key(provider)
    except HTTPException:
        game_state.history_summary = _fallback_summary_from_history()
        return

    players_snapshot = []
    for player in game_state.players.values():
        players_snapshot.append(
            {
                "id": player.id,
                "name": player.name,
                "cls": player.character_class,
                "status_word": player.status_word,
                "pending_join": player.pending_join,
                "pending_leave": player.pending_leave,
                "conditions": list(player.conditions),
                "inventory": list(player.inventory),
            }
        )

    payload = {
        "world_style": game_state.settings.get("world_style", "High fantasy"),
        "difficulty": game_state.settings.get("difficulty", "Normal"),
        "language": game_state.language,
        "turn_index": game_state.turn_index,
        "latest_turn": {
            "turn": latest_turn.index,
            "narrative": latest_turn.narrative,
            "image_prompt": latest_turn.image_prompt,
        },
        "previous_summary": summary_before,
        "players": players_snapshot,
        "departed_players": sorted(
            game_state.departed_players.values(),
            key=lambda value: value.casefold(),
        ),
        "max_bullets": MAX_HISTORY_SUMMARY_BULLETS,
    }

    try:
        result = await request_summary_payload(
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
        game_state.history_summary = lines[:MAX_HISTORY_SUMMARY_BULLETS]
    except Exception as exc:  # noqa: BLE001
        print(f"History summary update failed: {exc}", file=sys.stderr, flush=True)
        game_state.history_summary = _fallback_summary_from_history()


def build_turn_request_payload() -> Dict:
    history_mode = normalize_history_mode(game_state.settings.get("history_mode"))
    use_summary = history_mode == HISTORY_MODE_SUMMARY

    full_history = [
        {
            "turn": rec.index,
            "narrative": rec.narrative,
            "image_prompt": rec.image_prompt,
        }
        for rec in game_state.history
    ]

    summary_lines = list(game_state.history_summary)

    if use_summary:
        if not summary_lines and full_history:
            summary_lines = [
                fallback_summarize_turn(rec)
                for rec in game_state.history[-MAX_HISTORY_SUMMARY_BULLETS:]
            ]
        history_payload: Any = summary_lines
    else:
        history_payload = full_history

    players = {
        pid: {
            "name": p.name,
            "background": p.background,
            "cls": p.character_class,
            # Use Pydantic's serializer so we don't crash once abilities are populated.
            "ab": [
                (
                    a.model_dump(by_alias=True)
                    if isinstance(a, Ability)
                    else Ability.model_validate(a).model_dump(by_alias=True)
                )
                for a in p.abilities
            ],
            "inv": p.inventory,
            "cond": p.conditions,
            "status_word": p.status_word,
            "pending_join": p.pending_join,
            "pending_leave": p.pending_leave,
        }
        for pid, p in game_state.players.items()
    }

    lang = game_state.language if game_state.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    payload = {
        "world_style": game_state.settings.get("world_style", "High fantasy"),
        "difficulty": game_state.settings.get("difficulty", "Normal"),
        "turn_index": game_state.turn_index,
        "history": history_payload,
        "history_mode": history_mode,
        "history_summary": summary_lines,
        "players": players,
        "departed_players": sorted(
            game_state.departed_players.values(),
            key=lambda value: value.casefold(),
        ),
        "submissions": game_state.submissions,  # {player_id: "action text"}
        "language": lang,
        "note": USER_PAYLOAD_NOTES.get(lang, USER_PAYLOAD_NOTES[DEFAULT_LANGUAGE]),
    }
    return payload


def build_portrait_prompt(player: Player) -> str:
    """Create a deterministic, stylized prompt for a player's portrait using the configured world style.

    Also mentions world style explicitly, visible equipment from the character's inventory,
    and (when present) any current conditions.
    """
    descriptors: List[str] = []
    cls = player.character_class.strip() if player.character_class else ""
    background = player.background.strip() if player.background else ""

    if cls:
        descriptors.append(cls)
    if background and background.lower() != cls.lower():
        descriptors.append(background)

    world_style = game_state.settings.get("world_style", "High fantasy")
    tone_bits: List[str] = []
    status_word = player.status_word.strip().lower() if player.status_word else ""
    if status_word and status_word not in {"unknown", ""}:
        tone_bits.append(f"mood of {status_word}")
    if player.pending_join:
        tone_bits.append("freshly joined hero")

    tone_text = ", ".join(tone_bits)
    descriptor_text = ", ".join(descriptors) if descriptors else "adventurer"

    # Prepare concise equipment and condition mentions
    inv_items = [i.strip() for i in (player.inventory or []) if isinstance(i, str) and i.strip()]
    cond_items = [c.strip() for c in (player.conditions or []) if isinstance(c, str) and c.strip()]

    prompt = (
        f"Photorealistic detailed portrait of {player.name}, a {descriptor_text} from a {world_style} setting. "
        f"Style: {world_style}. Centered head and shoulders. Square format."
    )

    if inv_items:
        shown = ", ".join(inv_items[:6])
        prompt += f" Include visible equipment from inventory where appropriate: {shown}."
    if cond_items:
        shown_c = ", ".join(cond_items[:6])
        prompt += f" Reflect current conditions if visually apparent: {shown_c}."

    if tone_text:
        prompt += f" Convey a {tone_text}."

    return prompt


async def resolve_turn(initial: bool = False) -> None:
    pending_before: Set[str] = set()
    leaving_before: Set[str] = set()
    async with STATE_LOCK:
        if game_state.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        game_state.lock = LockState(active=True, reason="resolving_turn")
        await broadcast_public()
        pending_before = {pid for pid, p in game_state.players.items() if p.pending_join}
        leaving_before = {pid for pid, p in game_state.players.items() if p.pending_leave}
        game_state.last_token_usage = {}
        game_state.last_turn_runtime = None
        game_state.last_cost_usd = None

    try:
        # Build prompt + schema and call the selected text model once
        schema = build_turn_schema()
        system_text = make_gm_instruction(is_initial=initial)
        payload = build_turn_request_payload()
        default_text_model = cast(str, DEFAULT_SETTINGS["text_model"])
        model = _get_setting_str(game_state.settings, "text_model", default=default_text_model) or default_text_model

        result: TurnStructured = await request_turn_payload(
            model=model,
            system_prompt=system_text,
            user_payload=payload,
            schema=schema
        )

        # Apply updates
        for upd in result.updates:
            pid = upd.player_id
            if pid not in game_state.players:
                # Ignore unknown ids; model must stick to provided ids
                continue
            p = game_state.players[pid]
            p.character_class = upd.character_class
            p.abilities = [Ability.model_validate(a) for a in upd.abilities]
            p.inventory = list(upd.inventory)
            p.conditions = list(upd.conditions)
            p.pending_join = False

        # Update public statuses
        for status in result.public_statuses or []:
            pid = status.player_id
            if pid in game_state.players:
                raw = (status.status_word or "").strip()
                first_word = raw.split()[0] if raw else "unknown"
                game_state.players[pid].status_word = first_word.lower()

        # Commit scenario + history
        narrative_text = sanitize_narrative(result.narrative)
        image_prompt = result.image_prompt
        raw_video = getattr(result, "video", None)
        video_prompt: Optional[str] = None
        video_negative_prompt: Optional[str] = None

        if isinstance(raw_video, VideoPromptStructured):
            video_prompt = (raw_video.prompt or "").strip() or None
            neg = (raw_video.negative_prompt or "").strip() if raw_video.negative_prompt else ""
            video_negative_prompt = neg or None
        elif isinstance(raw_video, str):
            video_prompt = raw_video.strip() or None
        elif isinstance(raw_video, dict):
            prompt_value = str(raw_video.get("prompt", "")) if raw_video.get("prompt") is not None else ""
            video_prompt = prompt_value.strip() or None
            neg_value = raw_video.get("negative_prompt")
            if neg_value is None and "negativePrompt" in raw_video:
                neg_value = raw_video.get("negativePrompt")
            if isinstance(neg_value, str):
                neg_value = neg_value.strip()
                video_negative_prompt = neg_value or None

        current_turn_index = game_state.turn_index
        game_state.current_narrative = narrative_text
        game_state.last_image_prompt = image_prompt
        game_state.last_video_prompt = video_prompt or image_prompt
        game_state.last_video_negative_prompt = video_negative_prompt if video_prompt else None
        game_state.last_manual_scene_image_turn_index = None
        if isinstance(current_turn_index, int):
            current_bucket = _bind_turn_image_bucket(current_turn_index)
            current_bucket.clear()
        else:
            game_state.current_turn_image_counts = {}
            game_state.current_turn_index_for_image_counts = None

        rec = TurnRecord(
            index=current_turn_index,
            narrative=narrative_text,
            image_prompt=image_prompt,
            timestamp=time.time(),
        )
        game_state.history.append(rec)

        await update_history_summary(rec)

        # Clear submissions for next turn
        game_state.submissions.clear()

        # Remove players who were marked for departure before the turn and
        # remain pending_leave after narrative send-off.
        departed_now: List[str] = []
        for pid in list(game_state.players.keys()):
            player = game_state.players[pid]
            if player.pending_leave and pid in leaving_before:
                departed_now.append(pid)
                normalized_name = normalize_player_name(player.name)
                if normalized_name:
                    game_state.departed_players[normalized_name] = player.name
                game_state.players.pop(pid, None)

        reset_occurred = reset_session_if_inactive()

        joined_now: List[str] = []
        if not reset_occurred:
            # Turn advances AFTER applying
            game_state.turn_index += 1
            new_bucket = _bind_turn_image_bucket(game_state.turn_index)
            new_bucket.clear()

            await schedule_auto_tts(narrative_text or "", current_turn_index)

            # Inform everyone about joiners
            joined_now = [
                pid
                for pid in pending_before
                if pid in game_state.players and not game_state.players[pid].pending_join
            ]
            if not initial and joined_now:
                lang = game_state.language if game_state.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
                message = ANNOUNCEMENTS["new_player"].get(lang) or ANNOUNCEMENTS["new_player"][DEFAULT_LANGUAGE]
                await announce(message)

        # Clean up any lingering submissions for removed players (if turn advanced before clear)
        for pid in departed_now:
            game_state.submissions.pop(pid, None)

        # Push updated states
        await broadcast_public()
        # Private slices
        if not reset_occurred:
            for pid in list(game_state.players.keys()):
                await send_private(pid)
            await schedule_auto_scene_image(image_prompt, current_turn_index, force=False)
            await schedule_auto_scene_video(game_state.last_video_prompt, current_turn_index, force=False)

    finally:
        game_state.lock = LockState(active=False, reason="")
        await broadcast_public()


# -------------------- FastAPI app --------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    game_state.settings = load_settings()
    ensure_settings_file()
    game_state.language = normalize_language(game_state.settings.get("language"))
    yield


app = FastAPI(title="Nils' RPG", lifespan=lifespan)
static_dir = APP_DIR / "static"
static_mount_args: Dict[str, Any] = {"directory": str(static_dir)}
if not static_dir.exists():
    static_mount_args["check_dir"] = False
app.mount("/static", StaticFiles(**static_mount_args), name="static")

generated_media_dir = GENERATED_MEDIA_DIR
generated_mount_args: Dict[str, Any] = {"directory": str(generated_media_dir), "check_dir": False}
app.mount("/generated_media", StaticFiles(**generated_mount_args), name="generated_media")


# --------- Static root ---------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = APP_DIR / "static" / "index.html"
    if index_file.exists():
        html = index_file.read_text(encoding="utf-8")
    else:
        # Provide a minimal placeholder page when the bundled UI is missing.
        html = "<!doctype html><title>RPG</title><p>UI not found.</p>"
    return HTMLResponse(html, status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    favicon_file = APP_DIR / "static" / "favicon.ico"
    if not favicon_file.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(str(favicon_file))


# --------- Settings ---------
@app.get("/api/settings")
async def get_settings() -> Dict[str, Any]:
    return game_state.settings.copy()


@app.get("/api/public_url")
async def get_public_url(request: Request) -> Dict[str, str]:
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


@app.get("/api/join_backgrounds")
async def get_join_backgrounds() -> Dict[str, List[str]]:
    images_dir = static_dir / "img"
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    backgrounds: List[str] = []
    if images_dir.exists():
        candidates = [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_suffixes
        ]
        for path in sorted(candidates, key=lambda p: p.name.lower()):
            backgrounds.append(f"/static/img/{path.name}")
    return {"backgrounds": backgrounds}


@app.get("/api/join_songs")
async def get_join_songs() -> Dict[str, List[Dict[str, str]]]:
    songs_dir = static_dir / "songs"
    songs: List[Dict[str, str]] = []
    if songs_dir.exists():
        candidates = [p for p in songs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
        for path in sorted(candidates, key=lambda p: p.name.lower()):
            stem = path.stem.strip()
            pretty = re.sub(r"[_\-]+", " ", stem).strip()
            title = pretty.title() if pretty else path.name
            songs.append({
                "id": stem or path.name,
                "src": f"/static/songs/{path.name}",
                "title": title,
            })
    return {"songs": songs}


@app.get("/api/dev/text_inspect")
async def get_dev_text_inspect() -> Dict[str, Any]:
    async with STATE_LOCK:
        request_payload = copy.deepcopy(game_state.last_text_request)
        response_payload = copy.deepcopy(game_state.last_text_response)
        turn_request = copy.deepcopy(game_state.last_turn_request)
        turn_response = copy.deepcopy(game_state.last_turn_response)
        history_mode = normalize_history_mode(game_state.settings.get("history_mode"))
    return {
        "request": request_payload,
        "response": response_payload,
        "turn_request": turn_request,
        "turn_response": turn_response,
        "history_mode": history_mode,
    }


class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = Field(default=None)
    grok_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    world_style: Optional[str] = None
    difficulty: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    video_duration_seconds: Optional[int] = None
    narration_model: Optional[str] = None
    thinking_mode: Optional[str] = None
    history_mode: Optional[str] = None


@app.put("/api/settings")
async def update_settings(body: SettingsUpdate) -> Dict[str, Any]:
    global _ELEVENLABS_API_KEY_WARNING_LOGGED
    changed = False
    for k in [
        "world_style",
        "difficulty",
        "text_model",
        "image_model",
        "video_model",
        "video_duration_seconds",
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
                game_state.settings[k] = mode
            elif k == "video_model":
                model_val = str(v).strip()
                if not model_val:
                    continue
                game_state.settings[k] = model_val
            elif k == "video_duration_seconds":
                try:
                    duration_val = _normalize_video_duration_seconds(v)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="video_duration_seconds must be between 1 and 120 seconds.",
                    ) from exc
                game_state.settings[k] = duration_val
            elif k == "narration_model":
                model_val = str(v).strip()
                if not model_val:
                    continue
                game_state.settings[k] = model_val
            elif k == "history_mode":
                game_state.settings[k] = normalize_history_mode(v)
            else:
                game_state.settings[k] = v
            changed = True
    # API keys are optional but saved immediately when provided
    if body.gemini_api_key is not None:
        game_state.settings["gemini_api_key"] = body.gemini_api_key.strip()
        changed = True
    if body.grok_api_key is not None:
        game_state.settings["grok_api_key"] = body.grok_api_key.strip()
        changed = True
    if body.openai_api_key is not None:
        game_state.settings["openai_api_key"] = body.openai_api_key.strip()
        changed = True
    if body.elevenlabs_api_key is not None:
        game_state.settings["elevenlabs_api_key"] = body.elevenlabs_api_key.strip()
        _ELEVENLABS_API_KEY_WARNING_LOGGED = False
        changed = True
    if changed:
        await save_settings(game_state.settings)
    return {"ok": True}


# --------- Models list ---------
@app.get("/api/models")
async def api_models() -> Dict[str, Any]:
    items: List[ProviderModelInfo] = []
    errors: List[HTTPException] = []

    gemini_key = (
        game_state.settings.get("gemini_api_key")
        or game_state.settings.get("api_key")
        or ""
    ).strip()
    if gemini_key:
        try:
            items.extend(await gemini_list_models())
        except HTTPException as exc:  # propagate later if nothing else succeeds
            errors.append(exc)

    grok_key = (game_state.settings.get("grok_api_key") or "").strip()
    if grok_key:
        try:
            items.extend(await grok_list_models(grok_key))
        except HTTPException as exc:
            errors.append(exc)

    openai_key = (game_state.settings.get("openai_api_key") or "").strip()
    if openai_key:
        try:
            items.extend(await openai_list_models(openai_key))
        except HTTPException as exc:
            errors.append(exc)

    if not items and errors:
        raise errors[0]

    items.sort(
        key=lambda entry: (
            str(entry.get("displayName") or entry.get("name") or "").lower(),
            str(entry.get("name") or "").lower(),
        )
    )

    narration_models: List[ElevenLabsModelInfo] = []
    api_key = (game_state.settings.get("elevenlabs_api_key") or "").strip()
    if api_key:
        narration_models = await elevenlabs_list_models(api_key)
    return {"models": items, "narration_models": narration_models}


# --------- Join / Leave / State ---------
class LanguageBody(BaseModel):
    language: str
    player_id: Optional[str] = None
    token: Optional[str] = None


@app.post("/api/language")
async def set_language(body: LanguageBody) -> Dict[str, str]:
    if body.player_id and body.token:
        authenticate_player(body.player_id, body.token)
    changed = set_language_if_changed(body.language)
    if changed:
        await save_settings(game_state.settings)
        await broadcast_public()
    return {"language": game_state.language}


class JoinBody(BaseModel):
    name: Optional[str] = "Hephaest"
    background: Optional[str] = "Wizard"
    language: Optional[str] = None


@app.post("/api/join")
async def join_game(body: JoinBody) -> Dict[str, str]:
    cancel_pending_reset_task()
    if game_state.players:
        reset_session_if_inactive()
    set_language_if_changed(body.language)
    pid = secrets.token_hex(8)
    name = (body.name or "Hephaest").strip()[:40]
    background = (body.background or "Wizard").strip()[:200]
    token = secrets.token_hex(16)
    normalized_name = normalize_player_name(name)
    if normalized_name:
        game_state.departed_players.pop(normalized_name, None)
    p = Player(id=pid, name=name, background=background, pending_join=True, token=token)
    game_state.players[pid] = p

    # If this is the very first player and world not started -> run initial world-gen immediately
    if game_state.turn_index == 0 and not game_state.current_narrative:
        try:
            await announce(f"{name} is starting a new world…")
            await resolve_turn(initial=True)
        except Exception:
            # Undo the player registration so a failed turn doesn't leave ghosts around.
            game_state.players.pop(pid, None)
            await broadcast_public()
            raise

    await broadcast_public()
    return {"player_id": pid, "auth_token": token}


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return game_state.public_snapshot()


class SubmitBody(BaseModel):
    player_id: str
    token: str
    text: str
    language: Optional[str] = None


@app.post("/api/submit")
async def submit_action(body: SubmitBody) -> Dict[str, bool]:
    if game_state.lock.active:
        raise HTTPException(status_code=409, detail="Game is busy. Try again in a moment.")
    player = authenticate_player(body.player_id, body.token)
    set_language_if_changed(body.language)
    game_state.submissions[player.id] = body.text.strip()[:1000]
    await broadcast_public()
    return {"ok": True}


class NextTurnBody(BaseModel):
    player_id: str
    token: str
    language: Optional[str] = None


@app.post("/api/next_turn")
async def next_turn(body: NextTurnBody) -> Dict[str, bool]:
    # Anyone can advance the turn per spec
    authenticate_player(body.player_id, body.token)
    set_language_if_changed(body.language)
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
async def toggle_tts(body: ToggleTtsBody) -> Dict[str, bool]:
    authenticate_player(body.player_id, body.token)
    if body.enabled:
        api_key = (game_state.settings.get("elevenlabs_api_key") or "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Set the ElevenLabs API key in Settings before enabling narration.",
            )
        if not _elevenlabs_library_available():
            raise HTTPException(
                status_code=500,
                detail="elevenlabs package is not installed on the server.",
            )
    turn_index: Optional[int] = None
    async with STATE_LOCK:
        game_state.auto_tts_enabled = bool(body.enabled)
        if game_state.auto_tts_enabled:
            turn_index = game_state.history[-1].index if game_state.history else game_state.turn_index
    await broadcast_public()
    auto_tts_enabled = game_state.auto_tts_enabled
    if auto_tts_enabled and turn_index is not None:
        await schedule_auto_tts(game_state.current_narrative or "", turn_index)
    return {"auto_tts_enabled": auto_tts_enabled}


@app.post("/api/image_toggle")
async def toggle_scene_image(body: ToggleSceneImageBody) -> Dict[str, bool]:
    authenticate_player(body.player_id, body.token)
    turn_index: Optional[int] = None
    async with STATE_LOCK:
        game_state.auto_image_enabled = bool(body.enabled)
        if game_state.auto_image_enabled:
            game_state.auto_video_enabled = False
            turn_index = game_state.history[-1].index if game_state.history else game_state.turn_index
    await broadcast_public()
    auto_image_enabled = game_state.auto_image_enabled
    auto_video_enabled = game_state.auto_video_enabled
    if auto_image_enabled and turn_index is not None:
        await schedule_auto_scene_image(game_state.last_image_prompt, turn_index, force=True)
    return {
        "auto_image_enabled": auto_image_enabled,
        "auto_video_enabled": auto_video_enabled,
    }


class ToggleSceneVideoBody(BaseModel):
    player_id: str
    token: str
    enabled: bool


@app.post("/api/video_toggle")
async def toggle_scene_video(body: ToggleSceneVideoBody) -> Dict[str, bool]:
    authenticate_player(body.player_id, body.token)
    turn_index: Optional[int] = None
    async with STATE_LOCK:
        game_state.auto_video_enabled = bool(body.enabled)
        if game_state.auto_video_enabled:
            game_state.auto_image_enabled = False
            turn_index = game_state.history[-1].index if game_state.history else game_state.turn_index
    await broadcast_public()
    auto_video_enabled = game_state.auto_video_enabled
    auto_image_enabled = game_state.auto_image_enabled
    if auto_video_enabled and turn_index is not None:
        prompt = game_state.last_video_prompt or game_state.last_image_prompt
        await schedule_auto_scene_video(prompt, turn_index, force=True)
    return {
        "auto_video_enabled": auto_video_enabled,
        "auto_image_enabled": auto_image_enabled,
    }


class CreateImageBody(BaseModel):
    player_id: str
    token: str


@app.post("/api/create_image")
async def create_image(body: CreateImageBody) -> Dict[str, Any]:
    target_turn_index: Optional[int] = None
    async with STATE_LOCK:
        if game_state.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        authenticate_player(body.player_id, body.token)
        # Need an image prompt from the latest turn
        if not game_state.last_image_prompt:
            raise HTTPException(status_code=400, detail="No image prompt available yet.")
        if game_state.history:
            target_turn_index = game_state.history[-1].index
        elif isinstance(game_state.turn_index, int):
            target_turn_index = game_state.turn_index
        game_state.lock = LockState(active=True, reason="generating_image")
        await broadcast_public()

    try:
        img_model = game_state.settings.get("image_model") or "gemini-2.5-flash-image-preview"
        data_url = await gemini_generate_image(
            img_model,
            game_state.last_image_prompt,
            purpose="scene",
            turn_index=target_turn_index,
        )
        _clear_scene_video()
        game_state.last_image_data_url = data_url
        game_state.last_manual_scene_image_turn_index = target_turn_index
        await announce("Image generated.")
        await broadcast_public()
        return {"ok": True}
    finally:
        game_state.lock = LockState(active=False, reason="")
        await broadcast_public()


class AnimateSceneBody(BaseModel):
    player_id: str
    token: str


@app.post("/api/animate_scene")
async def animate_scene(body: AnimateSceneBody) -> Dict[str, Any]:
    async with STATE_LOCK:
        if game_state.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        authenticate_player(body.player_id, body.token)
        # Prefer the dedicated video prompt if available
        prompt = ((game_state.last_video_prompt or game_state.last_image_prompt) or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="No video/image prompt available yet.")
        game_state.lock = LockState(active=True, reason="generating_video")
        await broadcast_public()

    try:
        negative_prompt = game_state.last_video_negative_prompt if game_state.last_video_prompt else None
        if game_state.history:
            target_turn_index = game_state.history[-1].index
        elif isinstance(game_state.turn_index, int):
            target_turn_index = game_state.turn_index
        else:
            target_turn_index = None
        image_for_video: Optional[str] = None
        if (
            target_turn_index is not None
            and game_state.last_manual_scene_image_turn_index == target_turn_index
            and game_state.last_image_turn_index == target_turn_index
            and game_state.last_image_data_url
        ):
            image_for_video = game_state.last_image_data_url
        new_video = await generate_scene_video(
            prompt,
            game_state.settings.get("video_model"),
            image_data_url=image_for_video,
            negative_prompt=negative_prompt,
            turn_index=target_turn_index,
        )
        _clear_scene_video()
        game_state.scene_video = new_video
        game_state.last_video_prompt = prompt
        game_state.last_video_negative_prompt = negative_prompt
        if target_turn_index is not None:
            game_state.last_scene_video_turn_index = target_turn_index
        elif game_state.history:
            game_state.last_scene_video_turn_index = game_state.history[-1].index
        elif isinstance(game_state.turn_index, int):
            game_state.last_scene_video_turn_index = game_state.turn_index
        else:
            game_state.last_scene_video_turn_index = None
        await announce("Video generated.")
        await broadcast_public()
        return {"ok": True, "video": scene_video_payload(new_video)}
    finally:
        game_state.lock = LockState(active=False, reason="")
        await broadcast_public()


class CreatePortraitBody(BaseModel):
    player_id: str
    token: str


@app.post("/api/create_portrait")
async def create_portrait(body: CreatePortraitBody) -> Dict[str, Any]:
    async with STATE_LOCK:
        if game_state.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        player = authenticate_player(body.player_id, body.token)
        portrait_prompt = build_portrait_prompt(player)
        game_state.lock = LockState(active=True, reason="generating_portrait")
        await broadcast_public()

    try:
        img_model = game_state.settings.get("image_model") or "gemini-2.5-flash-image-preview"
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
        game_state.lock = LockState(active=False, reason="")
        await broadcast_public()


# --------- WebSockets ---------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    # Optional player_id supplied as query for private channel
    player_id = ws.query_params.get("player_id")
    auth_token = ws.query_params.get("auth_token")
    game_state.global_sockets.add(ws)
    authed_player = None
    if player_id and auth_token:
        candidate = game_state.players.get(player_id)
        if candidate and candidate.token == auth_token:
            authed_player = candidate
            authed_player.sockets.add(ws)
            authed_player.connected = True
            authed_player.pending_leave = False
            cancel_pending_reset_task()
    # Send initial snapshots
    await ws.send_json({"event": "state", "data": game_state.public_snapshot()})
    if authed_player:
        await ws.send_json({"event": "private", "data": game_state.private_snapshot_for(authed_player.id)})

    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=WEBSOCKET_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                # No inbound frames; treat as still alive so heartbeat-driven logic can continue.
                continue

            message_type = message.get("type") if isinstance(message, dict) else None
            if message_type in {"websocket.disconnect", "websocket.close"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup
        game_state.global_sockets.discard(ws)
        leave_player: Optional[Player] = None
        if authed_player and authed_player.id in game_state.players:
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
            sock4.bind(("0.0.0.0", bind_port))  # nosec B104
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
