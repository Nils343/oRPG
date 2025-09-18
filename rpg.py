# app.py
# Multiplayer text RPG server using FastAPI + WebSockets
# - settings.json (same folder) stores API key, world style, difficulty, and model choices
# - Uses Google Gemini API via REST (httpx)
# - One text-generation call per turn (after initial world gen) with structured JSON output
# - Optional image generation per turn via gemini-2.5-flash-image-preview

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).parent
SETTINGS_FILE = APP_DIR / "settings.json"
PROMPT_FILE = APP_DIR / "gm_prompt.txt"
TURN_DIRECTIVE_TOKEN = "<<TURN_DIRECTIVE>>"


def load_gm_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise RuntimeError(f"Missing GM prompt template: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


GM_PROMPT_TEMPLATE = load_gm_prompt()

# -------- Defaults --------
DEFAULT_SETTINGS = {
    "api_key": "",
    "text_model": "gemini-2.5-flash-lite",
    "image_model": "gemini-2.5-flash-image-preview",
    "world_style": "High fantasy",
    "difficulty": "Normal",  # Trivial, Easy, Normal, Hard, Impossible
}

# -------- Gemini endpoints (REST) --------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELS_LIST_URL = f"{GEMINI_BASE}/models"
GENERATE_CONTENT_URL = f"{GEMINI_BASE}/models/{{model}}:generateContent"


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
    """Structured output we expect from gemini-2.5-flash-lite per turn."""
    nar: str  # narrative for the next scenario
    img: str  # image prompt for gemini-2.5-flash-image-preview
    pub: List[PublicStatus]
    upd: List[PlayerUpdate]


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
    # WebSocket connections for private pushes
    sockets: Set[WebSocket] = field(default_factory=set, repr=False, compare=False)


@dataclass
class TurnRecord:
    index: int
    narrative: str
    image_prompt: str
    timestamp: float


@dataclass
class LockState:
    active: bool = False
    reason: str = ""  # "resolving_turn" | "generating_image"


@dataclass
class GameState:
    settings: Dict = field(default_factory=lambda: DEFAULT_SETTINGS.copy())
    players: Dict[str, Player] = field(default_factory=dict)  # player_id -> Player
    submissions: Dict[str, str] = field(default_factory=dict)  # player_id -> text
    current_scenario: str = ""
    turn_index: int = 0
    history: List[TurnRecord] = field(default_factory=list)
    lock: LockState = field(default_factory=LockState)
    global_sockets: Set[WebSocket] = field(default_factory=set, repr=False, compare=False)
    # last generated image
    last_image_data_url: Optional[str] = None
    last_image_prompt: Optional[str] = None

    def public_snapshot(self) -> Dict:
        """Sanitized state for all players."""
        return {
            "turn_index": self.turn_index,
            "current_scenario": self.current_scenario,
            "world_style": self.settings.get("world_style", "High fantasy"),
            "difficulty": self.settings.get("difficulty", "Normal"),
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "status_word": p.status_word,
                    "connected": p.connected,
                    "pending_join": p.pending_join,
                }
                for p in self.players.values()
            ],
            "submissions": [
                {"player_id": pid, "name": self.players.get(pid).name if pid in self.players else "Unknown", "text": txt}
                for pid, txt in self.submissions.items()
            ],
            "lock": {"active": self.lock.active, "reason": self.lock.reason},
            "image": {
                "data_url": self.last_image_data_url,
                "prompt": self.last_image_prompt,
            },
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
                "abilities": [a.model_dump() for a in p.abilities],
                "inventory": p.inventory,
                "conditions": p.conditions,
            }
        }


STATE = GameState()
STATE_LOCK = asyncio.Lock()  # coarse lock for turn/image operations
SETTINGS_LOCK = asyncio.Lock()  # for reading/writing settings.json


# -------------------- Helpers: settings I/O --------------------
def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def load_settings() -> Dict:
    ensure_settings_file()
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_SETTINGS.copy()


async def save_settings(new_settings: Dict):
    async with SETTINGS_LOCK:
        SETTINGS_FILE.write_text(json.dumps(new_settings, indent=2), encoding="utf-8")


# -------------------- Helpers: sockets --------------------
async def broadcast_public():
    payload = {"event": "state", "data": STATE.public_snapshot()}
    dead = []
    for ws in list(STATE.global_sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.global_sockets.discard(ws)


async def send_private(player_id: str):
    payload = {"event": "private", "data": STATE.private_snapshot_for(player_id)}
    p = STATE.players.get(player_id)
    if not p:
        return
    dead = []
    for ws in list(p.sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        p.sockets.discard(ws)


async def announce(message: str):
    payload = {"event": "announce", "data": {"message": message, "ts": time.time()}}
    dead = []
    for ws in list(STATE.global_sockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.global_sockets.discard(ws)


# -------------------- Helpers: Gemini REST --------------------
def check_api_key() -> str:
    api_key = STATE.settings.get("api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is not set in settings.")
    return api_key


async def gemini_list_models() -> List[Dict]:
    api_key = check_api_key()
    url = f"{MODELS_LIST_URL}?key={api_key}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Model list failed: {r.text}")
        data = r.json()
        return data.get("models", [])


async def gemini_generate_json(model: str, system_prompt: str, user_payload: Dict, schema: Dict) -> TurnStructured:
    """Calls generateContent with forced JSON schema; returns parsed TurnStructured."""
    api_key = check_api_key()
    url = GENERATE_CONTENT_URL.format(model=model)
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": json.dumps(user_payload)}]},
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.9,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Text generation failed: {r.text}")
        data = r.json()
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
    try:
        return TurnStructured.model_validate(parsed)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Schema validation error: {e}")


async def gemini_generate_image(model: str, prompt: str) -> str:
    """Returns a data URL (base64 image) from gemini-2.5-flash-image-preview."""
    api_key = check_api_key()
    url = GENERATE_CONTENT_URL.format(model=model)
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Image generation failed: {r.text}")
        data = r.json()

    # Extract first inline image part
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for prt in parts:
        inline = prt.get("inlineData") or prt.get("inline_data")
        if inline and inline.get("data"):
            b64 = inline["data"]
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
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


def make_gm_instruction(is_initial: bool) -> str:
    directive = (
        "INITIAL TURN: Create the opening scenario AND initial characters (cls/ab/inv/cond) for every joining player.\n"
        if is_initial
        else "ONGOING TURN: Incorporate any pending joiners naturally into the scene in nar, while also resolving all actions from current players.\n"
    )

    if TURN_DIRECTIVE_TOKEN in GM_PROMPT_TEMPLATE:
        return GM_PROMPT_TEMPLATE.replace(TURN_DIRECTIVE_TOKEN, directive)

    template = GM_PROMPT_TEMPLATE.rstrip("\n")
    separator = "\n" if template else ""
    return f"{template}{separator}{directive}"


def compile_user_payload() -> Dict:
    # Full history is included as requested; if very long, the model may internally summarize.
    history = [
        {
            "turn": rec.index,
            "narrative": rec.narrative,
            "image_prompt": rec.image_prompt,
        } for rec in STATE.history
    ]

    players = {
        pid: {
            "name": p.name,
            "background": p.background,
            "cls": p.cls,
            "ab": [asdict(a) for a in p.abilities],
            "inv": p.inventory,
            "cond": p.conditions,
            "status_word": p.status_word,
            "pending_join": p.pending_join,
        }
        for pid, p in STATE.players.items()
    }

    payload = {
        "world_style": STATE.settings.get("world_style", "High fantasy"),
        "difficulty": STATE.settings.get("difficulty", "Normal"),
        "turn_index": STATE.turn_index,
        "history": history,
        "players": players,
        "submissions": STATE.submissions,  # {player_id: "action text"}
        "note": (
            "Players only see their own abilities/inventory/conditions; others see one-word status only. "
            "Update 'pub' for everyone each turn. For 'upd', ensure all pending_join players get full initial kits."
        ),
    }
    return payload


async def resolve_turn(initial: bool = False):
    pending_before: Set[str] = set()
    async with STATE_LOCK:
        if STATE.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        STATE.lock = LockState(True, "resolving_turn")
        await broadcast_public()
        pending_before = {pid for pid, p in STATE.players.items() if p.pending_join}

    try:
        # Build prompt + schema and call Gemini once
        schema = build_turn_schema()
        system_text = make_gm_instruction(is_initial=initial)
        payload = compile_user_payload()
        model = STATE.settings.get("text_model") or "gemini-2.5-flash-lite"

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
            word = status.word
            if pid in STATE.players:
                STATE.players[pid].status_word = (word or "unknown").split()[0].strip().lower()

        # Commit scenario + history
        STATE.current_scenario = result.nar
        STATE.last_image_prompt = result.img
        rec = TurnRecord(
            index=STATE.turn_index,
            narrative=result.nar,
            image_prompt=result.img,
            timestamp=time.time(),
        )
        STATE.history.append(rec)

        # Clear submissions for next turn
        STATE.submissions.clear()

        # Turn advances AFTER applying
        STATE.turn_index += 1

        # Inform everyone about joiners
        joined_now = [
            pid
            for pid in pending_before
            if pid in STATE.players and not STATE.players[pid].pending_join
        ]
        if not initial and joined_now:
            await announce("A new player has joined the party.")

        # Push updated states
        await broadcast_public()
        # Private slices
        for pid in list(STATE.players.keys()):
            await send_private(pid)

    finally:
        STATE.lock = LockState(False, "")
        await broadcast_public()


# -------------------- FastAPI app --------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.settings = load_settings()
    ensure_settings_file()
    yield


app = FastAPI(title="LAN RPG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


# --------- Static root ---------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(APP_DIR / "static" / "index.html"))


# --------- Settings ---------
@app.get("/api/settings")
async def get_settings():
    # Do not leak full key; send a masked preview to client
    s = STATE.settings.copy()
    if s.get("api_key"):
        s["api_key_preview"] = s["api_key"][:6] + "…" + s["api_key"][-4:]
        s["api_key_set"] = True
        del s["api_key"]
    else:
        s["api_key_set"] = False
    return s


class SettingsUpdate(BaseModel):
    api_key: Optional[str] = None
    world_style: Optional[str] = None
    difficulty: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None


@app.put("/api/settings")
async def update_settings(body: SettingsUpdate):
    changed = False
    for k in ["world_style", "difficulty", "text_model", "image_model"]:
        v = getattr(body, k)
        if v is not None:
            STATE.settings[k] = v
            changed = True
    # API key is optional but if provided we save immediately
    if body.api_key is not None:
        STATE.settings["api_key"] = body.api_key.strip()
        changed = True
    if changed:
        await save_settings(STATE.settings)
    return {"ok": True}


# --------- Models list ---------
@app.get("/api/models")
async def api_models():
    models = await gemini_list_models()
    # Return bare essentials for dropdowns (no fallback; raw from server)
    items = []
    for m in models:
        items.append({
            "name": m.get("name", ""),
            "displayName": m.get("displayName") or m.get("description") or m.get("name", ""),
            "supported": m.get("supportedGenerationMethods") or m.get("supported_actions") or [],
        })
    return {"models": items}


# --------- Join / Leave / State ---------
class JoinBody(BaseModel):
    name: Optional[str] = "Hephaest"
    background: Optional[str] = "Wizard"


@app.post("/api/join")
async def join_game(body: JoinBody):
    pid = secrets.token_hex(8)
    name = (body.name or "Hephaest").strip()[:40]
    background = (body.background or "Wizard").strip()[:200]
    p = Player(id=pid, name=name, background=background, pending_join=True, connected=True)
    STATE.players[pid] = p

    # If this is the very first player and world not started -> run initial world-gen immediately
    if STATE.turn_index == 0 and not STATE.current_scenario:
        await announce(f"{name} is starting a new world…")
        await resolve_turn(initial=True)

    await broadcast_public()
    return {"player_id": pid}


@app.get("/api/state")
async def get_state():
    return STATE.public_snapshot()


class SubmitBody(BaseModel):
    player_id: str
    text: str


@app.post("/api/submit")
async def submit_action(body: SubmitBody):
    if STATE.lock.active:
        raise HTTPException(status_code=409, detail="Game is busy. Try again in a moment.")
    pid = body.player_id
    if pid not in STATE.players:
        raise HTTPException(status_code=404, detail="Unknown player.")
    STATE.submissions[pid] = body.text.strip()[:1000]
    await broadcast_public()
    return {"ok": True}


class NextTurnBody(BaseModel):
    player_id: str


@app.post("/api/next_turn")
async def next_turn(body: NextTurnBody):
    # Anyone can advance the turn per spec
    await resolve_turn(initial=False)
    return {"ok": True}


class CreateImageBody(BaseModel):
    player_id: str


@app.post("/api/create_image")
async def create_image(body: CreateImageBody):
    async with STATE_LOCK:
        if STATE.lock.active:
            raise HTTPException(status_code=409, detail="Another operation is in progress.")
        # Need an image prompt from the latest turn
        if not STATE.last_image_prompt:
            raise HTTPException(status_code=400, detail="No image prompt available yet.")
        STATE.lock = LockState(True, "generating_image")
        await broadcast_public()

    try:
        img_model = STATE.settings.get("image_model") or "gemini-2.5-flash-image-preview"
        data_url = await gemini_generate_image(img_model, STATE.last_image_prompt)
        STATE.last_image_data_url = data_url
        await announce("Image generated.")
        await broadcast_public()
        return {"ok": True}
    finally:
        STATE.lock = LockState(False, "")
        await broadcast_public()


# --------- WebSockets ---------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Optional player_id supplied as query for private channel
    player_id = ws.query_params.get("player_id")
    STATE.global_sockets.add(ws)
    if player_id and player_id in STATE.players:
        STATE.players[player_id].sockets.add(ws)
        STATE.players[player_id].connected = True
    # Send initial snapshots
    await ws.send_json({"event": "state", "data": STATE.public_snapshot()})
    if player_id:
        await ws.send_json({"event": "private", "data": STATE.private_snapshot_for(player_id)})

    try:
        while True:
            # We don't need to receive anything; just keep connection alive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup
        STATE.global_sockets.discard(ws)
        if player_id and player_id in STATE.players:
            p = STATE.players[player_id]
            p.sockets.discard(ws)
            if not p.sockets:
                p.connected = False
        await broadcast_public()


if __name__ == "__main__":
    # Provide a convenient CLI entry point for local running.
    import uvicorn

    host = os.environ.get("ORPG_HOST", "0.0.0.0")
    port = int(os.environ.get("ORPG_PORT", "8000"))
    uvicorn.run("rpg:app", host=host, port=port, reload=os.environ.get("ORPG_RELOAD") == "1")
