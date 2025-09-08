import os
import uuid
import time
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64

# -----------------------------
# Config
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")      # change if you prefer another local model
JOIN_CODE = os.getenv("JOIN_CODE", "").strip()            # optional: require a code to join
ALLOW_ANYONE_TO_RESOLVE = os.getenv("ALLOW_ANYONE_TO_RESOLVE", "1") == "1"

BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.getenv("BIND_PORT", "8000"))

# -----------------------------
# Minimal in-memory game state
# -----------------------------
class Player:
    def __init__(self, name: str, background: str, abilities: List[str], char_class: Optional[str] = None, status: Optional[str] = None, token: Optional[str] = None, equipment: Optional[List[str]] = None):
        self.id = str(uuid.uuid4())
        self.name = name.strip()[:40]
        self.background = background.strip()[:400]
        self.abilities = abilities[:5]
        self.char_class = (char_class or "").strip()[:40]
        self.status = (status or "Healthy").strip()[:20] or "Healthy"
        self.joined_at = time.time()
        self.last_seen = time.time()
        # Auth token used to verify privileged actions
        self.token = token
        self.equipment = list(equipment or [])[:8]

class Game:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.turn_number: int = 0
        self.current_scenario: Optional[str] = None
        self.current_actions: Dict[str, str] = {}
        self.last_summary: str = ""
        self.history: List[Dict] = []
        self.resolving: bool = False
        self.lock = asyncio.Lock()
        self.host_id: Optional[str] = None   # <- NEW


    def active_players(self, stale_seconds: int = 600) -> List[Player]:
        now = time.time()
        return [p for p in self.players.values() if now - p.last_seen < stale_seconds]

GAME = Game()
app = FastAPI(title="Nils's Online RPG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Tiny 1x1 PNG to avoid favicon 404s (base64-embedded)
FAVICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

# -----------------------------
# Helpers
# -----------------------------
class OllamaUnavailable(Exception):
    """Raised when the Ollama service is unreachable/unavailable."""
    pass

@app.exception_handler(OllamaUnavailable)
async def _handle_ollama_unavailable(request: Request, exc: OllamaUnavailable):
    # Return a concise, readable error for the browser alert
    msg = (
        f"AI service unavailable: {exc}\n"
        f"Host: {OLLAMA_HOST}\n"
        "Hint: start Ollama with 'ollama serve' or set OLLAMA_HOST/OLLAMA_MODEL."
    )
    return PlainTextResponse(msg, status_code=503)

async def choose_class_with_ollama(background: str, max_attempts: int = 5) -> str:
    """Use Ollama to choose a fitting character class based on the player's background."""
    bg = (background or "").strip()
    system = (
        "You assign concise fantasy character classes based on a short background. "
        "Output ONLY the class name — one or two words; prefer one word. No labels, no quotes, no punctuation, "
        "no extras. Prefer evocative archetypes that fit the background (e.g., 'Shadow Blade', 'Stormcaller'). "
        "Avoid articles or descriptors like 'the', 'a', 'class', 'role', or sentences."
    )
    user_tmpl = (
        "Background: {bg}\n\n"
        "Task: Name a fitting fantasy character class (max two words; prefer one word).\n"
        "Respond with ONLY the class name, no quotes or punctuation."
    )

    attempts = 0
    last = ""
    while attempts < max_attempts:
        attempts += 1
        content = await ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_tmpl.format(bg=bg)}
        ], options={"temperature": 0.3, "num_ctx": 8192})
        # normalize
        text = (content or "").strip()
        # remove surrounding quotes/backticks and punctuation-only decorations
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())
        # words by whitespace
        words = text.split(" ") if text else []
        # sometimes models prefix with labels; strip common prefixes
        if len(words) >= 2 and words[0].lower().rstrip(":") in {"class", "archetype", "role"}:
            words = words[1:]
        # keep only alphanumeric/hyphen words
        cleaned = []
        for w in words:
            w2 = "".join(ch for ch in w if ch.isalnum() or ch in "-'")
            if w2:
                cleaned.append(w2)
        if not cleaned:
            last = text
            continue
        if len(cleaned) <= 2:
            return " ".join(cleaned)[:40]
        # ask again with stricter instruction
        last = " ".join(cleaned)
        user_tmpl = (
            "Background: {bg}\n\n"
            "Task: Output ONLY a class name of one or two words; prefer one word. "
            "If you think of more than two, choose the best two-word version. "
            "Absolutely no extra words, labels, punctuation, or quotes."
        )
    # fallback: trim to first two words if still non-compliant
    trimmed = " ".join((last or bg or "Adventurer").split()[:2])
    return trimmed[:40]
async def generate_sheet_with_ollama(background: str, char_class: str, max_attempts: int = 4) -> Tuple[List[str], List[str]]:
    """Use the model to produce abilities and equipment as JSON only.

    Returns (abilities, equipment). If parsing fails after several attempts,
    returns ([], []).
    """
    sys_prompt = (
        "You create concise RPG character sheets. Respond ONLY with JSON. "
        "Given a background and class, output: {\n  \"abilities\": [short strings],\n  \"equipment\": [short strings]\n}. "
        "Keep abilities to 3-5 items and equipment to 3-6 items. "
        "No prose, no code fences."
    )
    user_tmpl = (
        "Background: {bg}\nClass: {cls}\n\n"
        "Task: Output JSON with arrays 'abilities' and 'equipment'."
    )

    def sanitize(items: List) -> List[str]:
        out: List[str] = []
        for x in (items or [])[:8]:
            try:
                s = str(x).strip()
            except Exception:
                continue
            if not s:
                continue
            s = " ".join(s.replace("\n", " ").split())
            out.append(s[:100])
        return out

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        text = await ollama_chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_tmpl.format(bg=(background or "").strip(), cls=(char_class or "").strip())},
        ], options={"temperature": 0.5, "num_ctx": 8192})
        raw = (text or "").strip()
        # Strip code fences if present
        if "```" in raw:
            first = raw.find("```")
            last = raw.rfind("```")
            if last > first:
                content = raw[first+3:last]
                # Drop language tag line if present
                if not content.lstrip().startswith("{") and "\n" in content:
                    content = content.split("\n", 1)[1]
                raw = content.strip()
        # Try to isolate the JSON object
        i0 = raw.find('{')
        i1 = raw.rfind('}')
        candidate = raw[i0:i1+1] if (i0 != -1 and i1 != -1 and i1 > i0) else raw
        try:
            data = json.loads(candidate)
        except Exception:
            data = None
        if isinstance(data, dict):
            abilities = sanitize(data.get("abilities") or [])
            equipment = sanitize(data.get("equipment") or [])
            if abilities or equipment:
                return abilities[:5], equipment[:6]
        sys_prompt = (
            "Respond ONLY with strict JSON of the exact shape: "
            "{\"abilities\":[...],\"equipment\":[...]}."
        )
    return [], []
def party_snapshot(players: List[Player]) -> str:
    if not players:
        return "(no one yet)"
    lines = []
    for p in players:
        cls = p.char_class or ""
        status = p.status or "Healthy"
        lines.append(f"- {p.name} - {cls}; Status: {status}; Abilities: {', '.join(p.abilities)}")
    return "\n".join(lines)
def actions_snapshot(actions: Dict[str, str]) -> str:
    if not actions:
        return "(no actions submitted)"
    lines = []
    for pid, text in actions.items():
        p = GAME.players.get(pid)
        who = p.name if p else pid[:8]
        lines.append(f"- {who}: {text.strip()}")
    return "\n".join(lines)

def auth_get_player_from_body(body: Dict) -> Optional[Player]:
    """Validate player authentication based on id + token in request body.
    Returns the Player if valid, or None if invalid/missing.
    """
    try:
        pid = body.get("player_id")
        token = body.get("player_token")
        if not pid or pid not in GAME.players:
            return None
        player = GAME.players[pid]
        if not token or not isinstance(token, str):
            # Missing token
            return None
        if player.token != token:
            # Wrong token
            return None
        return player
    except Exception:
        return None

async def ollama_chat(messages: List[Dict[str, str]], options: Optional[Dict]=None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    url = f"{OLLAMA_HOST}/api/chat"
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                return ""
            # Be robust to unexpected payload shapes
            if not isinstance(data, dict):
                return ""
            msg = data.get("message")
            if not isinstance(msg, dict):
                return ""
            content = msg.get("content", "")
            if not isinstance(content, str):
                return ""
            return content.strip()
    except Exception as e:
        # Treat likely connection failures as Ollama unavailable; otherwise re-raise
        emsg = str(e).lower()
        if any(tok in emsg for tok in (
            "connect", "connection", "refused", "timed out", "timeout", "dns", "name or service not known"
        )):
            raise OllamaUnavailable(
                f"cannot reach Ollama at {OLLAMA_HOST} ({e.__class__.__name__}: {e})"
            )
        raise

async def update_party_statuses_with_ollama(summary: str, latest_narration: str) -> None:
    """Ask Ollama to update per-player status as a single word, changing only if necessary.
    Uses player IDs in the prompt/output to avoid ambiguity; preserves order.
    If Ollama returns invalid JSON or unusable data, leave statuses unchanged.
    """
    try:
        players = GAME.active_players()
        if not players:
            return

        allowed_examples = (
            "healthy,wounded,unconscious,incapacitated,restrained,petrified,dead,missing,captured,poisoned,bleeding,stunned,exhausted"
        )

        # Build a compact JSON snapshot for the model
        snapshot = [
            {
                "id": p.id,
                "name": p.name,
                "class": p.char_class or "",
                "status": p.status or "Healthy",
            }
            for p in players
        ]

        system = (
            "You update each character's 'status' based on the game context. "
            "Return only JSON. For each input character, output exactly one word status, title-case if natural. "
            "Use a single token with no spaces (e.g., Healthy, Wounded, Unconscious). "
            "Only change a character's status if the Summary or Latest Narration clearly indicate a different condition; otherwise repeat the current status exactly. "
            "Prefer concise terms similar to: " + allowed_examples + "."
        )

        user = (
            "Summary of Facts (latest):\n" + (summary or "(none)") +
            "\n\nLatest Narration:\n" + (latest_narration or "(none)") +
            "\n\nCurrent Players (JSON array):\n" + json.dumps(snapshot, ensure_ascii=True) +
            "\n\nTask: Decide updates minimally. If no change is needed for a player, return their current status.\n" \
            "Output JSON ONLY in this shape (no prose): {\n  \"updates\": [ { \"id\": \"<id>\", \"status\": \"<OneWord>\" }, ... ]\n}"
        )

        raw = await ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], options={"temperature": 0.2, "num_ctx": 8192})

        # Try to extract JSON from response
        text = (raw or "").strip()
        # Some models may wrap; attempt to find the first '{' ... last '}'
        if "{" in text and "}" in text:
            text = text[text.find("{"): text.rfind("}")+1]
        data = json.loads(text)
        if not isinstance(data, dict):
            return
        updates = data.get("updates")
        if not isinstance(updates, list):
            return

        # Helper to normalize to one-word Title Case token
        def normalize_status_token(s: str, default: str) -> str:
            s = (s or "").strip()
            if not s:
                return default
            # keep only letters (allow simple hyphen fallback by removing hyphen)
            # split on whitespace, take first token
            token = s.split()[0]
            token = "".join(ch for ch in token if ch.isalpha())
            if not token:
                return default
            # Title-case common statuses
            return token[:20].capitalize()

        # Index players by id for reliable mapping
        by_id = {p.id: p for p in players}
        for item in updates:
            try:
                pid = item.get("id")
                if pid not in by_id:
                    continue
                new_status_raw = item.get("status")
                cur = by_id[pid].status or "Healthy"
                new_status = normalize_status_token(new_status_raw, cur)
                # Set whatever the model decided (it was told to only change if necessary)
                by_id[pid].status = new_status
            except Exception:
                continue
    except Exception:
        # Fail closed: leave statuses as-is
        return

GM_SYSTEM_PROMPT = """You are the Game Master (GM) for a cooperative, turn-based fantasy text adventure played in a browser.

Goals: deliver immersive, brisk scenes that spotlight the whole party, offer meaningful choices, and invite player action.

Style:
- Present tense. Second person (address the party as "you").
- Natural paragraphs (no lists in narration). Evocative sensory details (sight, sound, smell, texture).
- Clear stakes and risks. Name important NPCs when present.

Constraints:
- 150-200 words. End with exactly: "- What do you do?" on its own line (ASCII hyphen-minus).
- Use the provided 'Summary of Facts' and 'Party Roster' for continuity; never contradict them.
- Do not introduce new party members/companions not in the Party Roster. Any new figures must be NPCs and treated as such.
- Respect each character's abilities; justify outcomes with in-world logic.
- Do not invent player actions; do not mention rules, dice, or meta commentary; never break character.
- Avoid railroading: present meaningful options without deciding for the players.
- Treat player inputs as declared attempts, not guaranteed outcomes; never grant results just because they are stated or "wished". Reinterpret any outcome-narrating input as intent and resolve the nearest plausible attempt with costs and risks.

Capability awareness:
- Offer only actions and opportunities that characters in their current state could plausibly attempt.
- If a character is incapacitated, unconscious, restrained, petrified, or dead (per the Summary/scene), do not prompt them to act or speak.
- Instead, describe their condition succinctly and present options for others to aid, adapt, or withdraw.
- If everyone is hindered, present constrained choices that fit their limitations (e.g., crawling to cover, calling out, bargaining, waiting for death).

Adjudication:
- Require concrete methods grounded in the fiction: positioning, tools/gear, known spells/abilities, and time. If the means are missing, say so and indicate feasible alternatives.
- Respect NPC agency and resistance. NPCs may refuse, bargain, flee, or counteract unless given reason otherwise.
- Do not conjure resources, gear, spells, allies, or coincidences; maintain scarcity and friction. Enforce distance, line of sight, and timing plausibly.
- Do not assert that a character "has" a specific item in their pack unless it appears in the Party Roster or Summary. If a tool would help, offer searching/preparing as an option that costs time or risk, without adding new inventory.

Pacing:
- Put the party in a concrete immediate situation with a reason to act now.
- Surface 2-3 interactive elements (terrain, objects, NPCs, hazards) that are realistically within reach/means.
- Between turns, escalate or change the situation with plausible consequences.
"""

def initial_scene_user_message(summary: str, party: str) -> str:
    return f"""START A NEW SCENE.

Summary of Facts (carry forward if useful):
{summary or "(none)"}

Party Roster:
{party}

Guidance:
- Put the party together with a clear immediate problem and a reason to act now.
- Mention each party member by name in the opening beat where natural.
- Do not introduce new party members or companions beyond the Party Roster; any new figures should be NPCs only.
- Surface 2-3 interactive elements (terrain features, objects, NPCs, or hazards) that fit their means.
- Reflect current capability: if any member is incapacitated/unconscious/restrained/dead per the Summary, acknowledge it and do not prompt them to act; instead, present ways the others might respond.
- Use present tense and second person ("you"). Avoid meta commentary.
- Keep to 150-200 words.
- Frame opportunities as attempts that require means and carry risk; do not imply automatic success or allow players to dictate outcomes by fiat.
 - Do not introduce new character gear; rely on what the Party Roster/Summary or background implies. Environmental affordances are fine (e.g., loose stones, debris), but do not add items "in your pack" unless already established.
End with: "- What do you do?"

Also recognized rendering in some environments: "— What do you do?" (still output ASCII '-') """

def resolution_user_message(summary: str, party: str, scenario: str, actions: str) -> str:
    return f"""RESOLVE THE PARTY'S ACTIONS AND SET UP THE NEXT SITUATION.

Summary of Facts:
{summary or "(none)"}

Previous Scene:
{scenario}

Party Roster:
{party}

Actions Taken This Turn:
{actions}

Task:
- Resolve each stated action fairly, referencing abilities and status to justify outcomes.
- Ensure outcomes reflect current capability. If an action is impossible due to the actor's condition (e.g., incapacitated/unconscious/restrained/dead), do not have them act or speak; resolve with appropriate consequences (nearest feasible interpretation, failure, or need for aid) and note the constraint.
- Do not introduce new party members/companions not in the Party Roster; any new figures must be NPCs.
- Show consequences (including mixed results) and how actions interact with each other.
- Do not invent player actions; keep narration under 200 words in natural paragraphs.
- Treat player inputs as attempts, not accomplished outcomes. If a player narrates a result or "wishes" a world change without the means (gear, positioning, known spell/ability), reinterpret it as intent and resolve the nearest plausible attempt, or state why it fails and suggest feasible alternatives. Do not grant outcomes by fiat.
- Require concrete methods using established resources, positioning, and knowledge. Do not conjure resources, allies, or coincidences; respect distance, line of sight, time pressure, and NPC agency (they may resist, negotiate, or ignore).
- If an action is vague or purely wishful, implicitly clarify by presenting a couple of specific next-step options that fit the situation and risks.
 - If a declaration references a nonexistent NPC or resource, do not substitute a different player action. Reflect the non-effect or closest minimal interpretation, then set up consequences and invite a new choice.
 - Do not narrate characters performing additional actions beyond what they declared. Only resolve the stated attempt and its immediate, necessary micro-steps.
- Change the situation: reveal new information, escalate a threat, or open new opportunities.
- Set up the next situation and end with exactly "- What do you do?"

Also recognized rendering in some environments: "— What do you do?" (still output ASCII '-')

Avoid bullet lists in narration; write flowing prose. Keep momentum and clarity."""

SUMMARIZER_SYSTEM = (
    "You are an expert note-taker maintaining a running 'Summary of Facts' for an ongoing fantasy campaign. "
    "Write neutral, spoiler-free notes for the next turn."
)
SUMMARIZER_USER_TMPL = """Update the compact 'Summary of Facts' for the next turn.

Prior Summary:
{prior}

Party Roster:
{party}

Newest Narration:
{narr}

Rules:
- Do not add new party members; anyone not in the Party Roster is an NPC.
- Use ASCII characters; avoid fancy punctuation that may not render.
 - List only resources explicitly established in the Party Roster, prior Summary, or newest narration; do not invent new gear, spells, or allies.

Produce 8-12 single-line bullet points (each starting with "- "). Cover: current locations, key NPCs with short tags, per-character status (healthy/wounded/unconscious/incapacitated/restrained/petrified/dead/missing/captured), party resources (notable gear/spells), notable items/clues, active threats/timers, immediate goals, and open questions. Be concrete and avoid prose.
Keep it under ~180 words."""

# -----------------------------
# Post-processors
# -----------------------------
def normalize_narration(text: str) -> str:
    """Ensure the closing prompt line uses ASCII hyphen-minus.
    If a line ending with some dash-like char before 'What do you do?' exists, normalize it to '- What do you do?'.
    """
    try:
        s = (text or "")
        lines = s.splitlines()
        # scan from bottom
        dash_like = {"-", "–", "—", "−", "‒", "―"}
        for i in range(len(lines) - 1, -1, -1):
            raw = lines[i].strip()
            # allow optional leading dash-like + spaces
            if raw.lower().endswith("what do you do?"):
                # find first char that could be a dash
                j = 0
                while j < len(raw) and raw[j].isspace():
                    j += 1
                if j < len(raw) and raw[j] in dash_like:
                    # replace entire line with normalized form
                    lines[i] = "- What do you do?"
                    return "\n".join(lines)
        return s
    except Exception:
        return text or ""

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/favicon.ico")
async def favicon():
    return Response(content=FAVICON_PNG, media_type="image/png")

@app.get("/state")
async def get_state(player_id: Optional[str] = None):
    # heartbeat for presence
    if player_id and player_id in GAME.players:
        GAME.players[player_id].last_seen = time.time()

    active = GAME.active_players()
    you = GAME.players.get(player_id) if player_id else None
    is_host = bool(you and GAME.host_id == you.id)   # <- NEW
    your_action = GAME.current_actions.get(player_id, "") if player_id else ""
    actions_for_state = []
    for pid, txt in GAME.current_actions.items():
        p = GAME.players.get(pid)
        name = p.name if p else "Someone"
        actions_for_state.append({
            "name": name,
            "text": txt,
            "is_you": bool(player_id and pid == player_id),
        })

    you_obj = None
    if you:
        you_obj = {
            "name": you.name,
            "archetype": you.char_class or "",
            "abilities": list(you.abilities),
            "equipment": list(getattr(you, "equipment", []) or []),
            "status": you.status or "Healthy",
        }

    return {
        "turn": GAME.turn_number,
        "scenario": GAME.current_scenario,
        "summary": GAME.last_summary,
        "party": [{
            "name": p.name,
            "archetype": (p.char_class or ""),
            "status": (p.status or "Healthy"),
            "is_you": bool(player_id and p.id == player_id),
        } for p in active],
        "your_action": your_action,
        "you": you_obj,
        "actions": actions_for_state,
        "actions_submitted": len(GAME.current_actions),
        "resolving": GAME.resolving,
        "can_resolve": ALLOW_ANYONE_TO_RESOLVE or is_host,
        "join_code_required": bool(JOIN_CODE),
        "is_host": is_host,
    }

@app.post("/join")
async def join(req: Request):
    data = await req.json()
    name = (data.get("name") or "").strip()
    background = (data.get("background") or "").strip()
    code = (data.get("code") or "").strip()

    if not name or not background:
        return JSONResponse({"error": "Name and background are required."}, status_code=400)
    if JOIN_CODE and code != JOIN_CODE:
        return JSONResponse({"error": "Invalid join code."}, status_code=403)

    char_class = await choose_class_with_ollama(background)
    abilities, equipment = await generate_sheet_with_ollama(background, char_class)

    # Generate an auth token for this player
    import secrets
    token = secrets.token_urlsafe(24)
    p = Player(name, background, abilities, char_class=char_class, token=token, equipment=equipment)
    GAME.players[p.id] = p
    if GAME.host_id is None:
        GAME.host_id = p.id

    # if this is the first player, spin up an initial scene (lock-protected)
    if GAME.turn_number == 0 and not GAME.current_scenario:
        async with GAME.lock:                         
            if GAME.turn_number == 0 and not GAME.current_scenario:  
                try:
                    await ensure_initial_scene()
                except OllamaUnavailable:
                    # Roll back player creation so we don't keep a ghost player
                    try:
                        if p.id in GAME.players:
                            del GAME.players[p.id]
                        if GAME.host_id == p.id:
                            GAME.host_id = None
                    finally:
                        # Re-raise to be handled by the global exception handler
                        raise
    return {"player_id": p.id, "player_token": p.token, "name": p.name}

@app.post("/action")
async def submit_action(req: Request):
    data = await req.json()
    pid = data.get("player_id")
    text = (data.get("text") or "").strip()
    player = auth_get_player_from_body(data)
    if not pid or pid not in GAME.players:
        return JSONResponse({"error": "Invalid player."}, status_code=400)
    if player is None:
        # Distinguish missing token vs invalid
        if not data.get("player_token"):
            return JSONResponse({"error": "Auth token required."}, status_code=401)
        return JSONResponse({"error": "Invalid auth token."}, status_code=403)
    if not text:
        GAME.current_actions.pop(pid, None)
    else:
        GAME.current_actions[pid] = text[:500]
    GAME.players[pid].last_seen = time.time()
    return {"ok": True}

@app.post("/leave")
async def leave(req: Request):
    data = await req.json()
    pid = data.get("player_id")
    if not pid or pid not in GAME.players:
        return JSONResponse({"error": "Invalid player."}, status_code=400)
    player = auth_get_player_from_body(data)
    if player is None:
        if not data.get("player_token"):
            return JSONResponse({"error": "Auth token required."}, status_code=401)
        return JSONResponse({"error": "Invalid auth token."}, status_code=403)

    GAME.current_actions.pop(pid, None)
    departing_host = pid == GAME.host_id
    del GAME.players[pid]

    if departing_host:
        active = GAME.active_players()
        GAME.host_id = active[0].id if active else None

    return {"ok": True}

@app.post("/resolve")
async def resolve_turn(req: Request):
    if GAME.resolving:
        return {"status": "already resolving"}
    # anybody can trigger if enabled
    body = await req.json()
    pid = body.get("player_id")
    if not pid or pid not in GAME.players:
        return JSONResponse({"error": "Invalid player."}, status_code=400)
    player = auth_get_player_from_body(body)
    if player is None:
        if not body.get("player_token"):
            return JSONResponse({"error": "Auth token required."}, status_code=401)
        return JSONResponse({"error": "Invalid auth token."}, status_code=403)
    if not ALLOW_ANYONE_TO_RESOLVE and pid != GAME.host_id:
        return JSONResponse({"error": "Only the host can resolve turns."}, status_code=403)

    GAME.players[pid].last_seen = time.time()

    async with GAME.lock:
        if GAME.resolving:
            return {"status": "already resolving"}
        GAME.resolving = True
        try:
            await do_resolution()
            return {"ok": True}
        finally:
            GAME.resolving = False

@app.get("/healthz")
async def health():
    return {"ok": True, "players": len(GAME.players), "turn": GAME.turn_number}

# -----------------------------
# Core engine
# -----------------------------
async def ensure_initial_scene():
    # craft the very first scene when the first player joins
    party = party_snapshot(GAME.active_players())
    user_msg = initial_scene_user_message(GAME.last_summary, party)
    scene = await ollama_chat(
        [
            {"role": "system", "content": GM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.8, "num_ctx": 8192}
    )
    scene = normalize_narration(scene)
    GAME.turn_number = 1
    GAME.current_scenario = scene
    GAME.current_actions.clear()
    # also seed an initial summary from the opener
    GAME.last_summary = await ollama_chat(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": SUMMARIZER_USER_TMPL.format(prior="", narr=scene, party=party)},
        ],
        options={"temperature": 0.2, "num_ctx": 8192}
    )
    # Update per-player statuses based on the initial scene/summary
    try:
        await update_party_statuses_with_ollama(GAME.last_summary, scene)
    except Exception:
        pass

async def do_resolution():
    # if no scene, create one
    if not GAME.current_scenario:
        await ensure_initial_scene()
        return

    party = party_snapshot(GAME.active_players())
    actions = actions_snapshot(GAME.current_actions)
    user_msg = resolution_user_message(GAME.last_summary, party, GAME.current_scenario, actions)

    narration = await ollama_chat(
        [
            {"role": "system", "content": GM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.8, "num_ctx": 8192}
    )
    narration = normalize_narration(narration)

    # update history
    GAME.history.append({
        "turn": GAME.turn_number,
        "scenario": GAME.current_scenario,
        "actions": GAME.current_actions.copy(),
        "narration": narration,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    # new summary for next turn
    new_summary = await ollama_chat(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": SUMMARIZER_USER_TMPL.format(prior=GAME.last_summary, narr=narration, party=party)},
        ],
        options={"temperature": 0.2, "num_ctx": 8192}
    )

    GAME.last_summary = new_summary
    GAME.turn_number += 1
    GAME.current_scenario = narration
    GAME.current_actions.clear()

    # Ask Ollama to minimally update statuses based on new events
    try:
        await update_party_statuses_with_ollama(GAME.last_summary, GAME.current_scenario)
    except Exception:
        pass

# -----------------------------
# Minimal single-page client
# -----------------------------
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nils's Online RPG</title>
<link rel="icon" type="image/png" href="/favicon.ico"/>
<style>
:root{
  --bg:#0b0f14; --card:#0f141b; --ink:#e6edf3; --muted:#9fb1c3; --accent:#74c0ff; --accent2:#ffd27d;
  --ok:#8be28b; --warn:#ff9d5c;
}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b0f14,#0a0e13);color:var(--ink);font:16px/1.45 system-ui,Segoe UI,Roboto,Ubuntu,Arial}
.container{max-width:1100px;margin:0 auto;padding:20px}
header{display:flex;align-items:center;justify-content:space-between}
.brand{font-weight:700;font-size:20px;letter-spacing:.4px}
.card{background:linear-gradient(180deg,#0e141b,#0b1117);border:1px solid #19202a;border-radius:18px;box-shadow:0 10px 40px #0006;padding:16px}
.grid{display:grid;gap:16px}
@media(min-width:900px){.grid{grid-template-columns:2fr 1fr}}
label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
input,textarea{width:100%;background:#0a1118;border:1px solid #1b2531;color:var(--ink);border-radius:12px;padding:10px 12px;font:inherit}
textarea{min-height:120px;resize:vertical}
button{background:linear-gradient(180deg,#1b5a99,#164875);color:white;border:0;border-radius:12px;padding:10px 14px;font-weight:600;cursor:pointer}
button.secondary{background:#0f141b;border:1px solid #203044}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.small{font-size:12px;color:var(--muted)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#12202f;color:#a7c6e6;border:1px solid #1e3146;font-size:12px}
.badge.you{background:var(--accent2);color:#000;border-color:var(--accent2)}
#scenario{white-space:pre-wrap}
ul{padding-left:18px;margin:8px 0}
hr{border:0;border-top:1px solid #1d2733;margin:12px 0}
.kv{display:flex;gap:8px;flex-wrap:wrap}
.kv span{padding:4px 8px;background:#0a1118;border:1px solid #1b2531;border-radius:10px;color:#bdd0e2}
footer{margin-top:20px;color:#7b8b9b}
</style>
</head>
<body>
<div id="busy" class="small" style="display:none;position:fixed;top:10px;right:10px;background:#12202f;border:1px solid #1e3146;border-radius:8px;padding:4px 8px;z-index:1000">⏳ thinking…</div>
<div class="container">
  <header>
    <div class="brand">⚔️ Nils's Online RPG</div>
    <div class="small" id="serverInfo"></div>
  </header>

  <div id="join" class="card" style="display:none">
    <h3>Join the party</h3>
    <div class="grid">
      <div>
        <label>Display name</label>
        <input id="name" maxlength="40" placeholder="e.g. Rowan"/>
      </div>
      <div id="joinCodeWrap" style="display:none">
        <label>Join code</label>
        <input id="joinCode" placeholder="Ask the host"/>
      </div>
    </div>
    <div style="margin-top:10px">
      <label>Character background (1–3 sentences)</label>
      <textarea id="background" placeholder="Who are you, what shaped you, why are you adventuring?"></textarea>
    </div>
    <div class="row" style="margin-top:10px">
      <button id="joinBtn" onclick="doJoin()">Enter the world</button>
      <div class="small">Your abilities will be generated from your background.</div>
    </div>
  </div>

  <div id="game" style="display:none">
    <div class="grid">
      <section class="card">
        <div class="row" style="justify-content:space-between">
          <div class="badge">Turn <span id="turn">–</span></div>
          <div class="kv">
            <span id="resolving" style="display:none">⏳ resolving...</span>
            <span>Players: <b id="pcnt">0</b></span>
          </div>
        </div>
        <hr/>
        <div id="scenario">Waiting for the first scene…</div>

        <hr/>
        <label>Your action this turn</label>
        <textarea id="action" placeholder="Describe what your character attempts… (you can edit until the turn resolves)"></textarea>
        <div class="row" style="margin-top:8px">
          <button id="submitBtn" onclick="submitAction()">Submit</button>
          <button id="resolveBtn" class="secondary" style="display:none" onclick="resolveNow()">Resolve turn</button>
          <button id="leaveBtn" class="secondary" onclick="leaveGame()">Leave game</button>
        </div>
      </section>

      <aside class="card">
        <h3>Party</h3>
        <div id="party"></div>
        <hr/>
        <h4>Your Character</h4>
        <div id="youPanel" class="small"></div>
        <hr/>
        <h4>Submitted Actions</h4>
        <div id="actions" class="small"></div>
      </aside>
    </div>

    <footer class="small">
      Tip: anyone can click "Resolve turn" if the host enabled it. Otherwise, actions auto-collect until the host resolves.
    </footer>
  </div>
</div>

<script>
const S = {
  player_id: localStorage.getItem("player_id") || "",
  player_token: localStorage.getItem("player_token") || "",
  name: localStorage.getItem("name") || "",
  pollMs: 2000,
  joinCodeRequired: false,
  canResolve: false,
  actionDirty: false,
  lastTurn: 0,
  pendingOps: 0,
  serverResolving: false,
  submittedOnce: false
};

function qs(id){return document.getElementById(id)}
function show(id, v){qs(id).style.display = v ? "" : "none"}
function t(el, s){ if(el){ el.textContent = (s===undefined||s===null) ? '' : String(s); } }

function updateBusy(){
  show("busy", S.pendingOps > 0 || S.serverResolving);
}

function busy(on){
  S.pendingOps += on ? 1 : -1;
  if(S.pendingOps < 0) S.pendingOps = 0;
  updateBusy();
}

function btnBusy(id, on, txt, count){
  const b = qs(id);
  if(!b) return;
  if(on){
    b.dataset.orig = b.textContent;
    b.disabled = true;
    if(count){
      let secs = 0;
      b.textContent = txt ? `${txt} (0s)` : `${b.textContent} (0s)`;
      b._timer = setInterval(() => {
        secs++;
        b.textContent = txt ? `${txt} (${secs}s)` : `${b.dataset.orig} (${secs}s)`;
      }, 1000);
    }else if(txt){
      b.textContent = txt;
    }
  }else{
    if(b._timer){ clearInterval(b._timer); delete b._timer; }
    if(b.dataset.orig !== undefined){
      b.textContent = b.dataset.orig;
      delete b.dataset.orig;
    }
    b.disabled = false;
  }
}

qs("action").addEventListener("input", () => { S.actionDirty = true; });

async function api(path, opts={}){
  const r = await fetch(path, Object.assign({headers: {"Content-Type":"application/json"}}, opts));
  if(!r.ok){throw new Error(await r.text())}
  return r.json();
}

async function load(){
  t(qs("serverInfo"), "Model: " + (new URLSearchParams(location.search).get("model") || "local Ollama"));

  const state = await api("/state" + (S.player_id ? ("?player_id="+encodeURIComponent(S.player_id)) : ""));
  S.joinCodeRequired = !!state.join_code_required;

  // If we have an id but no token (from older sessions), treat as logged out
  if(S.player_id && !S.player_token){
    try{ alert("Your session needs an update. Please re-join to continue."); }catch(_){ }
    localStorage.removeItem("player_id");
    S.player_id = "";
  }

  if(!S.player_id){
    show("join", true);
    show("game", false);
    show("joinCodeWrap", S.joinCodeRequired);
  }else{
    show("join", false);
    show("game", true);
  }
  render(state);
  setInterval(refresh, S.pollMs);
}

function render(state){
  t(qs("turn"), state.turn || "-");
  t(qs("pcnt"), (state.party||[]).length);
  qs("scenario").textContent = state.scenario || "Waiting for the first scene…";
  const _sum = qs("summary"); if(_sum) _sum.textContent  = state.summary || "(none yet)";
  qs("resolving").style.display = state.resolving ? "" : "none";
  S.serverResolving = !!state.resolving;
  updateBusy();
  S.canResolve = state.can_resolve;
  show("resolveBtn", !!S.canResolve);

  if(state.turn !== undefined && state.turn !== S.lastTurn){
    S.lastTurn = state.turn;
    S.actionDirty = false;
  }

  const me = S.player_id;
  if(me && !S.actionDirty) qs("action").value = state.your_action || "";
  const submitBtn = qs("submitBtn");
  const hasAction = !!state.your_action;
  S.submittedOnce = hasAction;
  if(!submitBtn.disabled){
    submitBtn.textContent = hasAction ? "Submit again" : "Submit";
  }

  // Safely render party without using innerHTML for untrusted data
  (function renderParty(){
    const partyEl = qs("party");
    if(!partyEl) return;
    partyEl.innerHTML = "";
    const list = state.party || [];
    if(!list.length){
      partyEl.innerHTML = "<div class='small'>(empty)</div>";
      return;
    }
    list.forEach(p => {
      const row = document.createElement('div');
      row.className = 'row';
      row.style.justifyContent = 'space-between';

      const left = document.createElement('div');
      left.textContent = p.name || '';
      if(p.is_you){
        const you = document.createElement('span');
        you.className = 'badge you';
        you.textContent = 'you';
        left.appendChild(document.createTextNode(' '));
        left.appendChild(you);
      }

      const right = document.createElement('div');
      right.className = 'small';
      right.textContent = p.archetype || '';

      row.appendChild(left);
      row.appendChild(right);
      partyEl.appendChild(row);
    });
  })();

  // Append status badges without altering structure
  try{
    const partyEl = qs("party");
    if(partyEl){
      const rows = partyEl.querySelectorAll(".row");
      (state.party||[]).forEach((p, i) => {
        const right = rows[i]?.querySelector('.small');
        if(right && !right.dataset.statusAdded){
          right.dataset.statusAdded = '1';
          const span = document.createElement('span');
          span.className = 'badge';
          span.textContent = p.status || 'Healthy';
          right.appendChild(document.createTextNode(' '));
          right.appendChild(span);
        }
      });
    }
  }catch(_){ /* ignore DOM issues */ }

  // Render submitted actions in the side pane
  try{
    const actionsEl = qs("actions");
    if(actionsEl){
      const actions = state.actions || [];
      if(!actions.length){
        actionsEl.innerHTML = "<div class='small'>(none)</div>";
      }else{
        const ul = document.createElement('ul');
        actions.forEach(a => {
          const li = document.createElement('li');
          let who = a.name || '';
          if(a.is_you) who += " (you)";
          const strong = document.createElement('b');
          strong.textContent = who + ": ";
          li.appendChild(strong);
          li.appendChild(document.createTextNode(a.text || ""));
          ul.appendChild(li);
        });
        actionsEl.innerHTML = "";
        actionsEl.appendChild(ul);
      }
    }
  }catch(_){ /* ignore DOM issues */ }

  // Render Your Character panel (abilities + equipment)
  try{
    const yp = qs('youPanel');
    if(yp){
      yp.innerHTML = '';
      const you = state.you || null;
      if(!you){
        yp.textContent = '(join to view)';
      }else{
        // Abilities
        const labA = document.createElement('label');
        labA.textContent = 'Abilities';
        yp.appendChild(labA);
        const ulA = document.createElement('ul');
        (you.abilities||[]).forEach(item => {
          const li = document.createElement('li');
          li.textContent = item;
          ulA.appendChild(li);
        });
        if(!ulA.childElementCount){
          const li = document.createElement('li');
          li.textContent = '(none)';
          ulA.appendChild(li);
        }
        yp.appendChild(ulA);

        // Equipment
        const labE = document.createElement('label');
        labE.textContent = 'Equipment';
        labE.style.marginTop = '8px';
        yp.appendChild(labE);
        const ulE = document.createElement('ul');
        (you.equipment||[]).forEach(item => {
          const li = document.createElement('li');
          li.textContent = item;
          ulE.appendChild(li);
        });
        if(!ulE.childElementCount){
          const li = document.createElement('li');
          li.textContent = '(none)';
          ulE.appendChild(li);
        }
        yp.appendChild(ulE);
      }
    }
  }catch(_){ /* ignore DOM issues */ }
}

async function refresh(){
  try{
    const state = await api("/state" + (S.player_id ? ("?player_id="+encodeURIComponent(S.player_id)) : ""));
    render(state);
  }catch(e){ console.warn(e); }
}

async function doJoin(){
  const name = qs("name").value.trim();
  const background = qs("background").value.trim();
  const code = qs("joinCode")?.value.trim() || "";
  if(!name || !background){ alert("Please fill in name and background."); return; }
  btnBusy("joinBtn", true, "Entering...", true);
  busy(true);
  try{
    const res = await api("/join", {method:"POST", body: JSON.stringify({name, background, code})});
    S.player_id = res.player_id; S.player_token = res.player_token; S.name = res.name;
    localStorage.setItem("player_id", S.player_id);
    localStorage.setItem("player_token", S.player_token);
    localStorage.setItem("name", S.name);
    show("join", false); show("game", true);
    refresh();
  }catch(e){
    alert("Join failed: " + e.message);
  }finally{
    busy(false);
    btnBusy("joinBtn", false);
  }
}

async function submitAction(){
  if(!S.player_id){ alert("Join first!"); return; }
  const text = qs("action").value;
  btnBusy("submitBtn", true, "Submitting...");
  busy(true);
  try{
    await api("/action", {method:"POST", body: JSON.stringify({player_id: S.player_id, player_token: S.player_token, text})});
    S.actionDirty = false;
    S.submittedOnce = true;
    await refresh();
  }finally{
    busy(false);
    btnBusy("submitBtn", false);
    if(S.submittedOnce) qs("submitBtn").textContent = "Submit again";
  }
}

/* clearAction() removed with Clear button; delete empty action by deleting text and submitting */

async function leaveGame(){
  if(!S.player_id) return;
  btnBusy("leaveBtn", true, "Leaving...");
  busy(true);
  try{
    await api("/leave", {method:"POST", body: JSON.stringify({player_id: S.player_id, player_token: S.player_token})});
  }catch(e){
    console.warn(e);
  }finally{
    S.player_id = "";
    S.player_token = "";
    localStorage.removeItem("player_id");
    localStorage.removeItem("player_token");
    show("join", true);
    show("game", false);
    busy(false);
    btnBusy("leaveBtn", false);
    refresh();
  }
}

async function resolveNow(){
  if(!S.canResolve){ alert("Resolving is disabled by host."); return; }
  btnBusy("resolveBtn", true, "Resolving...", true);
  busy(true);
  try{
    show("resolving", true);
    await api("/resolve", {method:"POST", body: JSON.stringify({player_id: S.player_id, player_token: S.player_token})});
  }catch(e){
    alert("Resolve failed: " + e.message);
  }finally{
    show("resolving", false);
    busy(false);
    btnBusy("resolveBtn", false);
    refresh();
  }
}

load();
</script>
</body>
</html>
"""

# -----------------------------
# Entrypoint (dev convenience)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"Starting on http://{BIND_HOST}:{BIND_PORT}  (model={OLLAMA_MODEL})")
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)


