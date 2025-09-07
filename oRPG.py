import os
import uuid
import time
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

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
    def __init__(self, name: str, background: str, power: float, abilities: List[str], char_class: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name.strip()[:40]
        self.background = background.strip()[:400]
        self.power = round(power, 2)
        self.abilities = abilities[:5]
        self.char_class = (char_class or "").strip()[:40]
        self.joined_at = time.time()
        self.last_seen = time.time()

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
        self.host_id: Optional[str] = None   # <— NEW


    def active_players(self, stale_seconds: int = 600) -> List[Player]:
        now = time.time()
        return [p for p in self.players.values() if now - p.last_seen < stale_seconds]

GAME = Game()
app = FastAPI(title="Ollama Fantasy Party Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -----------------------------
# Helpers
# -----------------------------
async def choose_class_with_ollama(background: str, max_attempts: int = 5) -> str:
    """Use Ollama to choose a fitting character class based on the player's background.
    Constraints:
    - Return at most two words.
    - If Ollama returns more than two words, ask again (up to max_attempts),
      tightening instructions. Finally, fall back to the first two words.
    - Do not rely on any predefined class list.
    """
    bg = (background or "").strip()
    system = (
        "You assign concise fantasy character classes based on a short background. "
        "Return only the class name. Keep it to at most two words. "
        "No punctuation, no explanations."
    )
    user_tmpl = (
        "Background: {bg}\n\n"
        "Task: Name a fitting fantasy character class (max two words).\n"
        "Respond with ONLY the class name, no quotes."
    )

    attempts = 0
    last = ""
    while attempts < max_attempts:
        attempts += 1
        content = await ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_tmpl.format(bg=bg)}
        ], options={"temperature": 0.3, "num_ctx": 1024})
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
            "Task: Output ONLY a class name of one or two words. "
            "If you think of more than two, choose the best two-word version. "
            "Absolutely no extra words."
        )
    # fallback: trim to first two words if still non-compliant
    trimmed = " ".join((last or bg or "Adventurer").split()[:2])
    return trimmed[:40]
def abilities_for_class(char_class: str, power: float, bg: str) -> List[str]:
    """Derive simple, generic abilities without relying on predefined class lists.
    - First ability reflects a tier based on relative power.
    - Abilities are generic and class-agnostic to avoid predefined catalogs.
    - Last ability is a 'Signature' flavored by the first line of background.
    """
    tier = "Novice" if power < 0.95 else ("Seasoned" if power < 1.05 else "Expert")
    class_token = (char_class or "Class").splitlines()[0][:20]
    base = [
        f"{tier} {class_token} Techniques".strip(),
        "Resourcefulness",
        "Adaptability",
    ]
    flavored = base + [f"Signature: {bg.splitlines()[0][:60]}"]
    return flavored[:4]
def party_snapshot(players: List[Player]) -> str:
    if not players:
        return "(no one yet)"
    lines = []
    for p in players:
        cls = p.char_class or ""
        lines.append(f"- {p.name} ({p.power}x power) - {cls}; Abilities: {', '.join(p.abilities)}")
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

async def ollama_chat(messages: List[Dict[str, str]], options: Optional[Dict]=None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    url = f"{OLLAMA_HOST}/api/chat"
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

GM_SYSTEM_PROMPT = """You are the Game Master (GM) for a cooperative, turn-based fantasy text adventure played in a browser.
Write vivid but concise scenes (max ~200 words). Always end your output with: "— What do you do?"
Adhere to continuity using the 'Summary of Facts'. Respect the party roster and their abilities/power.
No dice mechanics; narrate plausible outcomes. Avoid railroading; present meaningful choices.
"""

def initial_scene_user_message(summary: str, party: str) -> str:
    return f"""START A NEW SCENE.

Summary of Facts (carry forward if useful):
{summary or "(none)"}

Party Roster:
{party}

Write an opening situation that puts the party together with a clear immediate problem, hooks, and sensory details.
Do NOT spoil future events. 150–200 words.
End with: "— What do you do?" """

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
1) Narrate the outcomes as a cohesive scene (<=200 words).
2) Immediately set up the next situation and end with "— What do you do?"

Avoid bullet lists; write natural narration. Keep momentum; reflect abilities and risks realistically."""

SUMMARIZER_SYSTEM = "You are an expert note-taker for an ongoing fantasy campaign."
SUMMARIZER_USER_TMPL = """Update the compact 'Summary of Facts' for the next turn.

Prior Summary:
{prior}

Newest Narration:
{narr}

Produce 8–12 concise bullet points including locations, NPCs, goals, party condition, important items, and open threads.
Keep it under ~180 words. """

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/state")
async def get_state(player_id: Optional[str] = None):
    # heartbeat for presence
    if player_id and player_id in GAME.players:
        GAME.players[player_id].last_seen = time.time()

    active = GAME.active_players()
    you = GAME.players.get(player_id) if player_id else None
    is_host = bool(you and GAME.host_id == you.id)   # <— NEW
    your_action = GAME.current_actions.get(player_id, "") if player_id else ""
    return {
        "turn": GAME.turn_number,
        "scenario": GAME.current_scenario,
        "summary": GAME.last_summary,
        "party": [{"id": p.id, "name": p.name, "power": p.power, "archetype": (p.char_class or "")} for p in active],
        "your_action": your_action,
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

    # scale power to party
    active = GAME.active_players()
    power = sum([p.power for p in active]) / len(active) if active else 1.0
    char_class = await choose_class_with_ollama(background)
    abilities = abilities_for_class(char_class, power, background)

    p = Player(name, background, power, abilities, char_class=char_class)
    GAME.players[p.id] = p
    if GAME.host_id is None:
        GAME.host_id = p.id

    # if this is the first player, spin up an initial scene (lock-protected)
    if GAME.turn_number == 0 and not GAME.current_scenario:
        async with GAME.lock:                         
            if GAME.turn_number == 0 and not GAME.current_scenario:  
                await ensure_initial_scene()
    return {"player_id": p.id, "name": p.name}

@app.post("/action")
async def submit_action(req: Request):
    data = await req.json()
    pid = data.get("player_id")
    text = (data.get("text") or "").strip()
    if not pid or pid not in GAME.players:
        return JSONResponse({"error": "Invalid player."}, status_code=400)
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
        options={"temperature": 0.8, "num_ctx": 4096}
    )
    GAME.turn_number = 1
    GAME.current_scenario = scene
    GAME.current_actions.clear()
    # also seed an initial summary from the opener
    GAME.last_summary = await ollama_chat(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": SUMMARIZER_USER_TMPL.format(prior="", narr=scene)},
        ],
        options={"temperature": 0.2, "num_ctx": 2048}
    )

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
        options={"temperature": 0.8, "num_ctx": 4096}
    )

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
            {"role": "user", "content": SUMMARIZER_USER_TMPL.format(prior=GAME.last_summary, narr=narration)},
        ],
        options={"temperature": 0.2, "num_ctx": 2048}
    )

    GAME.last_summary = new_summary
    GAME.turn_number += 1
    GAME.current_scenario = narration
    GAME.current_actions.clear()

# -----------------------------
# Minimal single-page client
# -----------------------------
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Ollama Fantasy Party</title>
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
    <div class="brand">⚔️ Ollama Fantasy Party</div>
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
      <div class="small">Your power/abilities will auto-balance to the party.</div>
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
        <h4>Summary of Facts</h4>
        <div id="summary" class="small" style="white-space:pre-wrap"></div>
      </aside>
    </div>

    <section class="card" style="margin-top:16px">
      <h3>How friends join</h3>
      <div class="small">Share this URL: <code id="shareURL"></code></div>
      <div class="small">If you're exposing to the internet, forward port <b>8000</b> to this machine and allow it in Windows Defender Firewall.</div>
    </section>
  </div>

  <footer class="small">
    Tip: anyone can click “Resolve turn” if the host enabled it. Otherwise, actions auto-collect until the host resolves.
  </footer>
</div>

<script>
const S = {
  player_id: localStorage.getItem("player_id") || "",
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
  qs("serverInfo").textContent = "Model: " + (new URLSearchParams(location.search).get("model") || "local Ollama");
  qs("shareURL").textContent = location.href;

  const state = await api("/state" + (S.player_id ? ("?player_id="+encodeURIComponent(S.player_id)) : ""));
  S.joinCodeRequired = !!state.join_code_required;

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
  qs("turn").textContent = state.turn || "–";
  qs("pcnt").textContent = (state.party||[]).length;
  qs("scenario").textContent = state.scenario || "Waiting for the first scene…";
  qs("summary").textContent  = state.summary || "(none yet)";
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

  qs("party").innerHTML = (state.party||[]).map(p => {
    const you = p.id === S.player_id ? ' <span class="badge you">you</span>' : '';
    return `<div class="row" style="justify-content:space-between">
      <div>${p.name}${you}</div>
      <div class="small">${p.archetype} <span class="badge">${p.power}×</span></div>
    </div>`;
  }).join("") || "<div class='small'>(empty)</div>";
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
    S.player_id = res.player_id; S.name = res.name;
    localStorage.setItem("player_id", S.player_id);
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
    await api("/action", {method:"POST", body: JSON.stringify({player_id: S.player_id, text})});
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
    await api("/leave", {method:"POST", body: JSON.stringify({player_id: S.player_id})});
  }catch(e){
    console.warn(e);
  }finally{
    S.player_id = "";
    localStorage.removeItem("player_id");
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
    await api("/resolve", {method:"POST", body: JSON.stringify({player_id: S.player_id})});
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
