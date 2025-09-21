# oRPG

A LAN-friendly multiplayer tabletop RPG host built on FastAPI, WebSockets, and structured LLM calls. oRPG keeps a shared world state, lets players join from a browser, and can enrich every turn with auto-generated images, voice narration, and concise history summaries. It currently supports Google Gemini and xAI Grok text models plus optional ElevenLabs speech synthesis.

## Highlights
- Real-time lobby and turn engine served from `static/index.html`; players join, submit actions, and watch turns resolve over WebSockets.
- Structured JSON contracts keep characters, inventories, and public status words consistent across turns.
- Dual text providers: drop in a Gemini or Grok model name and the server selects the right API key automatically.
- Optional scene art and portrait generation via Gemini image models, including auto-on-turn toggles and per-player portrait refresh.
- ElevenLabs narration integration with automatic queuing, error reporting, and cost tracking for spoken turns.
- English and German language modes with localized UI strings, GM prompt templates, and rules for the structured responses.
- History can be kept verbatim or condensed into short bullet summaries to manage context windows; both are surfaced to players.
- Session stats panel tracks token usage, cost estimates (via `model_prices.json`), request timings, and image/audio spend.

## Requirements
- Python 3.11+ (the codebase relies on modern type hints and `asyncio` features).
- A Google Gemini API key if you intend to use Gemini text or image models.
- An xAI Grok API key if you prefer Grok models for structured text.
- An ElevenLabs API key if you want automatic narration (optional).

All keys are stored locally in `settings.json`, which stays ignored by git.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On first launch the server will create `settings.json` with sane defaults. Populate the fields you plan to use before inviting players.

## Configuration (`settings.json`)
| Key | Purpose |
| --- | ------- |
| `api_key` | Google Gemini API key used for Gemini text/image calls. |
| `grok_api_key` | xAI Grok API key; required when `text_model` targets Grok. |
| `elevenlabs_api_key` | ElevenLabs key for narration; required before enabling auto-TTS. |
| `text_model` | Structured text model (e.g. `gemini-2.0-flash`, `grok-4-fast-non-reasoning`). Provider is auto-detected. |
| `image_model` | Gemini image model for scene and portrait generation. |
| `narration_model` | ElevenLabs model/voice to narrate turns. |
| `world_style` | Flavour string shown in the UI and injected into prompts. |
| `difficulty` | Narrative difficulty cue sent to the GM model. |
| `thinking_mode` | LLM reasoning budget (`none`, `brief`, `balanced`, `deep`). |
| `language` | Active language for prompts/UI (`en` or `de`). |
| `history_mode` | `full` keeps every turn transcript; `summary` maintains a concise bullet history. |

You can edit the file by hand or use the in-app **Settings** modal. `gm_prompt.txt` and `gm_prompt.de.txt` customize the GM persona; keep the `<<TURN_DIRECTIVE>>` token in place.

## Running the Server
```bash
python rpg.py
# or
uvicorn rpg:app --host 0.0.0.0 --port 8000 --reload
```

Environment variables:
- `ORPG_HOST` – override bind address (default is a dual-stack `::` listener).
- `ORPG_PORT` – override port (default `8000`).
- `ORPG_RELOAD=1` – enable automatic reload when running via `python rpg.py`.

Visit `http://localhost:8000/` for the web client. The UI attempts to detect a public URL via `/api/public_url` so you can quickly share a reachable host/IP with remote players.

## Gameplay Flow
1. A player joins from the lobby with a name/background. If the world is empty, the initial turn fires automatically to bootstrap the scenario.
2. Players type actions and hit **Submit**. The server queues actions per player and exposes them in the shared timeline.
3. Any player may press **Next Turn**. oRPG builds a structured payload with history, submissions, and per-player state, then calls the configured text model.
4. The model’s JSON response updates narrative, resolves player states, handles late joiners/leavers, and refreshes public status words.
5. WebSocket updates broadcast the new scenario, per-player private sheets, token/cost stats, and any media generated that turn.
6. When all players disconnect, the session automatically resets after a short delay.

Late joiners are flagged with `pending_join` so the next turn integrates them. If a player leaves mid-session, the server requests narrative closure and removes them cleanly afterwards.

Switching to `history_mode: "summary"` keeps only a rolling bullet chronicle so long-running games stay within model context limits.

## Media & Narration
- **Scene Images**: Toggle auto-generation in the UI or trigger manually with **Create Image**. Prompts merge the model’s response with saved portraits and status cues.
- **Portraits**: Players can request fresh portraits based on their current class/conditions via **Create Portrait**.
- **Narration**: Enable auto narration once an ElevenLabs key/model is configured. The server queues narration jobs, streams errors to clients, and tracks remaining credits when provided by the API.

All media toggles respect the global lock so narration and image requests never collide with turn resolution.

## Models & Cost Tracking
`/api/models` returns the available Gemini, Grok, and ElevenLabs models (when their respective keys are present). oRPG records token usage per turn, updates aggregate session metrics, and estimates USD costs using `model_prices.json`. Adjust that file if pricing changes or you want to add custom tiers.

## API Surface
| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the single-page application. |
| `GET` | `/api/state` | Public snapshot (turn, scenario, players, submissions, stats). |
| `GET` | `/api/settings` | Current settings (keys returned as stored). |
| `PUT` | `/api/settings` | Update settings and persist to disk. |
| `GET` | `/api/models` | List text and narration models for configured providers. |
| `GET` | `/api/public_url` | Provide a shareable host URL (falls back to placeholder). |
| `GET` | `/api/dev/text_inspect` | Last structured text request/response for debugging. |
| `POST` | `/api/join` | Register a new player (auto-starts world on first join). |
| `POST` | `/api/submit` | Submit a player action for the pending turn. |
| `POST` | `/api/next_turn` | Resolve the next turn using the configured text model. |
| `POST` | `/api/language` | Switch the active language (optionally requiring player auth). |
| `POST` | `/api/tts_toggle` | Enable/disable automatic ElevenLabs narration. |
| `POST` | `/api/image_toggle` | Enable/disable automatic scene images. |
| `POST` | `/api/create_image` | Request a new scene image immediately. |
| `POST` | `/api/create_portrait` | Refresh the requesting player’s portrait. |
| `WS` | `/ws` | Stream public state updates and private sheets to authenticated players. |

## Testing & Development
```bash
python -m pytest            # or python -m unittest
```

Tests reset the global state between cases and mock model calls. During development, the **Dev Info** panel in the UI and `/api/dev/text_inspect` expose the latest prompts/responses to help tune templates.

## Deployment & Security Notes
- Keep `settings.json` (`chmod 600`) and never commit keys. The project’s `.gitignore` already excludes it and `github_credentials.txt`.
- Run behind HTTPS and a reverse proxy when exposing the server to the internet.
- Rotate all API keys promptly if the host machine is compromised.

Happy adventuring!
