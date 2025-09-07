# oRPG

Ollama online RPG  is a small FastAPI application that turns a local LLM into a cooperative, turn-based text adventure. Players join with a name and backstory, receive an AI-chosen class and abilities, and submit actions that the AI resolves into the next scene.

## Features

- **Local Game Master** powered by [Ollama](https://ollama.ai/) (configurable model and host).
- **Multiplayer party** with AI-picked classes and flavored abilities.
- **Turn-based play**: players submit actions, the server narrates outcomes.
- **Minimal web client** served from the root endpoint.
- **REST API** endpoints for `/state`, `/join`, `/action`, `/resolve`, and `/healthz`.
- **In-memory game state**—no database required.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Start the server with:

```bash
python oRPG.py
```

or use Uvicorn directly:

```bash
uvicorn oRPG:app --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_HOST` | URL of the Ollama server | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Model name for the Game Master | `gpt-oss:20b` |
| `JOIN_CODE` | Optional code required to join | *(empty)* |
| `ALLOW_ANYONE_TO_RESOLVE` | Allow any player to trigger resolution (`1` or `0`) | `1` |
| `BIND_HOST` | Host interface to bind | `0.0.0.0` |
| `BIND_PORT` | Port to listen on | `8000` |

## Gameplay

1. Launch the server and open `http://<host>:8000` in a browser.
2. Each player provides a name, backstory, and optional join code.
3. Players submit actions each turn; the AI responds with the next scene and summary.

## Testing

Run the test suite with:

```bash
pytest
```

## Project Structure

```
├── oRPG.py          # FastAPI application and game logic
├── requirements.txt # Python dependencies
├── tests/           # Unit tests
└── README.md        # Project documentation
```

## Contributing

Pull requests and issues are welcome! Please ensure that all tests pass before submitting changes.

