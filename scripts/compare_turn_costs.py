import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import rpg


DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gpt-5-mini",
    "grok-4-fast-non-reasoning",
]

DEFAULT_PLAYERS = [
    {"id": "p1", "name": "Aela", "background": "Sylvan scout"},
    {"id": "p2", "name": "Borin", "background": "Forge cleric"},
]

DEFAULT_SUBMISSIONS = [
    {
        "p1": "Aela studies the glowing runes and searches for hidden mechanisms.",
        "p2": "Borin invokes a warding prayer to shield the party from dark magic.",
    },
    {
        "p1": "Aela scales the shattered tower to scout the enemy's position.",
        "p2": "Borin charges forward, hammer blazing with consecrated fire.",
    },
]


def reset_state() -> None:
    """Return the global game state to a clean baseline without touching saved settings."""

    rpg.STATE.settings = rpg.load_settings()
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
    rpg.STATE.session_image_cost_usd = 0.0
    rpg.STATE.session_image_requests = 0
    rpg.STATE.session_image_kind_counts = {}
    rpg.STATE.turn_image_kind_counts = {}
    rpg.STATE.last_scene_image_cost_usd = None
    rpg.STATE.last_scene_image_usd_per = None
    rpg.STATE.last_scene_image_model = None
    rpg.STATE.last_scene_image_turn_index = None
    rpg.STATE.last_token_usage = {}
    rpg.STATE.session_token_usage = {"input": 0, "output": 0, "thinking": 0}
    rpg.STATE.last_turn_runtime = None
    rpg.STATE.session_request_count = 0
    rpg.STATE.last_cost_usd = None
    rpg.STATE.session_cost_usd = 0.0
    rpg.STATE.auto_image_enabled = False
    rpg.STATE.auto_tts_enabled = False
    rpg.STATE.last_text_request = {}
    rpg.STATE.last_text_response = {}
    rpg.STATE.last_scenario_request = {}
    rpg.STATE.last_scenario_response = {}


def apply_players(players: Iterable[Dict[str, Any]]) -> None:
    for entry in players:
        pid = entry["id"]
        rpg.STATE.players[pid] = rpg.Player(
            id=pid,
            name=entry.get("name", pid),
            background=entry.get("background", ""),
            token=f"token-{pid}",
            pending_join=True,
            pending_leave=False,
            connected=True,
            status_word="unknown",
        )


def apply_settings(model_name: str, thinking_mode: str) -> None:
    settings = rpg.STATE.settings.copy()
    settings["text_model"] = model_name
    settings["thinking_mode"] = thinking_mode
    settings["history_mode"] = rpg.HISTORY_MODE_FULL
    rpg.STATE.settings = settings
    rpg.STATE.language = rpg.normalize_language(settings.get("language"))


def record_submissions(turn_idx: int, submissions: List[Dict[str, str]]) -> None:
    rpg.STATE.submissions.clear()
    if turn_idx == 0:
        return
    payload = submissions[turn_idx - 1] if turn_idx - 1 < len(submissions) else {}
    for pid, text in payload.items():
        rpg.STATE.submissions[pid] = text


async def simulate_model(
    model_name: str,
    *,
    thinking_mode: str,
    turns: int,
    players: Iterable[Dict[str, Any]],
    submissions: List[Dict[str, str]],
) -> Dict[str, Any]:
    reset_state()
    apply_players(players)
    apply_settings(model_name, thinking_mode)

    for turn_idx in range(turns):
        record_submissions(turn_idx, submissions)
        await rpg.resolve_turn(initial=(turn_idx == 0))

    return {
        "model": model_name,
        "cost_usd": rpg.STATE.session_cost_usd,
        "token_usage": dict(rpg.STATE.session_token_usage),
        "requests": rpg.STATE.session_request_count,
        "elapsed_seconds": rpg.STATE.last_turn_runtime,
        "thinking_mode": thinking_mode,
        "last_thinking_budget": rpg.STATE.last_text_request.get("thinking_budget"),
    }


async def run_suite(
    models: Iterable[str],
    *,
    thinking_mode: str,
    turns: int,
    players: Iterable[Dict[str, Any]],
    submissions: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model in models:
        start = time.time()
        result = await simulate_model(
            model,
            thinking_mode=thinking_mode,
            turns=turns,
            players=players,
            submissions=submissions,
        )
        result["elapsed_wall_seconds"] = time.time() - start
        results.append(result)

    min_cost = min((r["cost_usd"] for r in results), default=0.0)
    for entry in results:
        cost = entry.get("cost_usd")
        if isinstance(cost, (int, float)) and cost > 0 and min_cost > 0:
            entry["relative_percent"] = (cost / min_cost) * 100.0
        elif cost == min_cost:
            entry["relative_percent"] = 100.0
        else:
            entry["relative_percent"] = None
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare turn costs across text models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model identifiers to exercise (default: Gemini flash, GPT-5 mini, Grok fast).",
    )
    parser.add_argument(
        "--thinking-mode",
        default="balanced",
        choices=sorted(rpg.THINKING_MODES),
        help="Thinking mode to set in settings.json before each run.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of turns to simulate per model (default: 3).",
    )
    parser.add_argument(
        "--output",
        choices=["json"],
        default="json",
        help="Output format (currently json only).",
    )
    parser.add_argument(
        "--include-grok-reasoning",
        action="store_true",
        help="Automatically append grok-4-fast-reasoning to the model list if not present.",
    )
    return parser.parse_args()


def expand_models(models: Iterable[str], include_grok_reasoning: bool) -> List[str]:
    expanded = list(models)
    if include_grok_reasoning and "grok-4-fast-reasoning" not in expanded:
        expanded.append("grok-4-fast-reasoning")
    return expanded


def main() -> None:
    args = parse_args()
    models = expand_models(args.models, args.include_grok_reasoning)

    results = asyncio.run(
        run_suite(
            models,
            thinking_mode=args.thinking_mode,
            turns=args.turns,
            players=DEFAULT_PLAYERS,
            submissions=DEFAULT_SUBMISSIONS,
        )
    )

    if args.output == "json":
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
