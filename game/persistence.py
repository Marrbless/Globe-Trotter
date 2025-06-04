import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


SAVE_FILE = Path("save.json")
TICK_DURATION = 1  # seconds per tick


@dataclass
class GameState:
    timestamp: float
    resources: int
    population: int


def load_state() -> GameState:
    """Load the saved game state and grant offline gains."""
    now = time.time()
    if SAVE_FILE.exists():
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = GameState(**data)
        elapsed = now - state.timestamp
        ticks = int(elapsed // TICK_DURATION)
        if ticks > 0:
            state.resources += state.population * ticks
            state.timestamp = now
    else:
        state = GameState(timestamp=now, resources=0, population=0)
    return state


def save_state(state: GameState) -> None:
    """Persist the current game state to disk."""
    state.timestamp = time.time()
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f)
