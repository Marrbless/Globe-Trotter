import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .resources import ResourceManager

if TYPE_CHECKING:
    from .resources import ResourceManager
    from .game import Faction
    from world.world import World


SAVE_FILE = Path("save.json")
TICK_DURATION = 1  # seconds per tick


@dataclass
class GameState:
    timestamp: float
    resources: Dict[str, Dict[str, int]]
    population: int


def serialize_resources(data: Dict[str, Dict[str, int]]) -> dict:
    """Prepare nested resource data for JSON serialization."""
    return {f: dict(res) for f, res in data.items()}


def deserialize_resources(data: Any) -> Dict[str, Dict[str, int]]:
    """Convert JSON resource mapping back into proper types."""
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Dict[str, int]] = {}
    for faction, res in data.items():
        if isinstance(res, dict):
            result[faction] = {k: int(v) for k, v in res.items()}
    return result


def load_state(
    *,
    world: Optional["World"] = None,
    factions: Optional[List["Faction"]] = None,
) -> GameState:
    """Load the saved game state and optionally apply offline gains."""
    now = time.time()
    if SAVE_FILE.exists():
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        resources = deserialize_resources(data.get("resources", {}))
        state = GameState(
            timestamp=data.get("timestamp", now),
            resources=resources,
            population=data.get("population", 0),
        )
    else:
        state = GameState(timestamp=now, resources={}, population=0)

    elapsed = int((now - state.timestamp) // TICK_DURATION)

    if elapsed > 0 and world is not None and factions is not None:
        mgr = ResourceManager(world, state.resources)
        for _ in range(elapsed):
            mgr.tick(factions)
            for fac in factions:
                fac.citizens.count += 1
            state.population += 1
        state.resources = mgr.data

    state.timestamp = now
    return state


def save_state(state: GameState) -> None:
    """Persist the current game state to disk."""
    state.timestamp = time.time()
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        data = {
            "timestamp": state.timestamp,
            "resources": serialize_resources(state.resources),
            "population": state.population,
        }
        json.dump(data, f)
