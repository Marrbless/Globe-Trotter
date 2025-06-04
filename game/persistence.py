import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from world.world import ResourceType
from .resources import ResourceManager

if TYPE_CHECKING:
    from .game import Faction
    from world.world import World

SAVE_FILE = Path("save.json")
TICK_DURATION = 1  # seconds per tick


@dataclass
class GameState:
    timestamp: float
    resources: Dict[str, Dict[ResourceType, int]]
    population: int
    world: Dict[str, Any] = field(default_factory=dict)
    factions: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0


def serialize_resources(data: Dict[str, Dict[ResourceType, int]]) -> dict:
    """Prepare nested resource data for JSON serialization."""
    return {f: {k.value: v for k, v in res.items()} for f, res in data.items()}


def deserialize_resources(data: Any) -> Dict[str, Dict[ResourceType, int]]:
    """Convert JSON resource mapping back into proper types."""
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Dict[ResourceType, int]] = {}
    for faction, res in data.items():
        if isinstance(res, dict):
            result[faction] = {ResourceType(k): int(v) for k, v in res.items()}
    return result


def serialize_world(world: "World") -> Dict[str, Any]:
    """Convert world state into a JSON serializable structure."""
    return {
        "settings": asdict(world.settings),
        "roads": [list(r.start + r.end) for r in getattr(world, "roads", [])],
        "rivers": [list(r.start + r.end) for r in getattr(world, "rivers", [])],
        "hexes": {
            f"{q},{r}": {"flooded": h.flooded, "ruined": h.ruined}
            for r, row in enumerate(getattr(world, "hexes", []))
            for q, h in enumerate(row)
            if h.flooded or h.ruined
        },
    }


def deserialize_world(data: Any, world: "World") -> None:
    """Apply saved world data to an existing World instance."""
    if not isinstance(data, dict):
        return
    from world.world import Road, RiverSegment

    roads = data.get("roads")
    if isinstance(roads, list):
        world.roads = [
            Road(tuple(r[:2]), tuple(r[2:])) for r in roads if isinstance(r, list) and len(r) == 4
        ]

    rivers = data.get("rivers")
    if isinstance(rivers, list):
        world.rivers = [
            RiverSegment(tuple(r[:2]), tuple(r[2:])) for r in rivers if isinstance(r, list) and len(r) == 4
        ]

    hexes = data.get("hexes")
    if isinstance(hexes, dict):
        for key, value in hexes.items():
            try:
                q, r = map(int, key.split(","))
            except ValueError:
                continue
            hex_ = world.get(q, r)
            if hex_ and isinstance(value, dict):
                if "flooded" in value:
                    hex_.flooded = bool(value["flooded"])
                if "ruined" in value:
                    hex_.ruined = bool(value["ruined"])


def serialize_factions(factions: List["Faction"]) -> Dict[str, Any]:
    """Serialize faction state."""
    result: Dict[str, Any] = {}
    for fac in factions:
        result[fac.name] = {
            "citizens": fac.citizens.count,
            "workers": fac.workers.assigned,
            "buildings": [{"name": b.name, "level": b.level} for b in fac.buildings],
            "projects": [{"name": p.name, "progress": p.progress} for p in fac.projects],
        }
    return result


def deserialize_factions(data: Any) -> Dict[str, Any]:
    """Deserialize saved faction data."""
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Any] = {}
    for name, info in data.items():
        if not isinstance(info, dict):
            continue
        result[name] = {
            "citizens": int(info.get("citizens", 0)),
            "workers": int(info.get("workers", 0)),
            "buildings": info.get("buildings", []),
            "projects": info.get("projects", []),
        }
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
            world=data.get("world", {}),
            factions=deserialize_factions(data.get("factions", {})),
            turn=int(data.get("turn", 0)),
        )
    else:
        state = GameState(timestamp=now, resources={}, population=0, world={}, factions={}, turn=0)

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
            "world": state.world,
            "factions": state.factions,
            "turn": state.turn,
        }
        json.dump(data, f)
