from __future__ import annotations

import json
import time
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Union

from world.world import ResourceType
from .resources import ResourceManager
from .population import FactionManager
from .buildings import Building, ProcessingBuilding, BUILDING_ID_TO_CLASS
from .models import GreatProject, Faction, PROJECT_NAME_TO_CLASS

if TYPE_CHECKING:
    from world.world import World
    from .models import Faction as FactionModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SAVE_FILE: Path = Path("save.json")
TEMP_SAVE_FILE: Path = SAVE_FILE.with_suffix(".json.tmp")
MAX_TICKS_BATCH: int = 10_000  # Max ticks to simulate iteratively before switching to batched math
TICK_DURATION: int = 1  # Seconds per game tick


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------
class GameSaveError(Exception):
    """Exception raised when saving the game state fails."""


class GameLoadError(Exception):
    """Exception raised when loading the game state fails."""


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class GameState:
    timestamp: float
    resources: Dict[str, Dict[ResourceType, int]]
    population: int
    claimed_projects: List[str] = field(default_factory=list)
    world: Dict[str, Any] = field(default_factory=dict)
    factions: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    cooldowns: Dict[str, int] = field(default_factory=dict)
    roads: List[Any] = field(default_factory=list)
    event_turn_counters: Dict[str, int] = field(default_factory=dict)
    tech_levels: Dict[str, int] = field(default_factory=dict)
    god_powers: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"  # JSON schema version


@dataclass
class LoadResult:
    """Wrapper to allow unpacking ``load_state`` results flexibly."""
    state: GameState
    updates: Dict[str, Dict[str, int]]

    def __getattr__(self, item: str) -> Any:
        try:
            return getattr(self.state, item)
        except AttributeError as e:
            raise AttributeError(f"GameState has no attribute '{item}'. Key missing in save file?") from e

    def __iter__(self):
        return iter((self.state, self.updates))


# -----------------------------------------------------------------------------
# Serialization / Deserialization Helpers
# -----------------------------------------------------------------------------
def serialize_resources(data: Dict[str, Dict[ResourceType, int]]) -> dict:
    """Prepare nested resource data for JSON serialization."""
    return {
        faction_name: {res_type.value: count for res_type, count in faction_data.items()}
        for faction_name, faction_data in data.items()
    }


def deserialize_resources(data: Any) -> Dict[str, Dict[ResourceType, int]]:
    """Convert JSON resource mapping back into proper types."""
    result: Dict[str, Dict[ResourceType, int]] = {}
    if not isinstance(data, dict):
        return result

    for faction, res_dict in data.items():
        if not isinstance(res_dict, dict):
            continue
        result[faction] = {}
        for key, value in res_dict.items():
            try:
                res_type = ResourceType(key)
                res_count = int(value)
                result[faction][res_type] = res_count
            except (ValueError, TypeError):
                logging.warning(f"Skipping invalid resource entry: {faction} → {key}:{value}")
    return result


def serialize_world(world: "World") -> Dict[str, Any]:
    """Convert world state into a JSON-serializable structure."""
    serialized_hexes: Dict[str, Any] = {}
    for chunk in getattr(world, "chunks", {}).values():
        for row in chunk:
            for h in row:
                coord_key = f"{h.coord[0]},{h.coord[1]}"
                serialized_hexes[coord_key] = {
                    "terrain": h.terrain,
                    "flooded": bool(h.flooded),
                    "ruined": bool(h.ruined),
                    "lake": bool(h.lake),
                    "river": bool(h.river),
                }

    return {
        "settings": asdict(world.settings),
        "roads": [
            [int(c) for c in (r.start[0], r.start[1], r.end[0], r.end[1])]
            for r in getattr(world, "roads", [])
        ],
        "rivers": [
            [int(c) for c in (r.start[0], r.start[1], r.end[0], r.end[1])]
            for r in getattr(world, "rivers", [])
        ],
        "event_turn_counters": getattr(world, "event_turn_counters", {}),
        "god_powers": getattr(world, "god_powers", {}),
        "tech_levels": getattr(world, "tech_levels", {}),
        "hexes": serialized_hexes,
    }


def deserialize_world(data: Any, world: "World") -> None:
    """Apply saved world data to an existing World instance."""
    if not isinstance(data, dict):
        return

    from world.world import Road, RiverSegment

    # Rehydrate roads
    roads = data.get("roads")
    if isinstance(roads, list):
        reconstructed_roads = []
        for entry in roads:
            if isinstance(entry, list) and len(entry) == 4:
                try:
                    start = (int(entry[0]), int(entry[1]))
                    end = (int(entry[2]), int(entry[3]))
                    reconstructed_roads.append(Road(start, end))
                except (ValueError, TypeError):
                    logging.warning(f"Invalid road entry in save: {entry}")
        setattr(world, "roads", reconstructed_roads)

    # Rehydrate rivers
    rivers = data.get("rivers")
    if isinstance(rivers, list):
        reconstructed_rivers = []
        for entry in rivers:
            if isinstance(entry, list) and len(entry) == 4:
                try:
                    start = (int(entry[0]), int(entry[1]))
                    end = (int(entry[2]), int(entry[3]))
                    reconstructed_rivers.append(RiverSegment(start, end))
                except (ValueError, TypeError):
                    logging.warning(f"Invalid river entry in save: {entry}")
        setattr(world, "rivers", reconstructed_rivers)

    # Event turn counters
    etc = data.get("event_turn_counters")
    if isinstance(etc, dict):
        world.event_turn_counters = {str(k): int(v) for k, v in etc.items() if isinstance(v, (int, float))}

    # Tech levels
    tech_levels = data.get("tech_levels")
    if isinstance(tech_levels, dict):
        world.tech_levels = {str(k): int(v) for k, v in tech_levels.items() if isinstance(v, (int, float))}

    # God powers
    god_powers = data.get("god_powers")
    if isinstance(god_powers, dict):
        world.god_powers = god_powers

    # Hex states
    hexes = data.get("hexes")
    if isinstance(hexes, dict):
        for key, value in hexes.items():
            if not isinstance(value, dict):
                continue
            try:
                q_str, r_str = key.split(",")
                q, r = int(q_str), int(r_str)
            except (ValueError, TypeError):
                logging.warning(f"Skipping invalid hex coordinate key: {key}")
                continue

            hex_tile = world.get(q, r)
            if hex_tile is None:
                continue

            # Apply each attribute if present
            if "terrain" in value:
                hex_tile.terrain = value["terrain"]
            if "flooded" in value:
                hex_tile.flooded = bool(value["flooded"])
            if "ruined" in value:
                hex_tile.ruined = bool(value["ruined"])
            if "lake" in value:
                hex_tile.lake = bool(value["lake"])
            if "river" in value:
                hex_tile.river = bool(value["river"])


def serialize_factions(factions: List[FactionModel]) -> Dict[str, Any]:
    """Serialize faction state into a JSON-compatible structure."""
    result: Dict[str, Any] = {}
    for fac in factions:
        # Convert resource mapping to str → {res_type: count}
        resource_snapshot = {rt.value: count for rt, count in fac.resources.items()}

        # Serialize buildings as both id and name (if available)
        buildings_serialized: List[Dict[str, Union[str, int]]] = []
        for b in fac.buildings:
            entry: Dict[str, Union[str, int]] = {"level": b.level}
            # If the class has a CLASS_ID, store it.
            class_id = getattr(b, "CLASS_ID", None)
            if class_id is not None:
                entry["id"] = class_id
            # Always store the name for fallback
            entry["name"] = b.name
            buildings_serialized.append(entry)

        # Serialize projects
        projects_serialized = [
            {"name": p.name, "progress": p.progress} for p in fac.projects
        ]

        # Settlement info
        settlement_info = {
            "name": fac.settlement.name,
            "position": {"x": fac.settlement.position.x, "y": fac.settlement.position.y},
        }

        # Tech level
        tech_level_value: int
        if hasattr(fac, "tech_level") and hasattr(fac.tech_level, "value"):
            tech_level_value = int(fac.tech_level.value)
        else:
            tech_level_value = int(getattr(fac, "tech_level", 0))

        result[fac.name] = {
            "citizens": int(fac.citizens.count),
            "workers": int(fac.workers.assigned),
            "units": int(fac.units),
            "resources": resource_snapshot,
            "buildings": buildings_serialized,
            "projects": projects_serialized,
            "settlement": settlement_info,
            "tech_level": tech_level_value,
            "god_powers": getattr(fac, "god_powers", {}),
        }
    return result


def deserialize_factions(data: Any) -> Dict[str, Any]:
    """Deserialize saved faction data into raw dictionaries and reconstruct objects."""
    result: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return result

    for name, info in data.items():
        if not isinstance(info, dict):
            continue

        # Citizens, workers, units
        citizens = int(info.get("citizens", 0)) if isinstance(info.get("citizens"), (int, float)) else 0
        workers = int(info.get("workers", 0)) if isinstance(info.get("workers"), (int, float)) else 0
        units = int(info.get("units", 0)) if isinstance(info.get("units"), (int, float)) else 0

        # Reconstruct resources
        raw_res = info.get("resources", {})
        resources: Dict[ResourceType, int] = {}
        if isinstance(raw_res, dict):
            for k, v in raw_res.items():
                try:
                    res_type = ResourceType(k)
                    res_count = int(v)
                    resources[res_type] = res_count
                except (ValueError, TypeError):
                    logging.warning(f"Skipping invalid resource entry in faction '{name}': {k}:{v}")

        # Reconstruct buildings (as dicts for now; actual instantiation must happen elsewhere)
        buildings_data: List[Dict[str, Union[str, int]]] = []
        raw_buildings = info.get("buildings", [])
        if isinstance(raw_buildings, list):
            for binfo in raw_buildings:
                if not isinstance(binfo, dict):
                    continue
                entry: Dict[str, Union[str, int]] = {}
                # Accept either "id" or "name"
                if "id" in binfo:
                    entry["id"] = binfo["id"]
                if "name" in binfo:
                    entry["name"] = binfo["name"]
                entry["level"] = int(binfo.get("level", 0))
                buildings_data.append(entry)
        else:
            logging.warning(f"'buildings' for faction '{name}' is not a list—skipping")

        # Reconstruct projects (as dicts; instantiation elsewhere)
        projects_data: List[Dict[str, Union[str, int]]] = []
        raw_projects = info.get("projects", [])
        if isinstance(raw_projects, list):
            for pinfo in raw_projects:
                if not isinstance(pinfo, dict):
                    continue
                pname = pinfo.get("name")
                pprog = pinfo.get("progress", 0)
                if isinstance(pname, str) and isinstance(pprog, (int, float)):
                    projects_data.append({"name": pname, "progress": int(pprog)})
                else:
                    logging.warning(f"Skipping invalid project entry in faction '{name}': {pinfo}")
        else:
            logging.warning(f"'projects' for faction '{name}' is not a list—skipping")

        # Settlement
        raw_settlement = info.get("settlement", {})
        settlement_info: Dict[str, Any] = {}
        if isinstance(raw_settlement, dict):
            s_name = raw_settlement.get("name")
            s_pos = raw_settlement.get("position", {})
            if isinstance(s_name, str) and isinstance(s_pos, dict):
                x = s_pos.get("x", 0)
                y = s_pos.get("y", 0)
                try:
                    x_int = int(x)
                    y_int = int(y)
                    settlement_info = {"name": s_name, "position": {"x": x_int, "y": y_int}}
                except (ValueError, TypeError):
                    logging.warning(f"Invalid settlement position for faction '{name}': {s_pos}")
            else:
                logging.warning(f"Invalid settlement data for faction '{name}': {raw_settlement}")

        # Tech level
        raw_tech = info.get("tech_level", 0)
        tech_level_value = int(raw_tech) if isinstance(raw_tech, (int, float)) else 0

        # God powers
        god_powers = info.get("god_powers", {})

        result[name] = {
            "citizens": citizens,
            "workers": workers,
            "units": units,
            "resources": resources,
            "buildings": buildings_data,
            "projects": projects_data,
            "settlement": settlement_info,
            "tech_level": tech_level_value,
            "god_powers": god_powers,
        }

    return result


# -----------------------------------------------------------------------------
# Tick Simulation and Offline Gains
# -----------------------------------------------------------------------------
def simulate_tick(
    factions: List[FactionModel],
    pop_mgr: FactionManager,
    res_mgr: ResourceManager,
    cooldowns: Dict[str, int] | None = None,
) -> None:
    """Advance one tick for offline gains."""
    pop_mgr.tick()

    # Process each processing building for each faction
    for fac in factions:
        for b in getattr(fac, "buildings", []):
            if isinstance(b, ProcessingBuilding):
                b.process(fac)

        # Update resource snapshot for this faction
        if fac.name in res_mgr.data:
            res_mgr.data[fac.name].update(fac.resources)
        else:
            res_mgr.data[fac.name] = fac.resources.copy()

    res_mgr.tick(factions)

    # Decrement cooldowns if present
    if cooldowns is not None:
        for key in list(cooldowns.keys()):
            if cooldowns[key] > 0:
                cooldowns[key] -= 1


def apply_offline_gains(
    state: GameState,
    world: "World" | None,
    factions: List[FactionModel] | None,
) -> Dict[str, Dict[str, int]]:
    """
    Apply offline gains to the given state using provided world and factions.

    Returns:
        population_updates: Mapping from faction name to updated counts.
    """
    population_updates: Dict[str, Dict[str, int]] = {}
    if world is None or factions is None:
        return population_updates

    now = time.time()
    elapsed_seconds = int((now - state.timestamp) // TICK_DURATION)
    if elapsed_seconds <= 0:
        # Nothing to simulate
        state.timestamp = now
        return population_updates

    # Initialize managers
    res_mgr = ResourceManager(world, state.resources)
    pop_mgr = FactionManager(factions)

    # Restore each faction's saved resources so ResourceManager can resume
    for fac in factions:
        saved_res = state.resources.get(fac.name)
        if isinstance(saved_res, dict):
            fac.resources = {res_type: qty for res_type, qty in saved_res.items()}
            res_mgr.data[fac.name] = saved_res.copy()


    # If elapsed_seconds is large, batch up to MAX_TICKS_BATCH
    ticks_to_simulate = elapsed_seconds
    if ticks_to_simulate > MAX_TICKS_BATCH:
        # Handle the first MAX_TICKS_BATCH iteratively to trigger any state changes
        for _ in range(MAX_TICKS_BATCH):
            simulate_tick(factions, pop_mgr, res_mgr, state.cooldowns)
            for fac in factions:
                fac.progress_projects()

        # Compute the “steady-state” per-tick gains for each faction
        per_tick_production: Dict[str, Dict[ResourceType, int]] = {}
        for fac in factions:
            # Assume ResourceManager has a method .get_per_tick_output(fac)
            gains: Dict[ResourceType, int] = res_mgr.get_per_tick_output(fac)
            per_tick_production[fac.name] = gains

        remaining_ticks = ticks_to_simulate - MAX_TICKS_BATCH
        # Batch-apply (remaining_ticks × gain) to each faction’s resources & projects
        for fac in factions:
            # Resources
            cur_res = res_mgr.data.get(fac.name, {})
            incremental = per_tick_production.get(fac.name, {})
            for rtype, amount_per_tick in incremental.items():
                total_gain = amount_per_tick * remaining_ticks
                cur_res[rtype] = cur_res.get(rtype, 0) + total_gain
            res_mgr.data[fac.name] = cur_res

            # Projects (assume each project advances by 1 progress per tick)
            total_project_ticks = remaining_ticks
            for project in fac.projects:
                project.progress += total_project_ticks
                if project.progress > project.build_time:
                    project.progress = project.build_time

        # Simulate project advancement side-effects if needed
        for fac in factions:
            fac.finalize_project_states()

    else:
        # Normal case (small elapsed), simulate each tick
        for _ in range(ticks_to_simulate):
            simulate_tick(factions, pop_mgr, res_mgr, state.cooldowns)
            for fac in factions:
                fac.progress_projects()

    # Update the state’s resource and population data
    state.resources = res_mgr.data.copy()
    state.population = sum(f.citizens.count for f in factions)

    # Build population_updates for return
    for fac in factions:
        population_updates[fac.name] = {
            "citizens": fac.citizens.count,
            "workers": fac.workers.assigned,
        }

    state.factions = serialize_factions(factions)
    state.timestamp = now
    return population_updates


# -----------------------------------------------------------------------------
# Loading and Saving State
# -----------------------------------------------------------------------------
def load_state(
    *,
    world: Optional["World"] = None,
    factions: Optional[List[FactionModel]] = None,
    strict: bool = False,
    file_path: Optional[Path] = None,
) -> LoadResult:
    """
    Load the saved game state and optionally apply offline gains.

    Args:
        world: If provided, apply saved world data into this instance.
        factions: If provided, populate these faction objects from saved data,
                  then apply offline gains on them.
        strict: If True, treat missing or extra keys in save data as errors.

    Returns:
        A LoadResult containing (GameState, population_updates).
    """
    now = time.time()
    path = file_path or SAVE_FILE

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise GameLoadError(f"Failed to read or parse save file: {e}") from e

        # Optionally validate schema version
        version = raw_data.get("version", "0.0")
        if strict and version != "1.0":
            raise GameLoadError(f"Unsupported save version: {version}. Expected 1.0.")

        # Deserialize resources
        resources = deserialize_resources(raw_data.get("resources", {}))

        # Compute how many seconds have passed since last save
        saved_timestamp = raw_data.get("timestamp", now)
        try:
            saved_timestamp = float(saved_timestamp)
        except (ValueError, TypeError):
            saved_timestamp = now

        # Recompute cooldowns
        raw_cooldowns = raw_data.get("cooldowns", {})
        cooldowns: Dict[str, int] = {}
        if isinstance(raw_cooldowns, dict):
            for k, v in raw_cooldowns.items():
                try:
                    remaining = int(v) - int((now - saved_timestamp) // TICK_DURATION)
                    cooldowns[str(k)] = max(0, remaining)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid cooldown entry: {k}:{v}")
        else:
            logging.warning("'cooldowns' in save file is not a dict; resetting.")

        # Build initial GameState
        state = GameState(
            timestamp=saved_timestamp,
            resources=resources,
            population=int(raw_data.get("population", 0)) if isinstance(raw_data.get("population"), (int, float)) else 0,
            claimed_projects=list(raw_data.get("claimed_projects", [])) if isinstance(raw_data.get("claimed_projects"), list) else [],
            world=raw_data.get("world", {}) if isinstance(raw_data.get("world"), dict) else {},
            factions=raw_data.get("factions", {}) if isinstance(raw_data.get("factions"), dict) else {},
            turn=int(raw_data.get("turn", 0)) if isinstance(raw_data.get("turn"), (int, float)) else 0,
            cooldowns=cooldowns,
            roads=list(raw_data.get("roads", [])) if isinstance(raw_data.get("roads"), list) else [],
            event_turn_counters={k: int(v) for k, v in raw_data.get("event_turn_counters", {}).items() if isinstance(v, (int, float))},
            tech_levels={k: int(v) for k, v in raw_data.get("tech_levels", {}).items() if isinstance(v, (int, float))},
            god_powers=raw_data.get("god_powers", {}) if isinstance(raw_data.get("god_powers"), dict) else {},
            version=version,
        )

        # If a World instance is provided, apply saved world data
        if world is not None:
            deserialize_world(state.world, world)
            state.world = world

        # If faction objects are provided, rehydrate them from saved data
        population_updates: Dict[str, Dict[str, int]] = {}
        if factions is not None:
            raw_factions = raw_data.get("factions", {})
            rehydrated = deserialize_factions(raw_factions)
            name_to_faction_obj: Dict[str, FactionModel] = {fac.name: fac for fac in factions}

            for fac_name, fac_data in rehydrated.items():
                fac_obj = name_to_faction_obj.get(fac_name)
                if fac_obj is None:
                    logging.warning(f"No in-memory faction matches saved name '{fac_name}'—skipping.")
                    continue

                # Restore resource dictionary
                fac_obj.resources = fac_data.get("resources", {}).copy()

                # Rebuild and attach buildings
                fac_obj.buildings.clear()
                for binfo in fac_data.get("buildings", []):
                    # Try CLASS_ID first
                    new_building: Optional[Building] = None
                    if "id" in binfo:
                        cls = BUILDING_ID_TO_CLASS.get(binfo["id"])
                        if cls is not None:
                            new_building = cls()
                    if new_building is None and "name" in binfo:
                        # Fall back to name-based lookup
                        cls = Building.get_class_by_name(binfo["name"])
                        if cls is not None:
                            new_building = cls()
                    if new_building is not None:
                        new_building.level = int(binfo.get("level", 0))
                        fac_obj.buildings.append(new_building)
                    else:
                        logging.warning(f"Could not rehydrate building for faction '{fac_name}': {binfo}")

                # Rebuild and attach projects
                fac_obj.projects.clear()
                for pinfo in fac_data.get("projects", []):
                    pname = pinfo.get("name")
                    pprog = pinfo.get("progress", 0)
                    project_class = PROJECT_NAME_TO_CLASS.get(pname)
                    if project_class is not None:
                        proj = project_class()
                        proj.progress = int(pprog)
                        fac_obj.projects.append(proj)
                    else:
                        logging.warning(f"Unknown project '{pname}' for faction '{fac_name}'—skipping.")

                # Restore settlement position (the game code should validate position exists)
                sett_info = fac_data.get("settlement", {})
                if (
                    isinstance(sett_info, dict)
                    and "name" in sett_info
                    and isinstance(sett_info.get("position", {}), dict)
                ):
                    pos_dict = sett_info["position"]
                    try:
                        x_val = int(pos_dict.get("x", 0))
                        y_val = int(pos_dict.get("y", 0))
                        fac_obj.settlement.name = sett_info["name"]
                        fac_obj.settlement.position.x = x_val
                        fac_obj.settlement.position.y = y_val
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid settlement coords for faction '{fac_name}': {pos_dict}")

                # Restore tech level
                try:
                    fac_obj.tech_level = type(fac_obj.tech_level)(int(fac_data.get("tech_level", 0)))
                except (ValueError, TypeError):
                    logging.warning(f"Invalid tech_level for faction '{fac_name}': {fac_data.get('tech_level')}")

                # Restore god powers
                fac_obj.god_powers = fac_data.get("god_powers", {}).copy()

            # Now apply offline gains
            population_updates = apply_offline_gains(state, world, factions)

        return LoadResult(state=state, updates=population_updates)

    else:
        # No save file exists: create an initial empty state
        state = GameState(
            timestamp=now,
            resources={},
            population=0,
            claimed_projects=[],
            world={},
            factions={},
            turn=0,
            cooldowns={},
            roads=[],
            event_turn_counters={},
            tech_levels={},
            god_powers={},
            version="1.0",
        )
        population_updates: Dict[str, Dict[str, int]] = {}
        if factions is not None and world is not None:
            population_updates = apply_offline_gains(state, world, factions)
        return LoadResult(state=state, updates=population_updates)


def save_state(state: GameState, *, file_path: Optional[Path] = None) -> None:
    """
    Persist the current game state to disk in an atomic manner.

    Raises:
        GameSaveError: if writing or renaming fails.
    """
    state.timestamp = time.time()
    path = file_path or SAVE_FILE
    temp_file = path.with_suffix(".json.tmp")
    # Prepare data dict
    data = {
        "version": state.version,
        "timestamp": state.timestamp,
        "resources": serialize_resources(state.resources),
        "population": int(state.population),
        "claimed_projects": list(state.claimed_projects),
        "world": state.world,
        "factions": state.factions,
        "turn": int(state.turn),
        "cooldowns": {str(k): int(v) for k, v in state.cooldowns.items()},
        "roads": state.roads,
        "event_turn_counters": {str(k): int(v) for k, v in state.event_turn_counters.items()},
        "tech_levels": {str(k): int(v) for k, v in state.tech_levels.items()},
        "god_powers": state.god_powers,
    }

    # Write to a temporary file first
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            f.truncate()
    except OSError as e:
        raise GameSaveError(f"Failed to write to temporary save file: {e}") from e

    # Atomically move temp -> final
    try:
        shutil.move(str(temp_file), str(path))
    except OSError as e:
        # Attempt to remove leftover temp file, but do not mask original error
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass
        raise GameSaveError(f"Failed to rename temporary save file to final: {e}") from e
