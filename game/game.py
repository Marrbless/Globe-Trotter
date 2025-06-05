import argparse
import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from .persistence import (
    GameState,
    load_state,
    save_state,
    serialize_world,
    serialize_factions,
    deserialize_world,
    apply_offline_gains,
)
from .diplomacy import TradeDeal, Truce, DeclarationOfWar, Alliance
from . import ai
from .buildings import ALL_BUILDING_CLASSES
from .buildings import (
    Building,
    ProcessingBuilding,
    mitigate_building_damage,
    mitigate_population_loss,
)
from .population import Citizen, Worker, FactionManager
from . import settings
from world.world import World, ResourceType, WorldSettings, Road
from .resources import ResourceManager
from .models import Position, Settlement, GreatProject, Faction
from .god_powers import ALL_POWERS, GodPower

logger = logging.getLogger("mygame.Game")
logger.addHandler(logging.NullHandler())

# Type aliases for clarity
FactionName = str
ResourceDict = Dict[ResourceType, int]
SavedFactionsData = Dict[FactionName, Dict[str, Any]]

# --------------------------------------------------------------------
# Predefined templates for high-cost “Great Projects”
# --------------------------------------------------------------------
GREAT_PROJECT_TEMPLATES: Dict[str, GreatProject] = {
    "Grand Cathedral": GreatProject(
        name="Grand Cathedral",
        build_time=int(5 * settings.SCALE_FACTOR),
        victory_points=10,
        bonus="Increases faith across the realm",
    ),
    "Sky Fortress": GreatProject(
        name="Sky Fortress",
        build_time=int(8 * settings.SCALE_FACTOR),
        victory_points=15,
        bonus="Provides unmatched military power",
    ),
    "Great Dam": GreatProject(
        name="Great Dam",
        build_time=int(6 * settings.SCALE_FACTOR),
        victory_points=12,
        bonus="Controls flooding and provides power",
    ),
}

# Mapping from project name → unlocked “god power” action key
PROJECT_UNLOCKS: Dict[str, str] = {
    "Grand Cathedral": "celebrate_festival",
    "Sky Fortress": "air_strike",
    "Great Dam": "control_floods",
}


def apply_project_bonus(faction: Faction, project: GreatProject) -> None:
    """
    When a GreatProject is completed, append its associated action
    (from PROJECT_UNLOCKS) into the faction’s unlocked_actions list.
    """
    action = PROJECT_UNLOCKS.get(project.name)
    if action and action not in faction.unlocked_actions:
        faction.unlocked_actions.append(action)

    if project.name == "Great Dam" and faction.world is not None:
        pos = faction.settlement.position
        for nq, nr in faction.world._neighbors(pos.x, pos.y):
            hex_ = faction.world.get(nq, nr)
            if hex_ and hex_.river:
                hex_.river = False
                hex_.lake = True
                hex_.terrain = "water"
                faction.world.rivers = [
                    seg
                    for seg in faction.world.rivers
                    if seg.start != (nq, nr) and seg.end != (nq, nr)
                ]
                if (nq, nr) not in faction.world.lakes:
                    faction.world.lakes.append((nq, nr))
                break


# --------------------------------------------------------------------
# “Map” Class: Manages hex-based placement of factions & distance logic
# --------------------------------------------------------------------
class Map:
    """
    Manages placement of factions on a hex map and computes distances.
    """
    def __init__(self, width: int, height: int, world: Optional[World] = None):
        self.width = width
        self.height = height
        self.world = world
        self.factions: List[Faction] = []

    def is_occupied(self, position: Position) -> bool:
        for faction in self.factions:
            if faction.settlement.position == position:
                return True
        return False

    def distance(self, pos1: Position, pos2: Position) -> int:
        """
        Return axial hex distance between two positions.
        (|dq| + |dr| + |dq + dr|) // 2 = hex distance in axial coords.
        """
        dq = pos1.x - pos2.x
        dr = pos1.y - pos2.y
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    def add_faction(self, faction: Faction) -> None:
        if not self.is_occupied(faction.settlement.position):
            self.factions.append(faction)
        else:
            raise ValueError("Position is already occupied")

    def spawn_ai_factions(self, player_settlement: Settlement) -> List[Faction]:
        """
        Spawn exactly `settings.AI_FACTION_COUNT` AI factions, ensuring each
        spawn point is at least `settings.MIN_DISTANCE_FROM_PLAYER` hexes away
        from `player_settlement`. Attempts up to
        `settings.AI_SPAWN_MAX_ATTEMPTS_MULTIPLIER * count` in total.
        Returns the list of newly created AI Faction objects.
        """
        max_tries = settings.AI_SPAWN_MAX_ATTEMPTS_MULTIPLIER * settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
        new_factions: List[Faction] = []

        while spawned < settings.AI_FACTION_COUNT and attempts < max_tries:
            attempts += 1
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)

            if (
                not self.is_occupied(pos)
                and self.distance(pos, player_settlement.position)
                >= settings.MIN_DISTANCE_FROM_PLAYER
            ):
                ai_name = f"AI #{spawned + 1}"
                ai_faction = Faction(
                    name=ai_name,
                    settlement=Settlement(name=f"AI Town {spawned + 1}", position=pos),
                    citizens=Citizen(count=random.randint(8, 15)),
                    world=self.world,
                )
                self.add_faction(ai_faction)
                new_factions.append(ai_faction)
                spawned += 1
                # Immediately register the AI faction with managers:
                # (Assuming caller will call `_register_faction` for each new faction)

        if spawned < settings.AI_FACTION_COUNT:
            logger.warning(
                "Only spawned %d/%d AI factions after %d attempts",
                spawned, settings.AI_FACTION_COUNT, attempts
            )

        return new_factions


# --------------------------------------------------------------------
# “Game” Class: Core simulation, state management, diplomacy, & turns
# --------------------------------------------------------------------
class Game:
    """
    Core game class handling:
      - World initialization & serialization
      - Faction registration (resources, population, buildings, projects)
      - Diplomacy (trade, wars, truces, alliances)
      - “God Powers” (unlocking, cooldown tracking, application)
      - Turn advancement, scoring, & event simulation
    """
    def __init__(self, state: Optional[GameState] = None, world: Optional[World] = None):
        # 1) Initialize map- and world-level data
        self.world = world or World(*settings.MAP_SIZE)
        self.map = Map(self.world.width, self.world.height, world=self.world)
        # Guarantee a clean slate of factions on initialization
        self.map.factions = []

        # 2) Load or create fresh GameState container
        self.state: GameState = state or GameState(
            timestamp=time.time(), resources={}, population=0, claimed_projects=[]
        )

        # 3) Resource and population managers
        self.resources = ResourceManager(self.world, self.state.resources)
        self.faction_manager = FactionManager()
        self.population: int = self.state.population
        self.claimed_projects: set[str] = set(self.state.claimed_projects)

        # 4) Placeholder for the player’s faction & buildings they own
        self.player_faction: Optional[Faction] = None
        self.player_buildings: List[Building] = []

        # 5) Turn counter
        self.turn: int = 0

        # 6) Diplomacy / conflict tracking
        self.trade_deals: List[TradeDeal] = []
        self.truces: List[Truce] = []
        self.wars: List[DeclarationOfWar] = []

        # 7) Event system tracking (e.g., disasters, raids)
        self.event_turn_counters: Dict[str, int] = {}

        # 8) “Largest Army” & “Longest Road” leader tracking
        self.leaders: Dict[str, Optional[str]] = {
            "largest_army": None,
            "longest_road": None,
        }

        # 9) All possible GodPower definitions
        self.god_powers: Dict[str, GodPower] = {p.name: p for p in ALL_POWERS}
        # 10) Cooldowns for each GodPower (0 means “ready to use”)
        self.power_cooldowns: Dict[str, int] = {name: 0 for name in self.god_powers}

        # 11) List of active alliances
        self.alliances: List[Alliance] = []

        # 12) Prevent duplicate faction registration
        self._registered_factions: set[str] = set()

        # 13) If loading from saved state, restore any stored cooldowns
        if getattr(self.state, "cooldowns", None):
            self.power_cooldowns.update(self.state.cooldowns)

    def place_initial_settlement(self, x: int, y: int, name: str = "Player") -> None:
        """
        Place the player's home settlement at (x, y). Errors if occupied.
        Also registers the player faction with both ResourceManager and FactionManager.
        """
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")

        settlement = Settlement(name=name, position=pos)
        self.player_faction = Faction(name=name, settlement=settlement, world=self.world)
        self.map.add_faction(self.player_faction)
        self._register_faction(self.player_faction)
        self.population = self.player_faction.citizens.count

    def _register_faction(self, faction: Faction) -> None:
        """
        Helper to register a faction (both with ResourceManager and FactionManager),
        but avoid duplicates. Silently ignores if already registered.
        """
        if faction.name in self._registered_factions:
            return

        try:
            self.resources.register(faction)
        except ValueError:
            logger.debug("Faction %s already registered in ResourceManager", faction.name)

        try:
            self.faction_manager.add_faction(faction)
        except ValueError:
            logger.debug("Faction %s already registered in FactionManager", faction.name)

        self._registered_factions.add(faction.name)

    def begin(self) -> None:
        """
        Initialize or restore a saved game state. Steps:
          1. Ensure the player faction is already placed.
          2. Attempt to load & deserialize any saved world, roads, and factions.
          3. If loaded: reconstruct saved factions (AI + player) via `_restore_factions_from_state()`.
          4. If fresh: spawn AI factions using `spawn_ai_factions()`.
          5. If loaded: reinstantiate ResourceManager data, claimed projects, cooldowns.
          6. Apply offline gains (resource & population) if loading.
          7. Register & restore each faction’s data (resources, buildings, projects, diplomacy).
          8. Set turn counter & player population, log the scenario, then simulate events & update leaders.
        """
        if not self.player_faction:
            raise RuntimeError("Player settlement must be placed prior to calling begin()")

        # 1) Attempt to load any saved GameState (and rebuild world + roads)
        loaded_state = self._load_and_deserialize_world()

        # 2) Clear any existing factions to prepare for restoration or fresh spawns
        self.map.factions = []

        if loaded_state:
            # 3) Rebuild every faction (including the player) from serialized data
            self._restore_factions_from_state()
        else:
            # 4) Fresh game → spawn AI around the player's position
            ai_factions = self.map.spawn_ai_factions(self.player_faction.settlement)
            for ai_faction in ai_factions:
                self._register_faction(ai_faction)

        # 5) If we loaded saved state, restore ResourceManager data, claimed projects, cooldowns
        if loaded_state:
            self.resources = ResourceManager(self.world, self.state.resources)
            self.claimed_projects = set(self.state.claimed_projects)
            if getattr(self.state, "cooldowns", None):
                self.power_cooldowns.update(self.state.cooldowns)

        # 6) Apply offline gains if loading from a prior save
        updated_pops: Dict[str, Dict[str, int]] = {}
        if loaded_state:
            updated_pops = apply_offline_gains(self.state, self.world, self.map.factions)

        # 7) Register factions & restore per-faction data
        #    (Note: restored factions already exist in self.map.factions)
        self._initialize_and_restore_factions(updated_pops, loaded_state)

        # 8) Set turn counter & refresh player population
        self.turn = self.state.turn if loaded_state else 0
        if self.player_faction:
            self.population = self.player_faction.citizens.count

        if loaded_state:
            self.event_turn_counters = dict(self.state.event_turn_counters)

        # 9) Logging
        logger.info("Game started with factions:")
        for faction in self.map.factions:
            pos = faction.settlement.position
            logger.info(" - %s at (%d, %d)", faction.name, pos.x, pos.y)

        logger.debug("Initial Resources: %s", self.state.resources)
        logger.debug("Initial Population: %d", self.state.population)

        # 10) Demonstration event & leader update
        self.simulate_events()
        self.update_leaders()

    def _load_and_deserialize_world(self) -> bool:
        """
        Try to load GameState from disk. If successful:
          • Replace self.state
          • Reconstruct `self.world` (including stored road segments)
        Returns True if a valid saved state was loaded, False otherwise.
        """
        try:
            loaded_state, _ = load_state()
        except Exception as e:
            logger.warning("Failed to load saved state: %s. Starting fresh.", e)
            return False

        if not loaded_state:
            return False

        self.state = loaded_state

        # If stored world data exists, rebuild World object
        if self.state.world:
            try:
                settings_obj = WorldSettings(**self.state.world.get("settings", {}))
                self.world = World(
                    width=settings_obj.width,
                    height=settings_obj.height,
                    settings=settings_obj,
                )
                deserialize_world(self.state.world, self.world)
            except (KeyError, TypeError, Exception) as e:
                logger.warning("Failed to deserialize world: %s. Using new World.", e)

            # Restore roads (list of [x1, y1, x2, y2])
            if getattr(self.state, "roads", None):
                self.world.roads = [
                    Road(tuple(r[:2]), tuple(r[2:]))
                    for r in self.state.roads
                    if isinstance(r, list) and len(r) == 4
                ]

        return True

    def _restore_factions_from_state(self) -> None:
        """
        Rebuild every Faction object from `self.state.factions`. For each
        saved faction, reinstantiate name, settlement, citizens, workers,
        units, tech_level, god_powers, then append to `self.map.factions`
        and register with ResourceManager & FactionManager.
        """
        saved_factions: SavedFactionsData = self.state.factions

        for fname, fdata in saved_factions.items():
            # 1) Rebuild settlement
            sdata = fdata.get("settlement")
            if not sdata:
                existing = next((f for f in self.map.factions if f.name == fname), None)
                if not existing and self.player_faction and self.player_faction.name == fname:
                    existing = self.player_faction
                if existing:
                    sdata = {
                        "name": existing.settlement.name,
                        "position": {"x": existing.settlement.position.x, "y": existing.settlement.position.y},
                    }
                else:
                    logger.warning("Faction %r missing settlement data; skipping.", fname)
                    continue

            try:
                sx = int(sdata["position"]["x"])
                sy = int(sdata["position"]["y"])
            except (KeyError, TypeError):
                logger.warning(
                    "Invalid settlement position for faction %r: %r. Skipping.",
                    fname, sdata.get("position")
                )
                continue

            if not (0 <= sx < self.world.width and 0 <= sy < self.world.height):
                logger.warning(
                    "Saved settlement for faction %r out of bounds: (%d, %d). Clamping.",
                    fname, sx, sy
                )
                sx = max(0, min(self.world.width - 1, sx))
                sy = max(0, min(self.world.height - 1, sy))

            restored_settlement = Settlement(name=sdata.get("name", fname), position=Position(sx, sy))
            restored_faction = Faction(name=fname, settlement=restored_settlement, world=self.world)

            # 2) Citizen & worker counts
            restored_faction.citizens.count = int(fdata.get("citizens", 0))
            restored_faction.workers.assigned = int(fdata.get("workers", 0))

            # 3) Units & tech_level & god_powers
            restored_faction.units = int(fdata.get("units", 0))
            restored_faction.tech_level = int(fdata.get("tech_level", 0))
            restored_faction.god_powers = fdata.get("god_powers", {})

            # 4) Append to map & register
            self.map.factions.append(restored_faction)
            self._register_faction(restored_faction)
            logger.info("Restored faction %r at (%d, %d)", fname, sx, sy)

        # 5) If a “player” entry exists in fdata, set self.player_faction to it
        #    Otherwise, if we only have one faction named “Player”, that’s the default.
        if "Player" in self.state.factions:
            for f in self.map.factions:
                if f.name == "Player":
                    self.player_faction = f
                    break
        elif self.map.factions:
            # If no explicitly named “Player” saved, pick the first faction as player
            self.player_faction = self.map.factions[0]
            logger.warning("No saved faction named 'Player'; using %r as player.", self.player_faction.name)

    def _initialize_and_restore_factions(
        self, updated_pops: Dict[str, Dict[str, int]], has_loaded_state: bool
    ) -> None:
        """
        For each Faction currently in `self.map.factions`:
          1) Register it if not already done (_register_faction).
          2) If loading from saved, restore resources, buildings, projects, diplomacy states.
        """
        for faction in self.map.factions:
            self._register_faction(faction)

            if has_loaded_state:
                # 1) Restore saved resources
                saved_res: ResourceDict = self.state.resources.get(faction.name, {})
                for rtype, amount in saved_res.items():
                    faction.resources[rtype] = amount

                # 2) Merge offline gains if any
                if faction.name in updated_pops:
                    self.state.factions.setdefault(faction.name, {}).update(updated_pops[faction.name])

                # 3) Restore detailed fields
                fdata: SavedFactionsData = self.state.factions.get(faction.name, {})
                faction.citizens.count = int(fdata.get("citizens", faction.citizens.count))
                faction.workers.assigned = int(fdata.get("workers", faction.workers.assigned))
                faction.units = int(fdata.get("units", getattr(faction, "units", 0)))
                faction.tech_level = int(fdata.get("tech_level", getattr(faction, "tech_level", 0)))
                faction.god_powers = fdata.get("god_powers", getattr(faction, "god_powers", {}))

                # 4) Recreate each saved building & project
                self._restore_buildings(faction, fdata.get("buildings", []))
                self._restore_projects(faction, fdata.get("projects", []))

        # 5) If diplomacy (wars/truces/alliances) were serialized, restore those too:
        #    (This requires that state includes serialized lists of wars, truces, alliances.)
        if hasattr(self.state, "wars"):
            self.wars = []
            for wpair in getattr(self.state, "wars", []):
                # wpair is [ "FactionA_name", "FactionB_name" ]
                fa_name, fb_name = wpair
                fa = next((f for f in self.map.factions if f.name == fa_name), None)
                fb = next((f for f in self.map.factions if f.name == fb_name), None)
                if fa and fb:
                    self.wars.append(DeclarationOfWar((fa, fb)))
                else:
                    logger.warning("Cannot restore war between %r and %r", fa_name, fb_name)
        if hasattr(self.state, "truces"):
            self.truces = []
            for tpair in getattr(self.state, "truces", []):
                # tpair is [ "FactionA_name", "FactionB_name", duration ]
                fa_name, fb_name, dur = tpair
                fa = next((f for f in self.map.factions if f.name == fa_name), None)
                fb = next((f for f in self.map.factions if f.name == fb_name), None)
                if fa and fb:
                    self.truces.append(Truce((fa, fb), int(dur)))
                else:
                    logger.warning("Cannot restore truce between %r and %r", fa_name, fb_name)
        if hasattr(self.state, "alliances"):
            self.alliances = []
            for apair in getattr(self.state, "alliances", []):
                # apair is [ "FactionA_name", "FactionB_name" ]
                fa_name, fb_name = apair
                fa = next((f for f in self.map.factions if f.name == fa_name), None)
                fb = next((f for f in self.map.factions if f.name == fb_name), None)
                if fa and fb:
                    self.alliances.append(Alliance((fa, fb)))
                else:
                    logger.warning("Cannot restore alliance between %r and %r", fa_name, fb_name)

    def simulate_events(self) -> None:
        """
        Run a demonstration attack/disaster to show how defensive buildings
        mitigate population loss & building damage (for logging purposes).
        """
        if not self.player_faction:
            return

        base_pop_loss = 100
        base_damage = 50
        mitigated_pop_loss = mitigate_population_loss(self.player_buildings, base_pop_loss)
        mitigated_damage = mitigate_building_damage(self.player_buildings, base_damage)

        logger.info(
            "Simulated event: Population loss mitigated from %d → %d",
            base_pop_loss,
            mitigated_pop_loss,
        )
        logger.info(
            "Simulated event: Building damage mitigated from %d → %d",
            base_damage,
            mitigated_damage,
        )

    def build_for_player(self, building: Building) -> None:
        """
        Construct a building in the player's settlement. Raises if player not set.
        """
        if not self.player_faction:
            raise RuntimeError("Cannot build: player faction not initialized")
        self.player_faction.build_structure(building)

    def upgrade_player_building(self, building: Building) -> None:
        """
        Upgrade an existing building in the player's settlement. Raises if player not set.
        """
        if not self.player_faction:
            raise RuntimeError("Cannot upgrade: player faction not initialized")
        self.player_faction.upgrade_structure(building)

    def tick(self) -> None:
        """
        Advance one “tick” of the simulation. Steps:
          1) Population growth via FactionManager.tick()
          2) Base food generation from population
          3) Building-driven resource bonuses & processing
          4) Research progress for each faction
          5) Diplomacy (apply active trades, advance truces)
          6) AI relationship evaluation (ai.evaluate_relations)
          7) Update total population & ResourceManager.tick()
          8) Perform end-of-turn cleanup (decrement cooldowns, event counters, etc.)
          9) Debug-log player’s resources & population
        """
        # 1) Advance population counts
        self.faction_manager.tick()

        for faction in self.map.factions:
            # 2a) Each 2 citizens → 1 FOOD (scaled)
            food_gain = int((faction.citizens.count // 2) * settings.SCALE_FACTOR)
            faction.resources[ResourceType.FOOD] = (
                faction.resources.get(ResourceType.FOOD, 0) + food_gain
            )

            # 2b) Building outputs: either process (if ProcessingBuilding) or add resource_bonus
            for building in faction.buildings:
                if isinstance(building, ProcessingBuilding):
                    building.process(faction)
                elif building.resource_type is not None:
                    current = faction.resources.get(building.resource_type, 0)
                    faction.resources[building.resource_type] = (
                        current + building.resource_bonus
                    )

            # 2c) Advance research for this faction
            faction.progress_research()

        # 3) Diplomacy: perform active trades, reduce durations
        self._apply_trade_deals()
        self._advance_truces()

        # 4) AI relationship updates
        ai.evaluate_relations(self)

        # 5) Refresh total population and tick ResourceManager
        self.population = sum(f.citizens.count for f in self.map.factions)
        self.resources.tick(self.map.factions)

        # 6) End-of-turn cleanup: decrement GodPower cooldowns, event counters, etc.
        self._end_of_turn_cleanup()

        # 7) Debug-log current player faction resources & pop
        if self.player_faction:
            res_snapshot = self.player_faction.resources
            pop_count = self.player_faction.citizens.count
            logger.debug(
                "Tick update — Player Resources: %s | Population: %d",
                res_snapshot,
                pop_count,
            )

    def _end_of_turn_cleanup(self) -> None:
        """
        Handle any “end of turn” housekeeping:
          • Decrement GodPower cooldowns (if > 0)
          • Decrement any event_turn_counters if used for event scheduling
        """
        for name in list(self.power_cooldowns.keys()):
            if self.power_cooldowns[name] > 0:
                self.power_cooldowns[name] -= 1

        for evt, counter in list(self.event_turn_counters.items()):
            if counter > 0:
                self.event_turn_counters[evt] = counter - 1

    def save(self) -> None:
        """
        Persist **all** aspects of current game state to disk. This populates:
          • state.resources, state.population, state.claimed_projects
          • state.world (via serialize_world) & state.roads
          • state.factions (via serialize_factions)
          • state.turn, state.event_turn_counters, state.tech_levels,
            state.god_powers, state.wars, state.truces, state.alliances, state.cooldowns
        Then calls `save_state(self.state)` (catching any exception).
        """
        # 1) Recompute actual total population
        self.population = sum(f.citizens.count for f in self.map.factions)

        # 2) Snapshot basic state fields
        self.state.resources = self.resources.data
        self.state.population = self.population
        self.state.claimed_projects = list(self.claimed_projects)
        self.state.world = serialize_world(self.world)
        self.state.factions = serialize_factions(self.map.factions)
        self.state.turn = self.turn

        # 3) Roads → list of [x1, y1, x2, y2]
        self.state.roads = [
            list(r.start + r.end) for r in getattr(self.world, "roads", [])
        ]

        # 4) Event counters, tech levels, god powers, cooldowns
        self.state.event_turn_counters = dict(self.event_turn_counters)
        self.state.tech_levels = {
            f.name: (f.tech_level.value if hasattr(f.tech_level, "value") else int(f.tech_level))
            for f in self.map.factions
        }
        self.state.god_powers = {f.name: getattr(f, "god_powers", {}) for f in self.map.factions}
        self.state.cooldowns = self.power_cooldowns

        # 5) Serialize wars, truces, alliances as lists of lists
        self.state.wars = [
            [war.factions[0].name, war.factions[1].name] for war in self.wars
        ]
        self.state.truces = [
            [truce.factions[0].name, truce.factions[1].name, truce.duration] for truce in self.truces
        ]
        self.state.alliances = [
            [alliance.factions[0].name, alliance.factions[1].name] for alliance in self.alliances
        ]

        # 6) Finally, call save_state (with error-logging)
        try:
            save_state(self.state)
        except Exception as e:
            logger.error("Failed to save game state: %s", e)

    def advance_turn(self) -> None:
        """
        Progress each faction’s GreatProjects by one turn, then update
        “largest army” / “longest road” leaders. (Cooldowns already handled
        in `tick()`.)
        """
        self.turn += 1
        for faction in self.map.factions:
            faction.progress_projects()
        self.update_leaders()

    def calculate_scores(self) -> Dict[str, int]:
        """
        Return a dict: { faction_name → total victory points }, including
        +2 bonus for largest army and +2 bonus for longest road (if applicable).
        Assumes each Faction implements `get_victory_points(self_game)`.
        """
        scores: Dict[str, int] = {}
        for faction in self.map.factions:
            vp = faction.get_victory_points(self)
            if faction.name == self.leaders.get("largest_army"):
                vp += 2
            if faction.name == self.leaders.get("longest_road"):
                vp += 2
            scores[faction.name] = vp
        return scores

    # ----------------------------------------------------------------
    # Diplomacy Helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _match_pair(pair: Tuple[Faction, Faction], a: Faction, b: Faction) -> bool:
        return (pair[0] is a and pair[1] is b) or (pair[0] is b and pair[1] is a)

    def form_trade_deal(
        self,
        faction_a: Faction,
        faction_b: Faction,
        resources_a_to_b: Optional[ResourceDict] = None,
        resources_b_to_a: Optional[ResourceDict] = None,
        duration: int = 0,
    ) -> TradeDeal:
        """
        Build a new TradeDeal (a ↔ b), append it to self.trade_deals,
        and return it. Each side’s resource dict defaults to {} if None.
        """
        deal = TradeDeal(
            faction_a=faction_a,
            faction_b=faction_b,
            resources_a_to_b=resources_a_to_b or {},
            resources_b_to_a=resources_b_to_a or {},
            duration=duration,
        )
        self.trade_deals.append(deal)
        return deal

    def declare_war(self, faction_a: Faction, faction_b: Faction) -> None:
        """
        If no existing war, append a new DeclarationOfWar for (a, b).
        If they’re already at war, log at DEBUG and do nothing.
        """
        if not self.is_at_war(faction_a, faction_b):
            self.wars.append(DeclarationOfWar((faction_a, faction_b)))
        else:
            logger.debug("%s and %s are already at war", faction_a.name, faction_b.name)

    def form_truce(self, faction_a: Faction, faction_b: Faction, duration: int) -> None:
        """
        Remove any existing war between (a, b), then add a Truce with the given duration.
        """
        self.wars = [
            w for w in self.wars if not self._match_pair(w.factions, faction_a, faction_b)
        ]
        self.truces.append(Truce((faction_a, faction_b), duration))

    def break_truce(self, faction_a: Faction, faction_b: Faction) -> None:
        """
        Permanently remove any Truce between (a, b).
        """
        self.truces = [
            t for t in self.truces if not self._match_pair(t.factions, faction_a, faction_b)
        ]

    def form_alliance(self, faction_a: Faction, faction_b: Faction) -> Alliance:
        """
        Cancel any existing wars/truces between (a, b), then record a new Alliance.
        Returns the created Alliance object.
        """
        # Cancel wars
        self.wars = [
            w for w in self.wars if not self._match_pair(w.factions, faction_a, faction_b)
        ]
        # Remove any existing truce
        self.break_truce(faction_a, faction_b)

        alliance = Alliance((faction_a, faction_b))
        self.alliances.append(alliance)
        return alliance

    def break_alliance(self, faction_a: Faction, faction_b: Faction) -> None:
        """
        End any existing Alliance between (a, b).
        """
        self.alliances = [
            a for a in self.alliances if not self._match_pair(a.factions, faction_a, faction_b)
        ]

    def is_allied(self, faction_a: Faction, faction_b: Faction) -> bool:
        """
        Return True if (a, b) share any active Alliance.
        """
        return any(
            self._match_pair(alliance.factions, faction_a, faction_b)
            for alliance in self.alliances
        )

    def is_under_truce(self, faction_a: Faction, faction_b: Faction) -> bool:
        """
        Return True if (a, b) share any active Truce.
        """
        return any(
            self._match_pair(truce.factions, faction_a, faction_b)
            for truce in self.truces
        )

    def is_at_war(self, faction_a: Faction, faction_b: Faction) -> bool:
        """
        Return True if (a, b) share any active DeclarationOfWar.
        """
        return any(
            self._match_pair(war.factions, faction_a, faction_b)
            for war in self.wars
        )

    def _apply_trade_deals(self) -> None:
        """
        For each active TradeDeal in self.trade_deals:
          • Execute resources transfer.
          • Decrement its duration if > 0; remove if duration hits 0.
        """
        for deal in list(self.trade_deals):
            self._execute_trade(deal)
            if deal.duration > 0:
                deal.duration -= 1
                if deal.duration <= 0:
                    self.trade_deals.remove(deal)

    def _advance_truces(self) -> None:
        """
        For each active Truce in self.truces:
          • Decrement its duration; if it hits 0, remove it.
        """
        for truce in list(self.truces):
            truce.duration -= 1
            if truce.duration <= 0:
                self.truces.remove(truce)

    def _longest_road_for(self, faction: Faction) -> int:
        """
        Compute the length of the longest continuous road belonging to `faction`,
        starting from their settlement’s coordinates. Uses DFS on self.world.roads.
        Returns 0 if no roads exist.
        """
        roads_list = getattr(self.world, "roads", [])
        if not roads_list:
            return 0

        start = (faction.settlement.position.x, faction.settlement.position.y)
        adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for r in roads_list:
            adjacency.setdefault(r.start, []).append(r.end)
            adjacency.setdefault(r.end, []).append(r.start)

        def dfs(
            node: Tuple[int, int],
            visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]]
        ) -> int:
            best_length = 0
            for neighbor in adjacency.get(node, []):
                edge = tuple(sorted((node, neighbor)))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                length = 1 + dfs(neighbor, visited_edges)
                best_length = max(best_length, length)
                visited_edges.remove(edge)
            return best_length

        return dfs(start, set())

    def update_leaders(self) -> None:
        """
        Recompute which faction has:
          • The “largest army” (highest `units`)
          • The “longest road” (via `_longest_road_for`)
        Stores their names in self.leaders["largest_army"] / ["longest_road"].
        """
        # 1) Largest army
        armies: Dict[str, int] = {f.name: getattr(f, "units", 0) for f in self.map.factions}
        if armies:
            max_units = max(armies.values())
            self.leaders["largest_army"] = next(
                (name for name, u in armies.items() if u == max_units and max_units > 0),
                None,
            )

        # 2) Longest road
        roads: Dict[str, int] = {f.name: self._longest_road_for(f) for f in self.map.factions}
        if roads:
            max_length = max(roads.values())
            self.leaders["longest_road"] = next(
                (name for name, length in roads.items() if length == max_length and max_length > 0),
                None,
            )

    @staticmethod
    def _execute_trade(deal: TradeDeal) -> None:
        """
        Carry out resource transfers for a given TradeDeal:
          • Take each resource in deal.resources_a_to_b from A → B
          • Take each resource in deal.resources_b_to_a from B → A
        """
        if deal.resources_a_to_b:
            deal.faction_a.transfer_resources(deal.faction_b, deal.resources_a_to_b)
        if deal.resources_b_to_a:
            deal.faction_b.transfer_resources(deal.faction_a, deal.resources_b_to_a)

    # ----------------------------------------------------------------
    # Building & Project Restoration Helpers (invoked during load)
    # ----------------------------------------------------------------
    def _restore_buildings(self, faction: Faction, data: List[Dict[str, Any]]) -> None:
        """
        Given serialized building data for a Faction, clear their current
        .buildings list and reinstantiate each building class by CLASS_ID,
        setting its stored `level`. Skips unknown CLASS_IDs with a warning.
        """
        cls_map: Dict[str, type[Building]] = {cls.CLASS_ID: cls for cls in ALL_BUILDING_CLASSES}

        faction.buildings.clear()
        for bdict in data:
            cls_id = bdict.get("id")
            cls_type = cls_map.get(cls_id)
            if not cls_type:
                logger.warning(
                    "Unknown building CLASS_ID %r for faction %r – skipping",
                    cls_id,
                    faction.name,
                )
                continue
            inst = cls_type()
            inst.level = int(bdict.get("level", 1))
            faction.buildings.append(inst)

    def _restore_projects(self, faction: Faction, data: List[Dict[str, Any]]) -> None:
        """
        Given serialized project data for a Faction, clear their .projects list
        and reinstantiate each GreatProject by copying from GREAT_PROJECT_TEMPLATES,
        then set its `progress`. Skips unknown project names with a warning.
        """
        from copy import deepcopy

        faction.projects.clear()
        for pdict in data:
            template = GREAT_PROJECT_TEMPLATES.get(pdict.get("name"))
            if not template:
                logger.warning(
                    "Unknown project %r for faction %r – skipping",
                    pdict.get("name"),
                    faction.name,
                )
                continue
            proj_copy = deepcopy(template)
            proj_copy.progress = int(pdict.get("progress", 0))
            faction.projects.append(proj_copy)

    # ----------------------------------------------------------------
    # God Power Utilities
    # ----------------------------------------------------------------
    def available_powers(self) -> List[GodPower]:
        """
        Return all GodPower objects that:
          1) Are unlocked (via completed_projects)
          2) Have cooldown ≤ 0 (ready to cast)
        If no player faction, returns an empty list.
        """
        if not self.player_faction:
            return []

        # Collect names of all completed GreatProjects
        completed_set = {p.name for p in self.player_faction.completed_projects()}

        return [
            pwr
            for pwr in self.god_powers.values()
            if pwr.is_unlocked(self.player_faction, completed_set)
            and self.power_cooldowns.get(pwr.name, 0) <= 0
        ]

    def use_power(self, name: str) -> None:
        """
        Attempt to cast the named GodPower.
          • Raises ValueError if name not recognized or power still on cooldown.
          • If successful, calls `power.apply(self)` and resets its cooldown.
        """
        if name not in self.god_powers:
            raise ValueError(f"Unknown power: {name}")
        if self.power_cooldowns.get(name, 0) > 0:
            raise ValueError(f"{name} is on cooldown (remaining: {self.power_cooldowns[name]})")

        power = self.god_powers[name]
        power.apply(self)
        self.power_cooldowns[name] = power.cooldown

# --------------------------------------------------------------------
# Module-Level “main()” Function with CLI Support
# --------------------------------------------------------------------
def main() -> int:
    """
    Entry point for running this game script.
    Supports options:
      --player-x, --player-y : initial player settlement coordinates
      --load-file            : path to an existing save file
      --no-save              : skip saving at end (for dry-runs/tests)
    Returns exit code 0 on success, nonzero on failure.
    """
    parser = argparse.ArgumentParser(description="Launch the hex-based strategy game.")
    parser.add_argument(
        "--player-x", type=int, default=0,
        help="X coordinate for the player's starting settlement (default: 0)"
    )
    parser.add_argument(
        "--player-y", type=int, default=0,
        help="Y coordinate for the player's starting settlement (default: 0)"
    )
    parser.add_argument(
        "--load-file", type=str, default="",
        help="Path to an existing saved game state (skip fresh start if provided)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run the game without writing back to disk (dry-run mode)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Set up logging as early as possible
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1) If a load file is provided and valid, load state; else create fresh
    initial_state: Optional[GameState] = None
    if args.load_file:
        try:
            loaded_state, _ = load_state()
            if loaded_state:
                initial_state = loaded_state
                logger.info("Loaded saved game from %r", args.load_file)
            else:
                logger.warning("No saved state found in %r; starting fresh.", args.load_file)
        except Exception as e:
            logger.error("Error loading save file %r: %s. Starting fresh.", args.load_file, e)

    # 2) Instantiate Game (with or without a loaded state)
    game = Game(state=initial_state)

    # 3) If no loaded state or if the loaded state lacked a player faction, place a new one
    if not initial_state or not game.player_faction:
        try:
            game.place_initial_settlement(args.player_x, args.player_y, name="Player")
        except ValueError as e:
            logger.error("Failed to place player settlement at (%d, %d): %s", args.player_x, args.player_y, e)
            return 1

    # 4) Kick off the simulation (or restoration)
    try:
        game.begin()
    except Exception as e:
        logger.error("Error during game.begin(): %s", e)
        return 2

    # 5) (Optional) Run a single tick for demonstration
    #game.tick()  # Uncomment if you want to step one tick

    # 6) Save unless --no-save was specified
    if not args.no_save:
        try:
            game.save()
        except Exception as e:
            logger.error("Error while saving game: %s", e)
            return 3

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
