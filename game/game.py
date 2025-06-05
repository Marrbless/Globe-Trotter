import random
import time
import logging
from typing import List, Dict, Any

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
from world.world import World, ResourceType, WorldSettings
from .resources import ResourceManager
from .models import Position, Settlement, GreatProject, Faction
from .god_powers import ALL_POWERS, GodPower

logger = logging.getLogger("mygame.Game")
logger.addHandler(logging.NullHandler())


# Predefined templates for special high-cost projects
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
}

# Mapping of project names to new actions unlocked upon completion
PROJECT_UNLOCKS = {
    "Grand Cathedral": "celebrate_festival",
    "Sky Fortress": "air_strike",
}


def apply_project_bonus(faction: Faction, project: GreatProject) -> None:
    """
    Grant faction bonuses when a project is finished.
    Appends the action key from PROJECT_UNLOCKS if not already present.
    """
    action = PROJECT_UNLOCKS.get(project.name)
    if action and action not in faction.unlocked_actions:
        faction.unlocked_actions.append(action)


class Map:
    """
    Manages placement of factions on a hex map and computes distances.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.factions: List[Faction] = []

    def is_occupied(self, position: Position) -> bool:
        for faction in self.factions:
            if faction.settlement.position == position:
                return True
        return False

    def distance(self, pos1: Position, pos2: Position) -> int:
        """
        Return axial hex distance between two positions.
        (dx + dy + |dx+dy|)/2 formula for cube coords derived distance.
        """
        dq = pos1.x - pos2.x
        dr = pos1.y - pos2.y
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    def add_faction(self, faction: Faction):
        if not self.is_occupied(faction.settlement.position):
            self.factions.append(faction)
        else:
            raise ValueError("Position is already occupied")

    def spawn_ai_factions(self, player_settlement: Settlement) -> List[Faction]:
        """
        Attempt to spawn `settings.AI_FACTION_COUNT` AI factions
        at random positions that are at least MIN_DISTANCE_FROM_PLAYER
        hexes away from `player_settlement`.
        """
        count = settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
        new_factions: List[Faction] = []
        while spawned < count and attempts < count * 10:
            attempts += 1
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if (
                not self.is_occupied(pos)
                and self.distance(pos, player_settlement.position)
                >= settings.MIN_DISTANCE_FROM_PLAYER
            ):
                ai = Faction(
                    name=f"AI #{spawned + 1}",
                    settlement=Settlement(name=f"AI Town {spawned + 1}", position=pos),
                    citizens=Citizen(count=random.randint(8, 15)),
                )
                self.add_faction(ai)
                new_factions.append(ai)
                spawned += 1
        return new_factions


class Game:
    """
    Core game class handling world, factions, resources, buildings, diplomacy, and turns.
    """
    def __init__(self, state: GameState | None = None, world: World | None = None):
        # Initialize map and world
        self.map = Map(*settings.MAP_SIZE)
        self.world = world or World(*settings.MAP_SIZE)

        # Prepare state container; actual saved data will be loaded in begin()
        self.state = state or GameState(
            timestamp=time.time(), resources={}, population=0, claimed_projects=[]
        )

        # Initialize managers for resources and populations
        self.resources = ResourceManager(self.world, self.state.resources)
        self.faction_manager = FactionManager()

        self.population = self.state.population
        self.claimed_projects: set[str] = set(self.state.claimed_projects)
        self.player_faction: Faction | None = None
        self.player_buildings: List[Building] = []
        self.turn = 0

        # Diplomacy & conflict tracking
        self.trade_deals: List[TradeDeal] = []
        self.truces: List[Truce] = []
        self.wars: List[DeclarationOfWar] = []

        # Leaders for largest army & longest road
        self.leaders: Dict[str, str | None] = {"largest_army": None, "longest_road": None}

        # God powers & alliances
        self.god_powers: Dict[str, GodPower] = {p.name: p for p in ALL_POWERS}
        self.alliances: List[Alliance] = []

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        """
        Place the player's settlement at (x, y). Raises ValueError if
        that position is already occupied.
        """
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)
        # Register resources and population management for the new faction
        self.resources.register(self.player_faction)
        self.faction_manager.add_faction(self.player_faction)
        self.population = self.player_faction.citizens.count

    def add_building(self, building: Building):
        """Add a defensive building to the player's settlement (for mitigate calculations)."""
        self.player_buildings.append(building)

    def begin(self):
        """
        Initialize or restore a saved game state. Steps:
          1. Ensure player settlement is placed.
          2. Spawn AI factions (no registration yet).
          3. Attempt to load GameState from disk (with error handling).
          4. Apply offline gains if a valid state was loaded.
          5. Register all factions exactly once (ResourceManager & FactionManager).
          6. Restore saved resources, populations, buildings, and projects.
          7. Initialize turn, leaders, and log starting information.
        """
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")

        # 1. Spawn AI factions on empty map
        ai_factions = self.map.spawn_ai_factions(self.player_faction.settlement)

        # 2. Attempt to load saved state
        loaded_state: GameState | None = None
        try:
            loaded_state, _ = load_state()
        except (IOError, ValueError) as e:
            logger.warning("Failed to load saved state: %s. Starting new game.", e)

        # 3. If loaded_state exists & has world data, attempt to reconstruct
        if loaded_state and loaded_state.world:
            try:
                settings_obj = WorldSettings(**loaded_state.world.get("settings", {}))
                self.world = World(
                    width=settings_obj.width,
                    height=settings_obj.height,
                    settings=settings_obj,
                )
                deserialize_world(loaded_state.world, self.world)
            except (KeyError, TypeError) as e:
                logger.warning("Failed to deserialize world: %s. Using fresh world.", e)
                loaded_state = None

        # 4. Apply offline gains or default to empty
        if loaded_state:
            self.state = loaded_state
            updated_pops = apply_offline_gains(self.state, self.world, self.map.factions)
        else:
            updated_pops = {}

        # 5. Single pass: register factions and restore state
        for faction in self.map.factions:
            # Register with ResourceManager and FactionManager (once)
            try:
                self.resources.register(faction)
            except ValueError:
                logger.debug("Faction %s already registered with ResourceManager", faction.name)
            try:
                self.faction_manager.add_faction(faction)
            except ValueError:
                logger.debug("Faction %s already registered with FactionManager", faction.name)

            # 6a. Restore saved resources (if present)
            if loaded_state:
                saved_res = self.state.resources.get(faction.name, {})
                faction.resources.update(saved_res)

            # 6b. Restore population & units if offline_gains or saved data exist
            if faction.name in updated_pops:
                self.state.factions.setdefault(faction.name, {}).update(updated_pops[faction.name])

            if loaded_state:
                fdata = self.state.factions.get(faction.name, {})
                faction.citizens.count = fdata.get("citizens", faction.citizens.count)
                faction.workers.assigned = fdata.get("workers", faction.workers.assigned)
                faction.units = fdata.get("units", getattr(faction, "units", 0))
                self._restore_buildings(faction, fdata.get("buildings", []))
                self._restore_projects(faction, fdata.get("projects", []))

        self.turn = (self.state.turn if loaded_state else 0)

        if self.player_faction:
            self.population = self.player_faction.citizens.count

        # 7. Final initialization logging
        logger.info("Game started with factions:")
        for faction in self.map.factions:
            pos = faction.settlement.position
            logger.info("- %s at (%d, %d)", faction.name, pos.x, pos.y)

        logger.debug("Initial Resources: %s", self.state.resources)
        logger.debug("Initial Population: %d", self.state.population)

        # Simulate a sample event for demonstration
        self.simulate_events()
        self.update_leaders()

    def simulate_events(self):
        """
        Run a sample attack/disaster to demonstrate how defensive buildings mitigate
        population loss and building damage.
        """
        if not self.player_faction:
            return

        base_pop_loss = 100
        base_damage = 50
        pop_loss = mitigate_population_loss(self.player_buildings, base_pop_loss)
        damage = mitigate_building_damage(self.player_buildings, base_damage)
        logger.info(
            "Population loss mitigated from %d to %d", base_pop_loss, pop_loss
        )
        logger.info(
            "Building damage mitigated from %d to %d", base_damage, damage
        )

    def build_for_player(self, building: Building) -> None:
        """
        Construct a building in the player's settlement. Raises RuntimeError if
        the player faction is not yet initialized.
        """
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.build_structure(building)

    def upgrade_player_building(self, building: Building) -> None:
        """
        Upgrade an existing building in the player's settlement. Raises RuntimeError
        if the player faction is not yet initialized.
        """
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.upgrade_structure(building)

    def tick(self) -> None:
        """
        Advance the game state by one tick. This includes:
          1. Population growth via `FactionManager`
          2. Basic resource generation (food from population)
          3. Building-based resource bonuses
          4. Progress research & diplomacy & AI relations
        """
        # 1. Update population for all factions
        self.faction_manager.tick()

        for faction in self.map.factions:
            # 2a. Generate base food from population
            food_gain = int((faction.citizens.count // 2) * settings.SCALE_FACTOR)
            faction.resources[ResourceType.FOOD] = (
                faction.resources.get(ResourceType.FOOD, 0) + food_gain
            )

            # 2b. Building effects (resource bonuses & processing)
            for building in faction.buildings:
                if isinstance(building, ProcessingBuilding):
                    building.process(faction)
                    continue
                if building.resource_type is not None:
                    current = faction.resources.get(building.resource_type, 0)
                    faction.resources[building.resource_type] = (
                        current + building.resource_bonus
                    )

            # 2c. Progress research for each faction
            faction.progress_research()

        # 3. Diplomacy: apply active trade deals, decrease truce durations
        self._apply_trade_deals()
        self._advance_truces()

        # 4. Evaluate AI relationships this turn
        ai.evaluate_relations(self)

        # 5. Update overall population & ResourceManager data
        self.population = sum(f.citizens.count for f in self.map.factions)
        self.resources.tick(self.map.factions)

        # 6. Debug logging for player faction
        if self.player_faction:
            res = self.player_faction.resources
            pop = self.player_faction.citizens.count
            logger.debug("Tick update — Resources: %s | Population: %d", res, pop)

    def save(self) -> None:
        """
        Persist the current game state to disk. Updates:
          - state.resources
          - state.population
          - state.claimed_projects
          - state.world
          - state.factions
          - state.turn
        Then calls `save_state(self.state)`.
        """
        # Ensure population is accurate
        self.population = sum(f.citizens.count for f in self.map.factions)
        self.state.resources = self.resources.data
        self.state.population = self.population
        self.state.claimed_projects = list(self.claimed_projects)
        self.state.world = serialize_world(self.world)
        self.state.factions = serialize_factions(self.map.factions)
        self.state.turn = self.turn

        try:
            save_state(self.state)
        except IOError as e:
            logger.error("Failed to save game state: %s", e)

    def advance_turn(self) -> None:
        """
        Progresses construction on all ongoing projects and updates leaders.
        """
        self.turn += 1
        for faction in self.map.factions:
            faction.progress_projects()
        self.update_leaders()

    def calculate_scores(self) -> Dict[str, int]:
        """
        Return a mapping of faction name → victory points, including bonuses for
        largest army and longest road. Assumes `Faction.get_victory_points(game)`
        is implemented consistently for all factions.
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

    # ------------------------------------------------------------------
    # Diplomacy utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _match_pair(pair: tuple[Faction, Faction], a: Faction, b: Faction) -> bool:
        return (pair[0] is a and pair[1] is b) or (pair[0] is b and pair[1] is a)

    def form_trade_deal(
        self,
        faction_a: Faction,
        faction_b: Faction,
        resources_a_to_b: Dict[ResourceType, int] | None = None,
        resources_b_to_a: Dict[ResourceType, int] | None = None,
        duration: int = 0,
    ) -> TradeDeal:
        """
        Create and register a new `TradeDeal` between `faction_a` and `faction_b`.
        Returns the constructed `TradeDeal`.
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
        Initiate a war between `faction_a` and `faction_b`, unless one already exists.
        """
        if not self.is_at_war(faction_a, faction_b):
            self.wars.append(DeclarationOfWar((faction_a, faction_b)))
        else:
            logger.debug("%s and %s are already at war", faction_a.name, faction_b.name)

    def form_truce(self, faction_a: Faction, faction_b: Faction, duration: int) -> None:
        """
        Establish a truce for `duration` turns. Cancels any existing war between them.
        """
        self.wars = [
            w for w in self.wars
            if not self._match_pair(w.factions, faction_a, faction_b)
        ]
        self.truces.append(Truce((faction_a, faction_b), duration))

    def break_truce(self, faction_a: Faction, faction_b: Faction) -> None:
        """
        Break any existing truce between `faction_a` and `faction_b`.
        """
        self.truces = [
            t for t in self.truces
            if not self._match_pair(t.factions, faction_a, faction_b)
        ]

    def form_alliance(self, faction_a: Faction, faction_b: Faction) -> Alliance:
        """
        Form an alliance between `faction_a` and `faction_b`. Cancels wars/truces first.
        Returns the created `Alliance` object.
        """
        self.wars = [
            w for w in self.wars
            if not self._match_pair(w.factions, faction_a, faction_b)
        ]
        self.break_truce(faction_a, faction_b)
        alliance = Alliance((faction_a, faction_b))
        self.alliances.append(alliance)
        return alliance

    def break_alliance(self, faction_a: Faction, faction_b: Faction) -> None:
        """
        Break an existing alliance between `faction_a` and `faction_b`.
        """
        self.alliances = [
            a for a in self.alliances
            if not self._match_pair(a.factions, faction_a, faction_b)
        ]

    def is_allied(self, faction_a: Faction, faction_b: Faction) -> bool:
        return any(
            self._match_pair(alliance.factions, faction_a, faction_b)
            for alliance in self.alliances
        )

    def is_under_truce(self, faction_a: Faction, faction_b: Faction) -> bool:
        return any(
            self._match_pair(truce.factions, faction_a, faction_b)
            for truce in self.truces
        )

    def is_at_war(self, faction_a: Faction, faction_b: Faction) -> bool:
        return any(
            self._match_pair(war.factions, faction_a, faction_b)
            for war in self.wars
        )

    def _apply_trade_deals(self) -> None:
        """
        Execute resource transfers for all active TradeDeals. Decrement duration,
        and remove deals whose duration hits zero.
        """
        for deal in list(self.trade_deals):
            self._execute_trade(deal)
            if deal.duration > 0:
                deal.duration -= 1
                if deal.duration <= 0:
                    self.trade_deals.remove(deal)

    def _advance_truces(self) -> None:
        """
        Decrement the duration on active truces and remove expired ones.
        """
        for truce in list(self.truces):
            truce.duration -= 1
            if truce.duration <= 0:
                self.truces.remove(truce)

    def _longest_road_for(self, faction: Faction) -> int:
        """
        Compute the longest continuous road chain starting from the faction’s
        settlement position via DFS over `self.world.roads` adjacency.
        """
        start = (faction.settlement.position.x, faction.settlement.position.y)
        adjacency: Dict[tuple[int, int], List[tuple[int, int]]] = {}
        for r in self.world.roads:
            adjacency.setdefault(r.start, []).append(r.end)
            adjacency.setdefault(r.end, []).append(r.start)

        def dfs(node: tuple[int, int], visited: set[tuple[tuple[int, int], tuple[int, int]]]) -> int:
            best = 0
            for neigh in adjacency.get(node, []):
                edge = tuple(sorted((node, neigh)))
                if edge in visited:
                    continue
                visited.add(edge)
                best = max(best, 1 + dfs(neigh, visited))
                visited.remove(edge)
            return best

        return dfs(start, set())

    def update_leaders(self) -> None:
        """
        Determine which faction has the largest army and which has the
        longest road, storing their names in `self.leaders`.
        """
        armies = {f.name: f.units for f in self.map.factions}
        if armies:
            max_units = max(armies.values())
            self.leaders["largest_army"] = (
                next((name for name, u in armies.items() if u == max_units and max_units > 0), None)
            )

        roads = {f.name: self._longest_road_for(f) for f in self.map.factions}
        if roads:
            max_len = max(roads.values())
            self.leaders["longest_road"] = (
                next((name for name, l in roads.items() if l == max_len and max_len > 0), None)
            )

    @staticmethod
    def _execute_trade(deal: TradeDeal) -> None:
        """
        Transfer resources specified in a TradeDeal: 
        - from A → B (`resources_a_to_b`) 
        - from B → A (`resources_b_to_a`)
        """
        if deal.resources_a_to_b:
            deal.faction_a.transfer_resources(deal.faction_b, deal.resources_a_to_b)
        if deal.resources_b_to_a:
            deal.faction_b.transfer_resources(deal.faction_a, deal.resources_b_to_a)

    # ------------------------------------------------------------------
    # Building & Project Restoration Helpers
    # ------------------------------------------------------------------

    def _restore_buildings(self, faction: Faction, data: List[Dict[str, Any]]):
        """
        Restore `faction.buildings` from serialized data.
        Each building dictionary must have:
          - "id" (CLASS_ID, not freeform name)
          - "level"
        Any unknown CLASS_ID will be skipped with a warning.
        """
        cls_map = {cls.CLASS_ID: cls for cls in ALL_BUILDING_CLASSES}

        faction.buildings.clear()
        for b in data:
            cls_id = b.get("id")
            cls = cls_map.get(cls_id)
            if not cls:
                logger.warning("Unknown building CLASS_ID %r for faction %r – skipping", cls_id, faction.name)
                continue
            inst = cls()
            inst.level = int(b.get("level", 1))
            faction.buildings.append(inst)

    def _restore_projects(self, faction: Faction, data: List[Dict[str, Any]]):
        """
        Restore `faction.projects` from serialized data.
        Each project dictionary must have:
          - "name"
          - "progress"
        Looks up in GREAT_PROJECT_TEMPLATES and deep-copies the template if found.
        """
        from copy import deepcopy

        faction.projects.clear()
        for p in data:
            template = GREAT_PROJECT_TEMPLATES.get(p.get("name"))
            if not template:
                logger.warning("Unknown project %r for faction %r – skipping", p.get("name"), faction.name)
                continue
            proj = deepcopy(template)
            proj.progress = int(p.get("progress", 0))
            faction.projects.append(proj)

    # ------------------------------------------------------------------
    # God power utilities
    # ------------------------------------------------------------------

    def available_powers(self) -> List[GodPower]:
        """
        Return a list of GodPower objects that the player faction can use,
        based on completed projects. If no player faction, returns [].
        """
        if not self.player_faction:
            return []
        completed = {p.name for p in self.player_faction.completed_projects()}
        return [
            p
            for p in self.god_powers.values()
            if p.is_unlocked(self.player_faction, completed)
        ]

    def use_power(self, name: str) -> None:
        """
        Apply the named GodPower to this game. Raises ValueError if unknown.
        """
        if name not in self.god_powers:
            raise ValueError(f"Unknown power: {name}")
        power = self.god_powers[name]
        power.apply(self)


def main():
    """
    Entry point: create a new Game, place a settlement, begin, and save.
    Logging configuration should be done by the caller (e.g., if __name__ == "__main__").
    """
    # Example logging setup; actual application should configure outside.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    game = Game()
    # Example: place the player's settlement at (0, 0). Change coordinates as needed.
    game.place_initial_settlement(0, 0)
    game.begin()
    game.save()


if __name__ == "__main__":
    main()
