import random
import time
from typing import List, Dict
from dataclasses import dataclass, field

from .persistence import GameState, load_state, save_state
from .buildings import (
    Building,
    ProcessingBuilding,
    mitigate_building_damage,
    mitigate_population_loss,
)
from .population import Citizen, Worker
from . import settings
from world.world import World, ResourceType
from .resources import ResourceManager
from .models import Position, Settlement, GreatProject


# Predefined templates for special high-cost projects
GREAT_PROJECT_TEMPLATES: Dict[str, GreatProject] = {
    "Grand Cathedral": GreatProject(
        name="Grand Cathedral",
        build_time=5,
        victory_points=10,
        bonus="Increases faith across the realm",
    ),
    "Sky Fortress": GreatProject(
        name="Sky Fortress",
        build_time=8,
        victory_points=15,
        bonus="Provides unmatched military power",
    ),
}

# Mapping of project names to new actions unlocked upon completion
PROJECT_UNLOCKS = {
    "Grand Cathedral": "celebrate_festival",
    "Sky Fortress": "air_strike",
}


def apply_project_bonus(faction: "Faction", project: GreatProject) -> None:
    """Grant faction bonuses when a project is finished."""
    action = PROJECT_UNLOCKS.get(project.name)
    if action and action not in faction.unlocked_actions:
        faction.unlocked_actions.append(action)


@dataclass
class Faction:
    name: str
    settlement: Settlement
    citizens: Citizen = field(default_factory=lambda: Citizen(count=10))
    resources: Dict[ResourceType, int] = field(
        default_factory=lambda: {
            ResourceType.FOOD: 100,
            ResourceType.WOOD: 50,
            ResourceType.STONE: 30,
            ResourceType.ORE: 0,
            ResourceType.METAL: 0,
            ResourceType.CLOTH: 0,
        }
    )
    workers: Worker = field(default_factory=lambda: Worker(assigned=10))
    buildings: List[Building] = field(default_factory=list)
    projects: List[GreatProject] = field(default_factory=list)
    unlocked_actions: List[str] = field(default_factory=list)

    @property
    def population(self) -> int:
        """Return total citizens for backward compatibility."""
        return self.citizens.count

    @population.setter
    def population(self, value: int) -> None:
        self.citizens.count = value

    def start_project(self, project: GreatProject) -> None:
        """Begin constructing a great project."""
        self.projects.append(project)

    def progress_projects(self) -> None:
        for proj in self.projects:
            if not proj.is_complete():
                proj.advance()
            if proj.is_complete() and not getattr(proj, "bonus_applied", False):
                apply_project_bonus(self, proj)
                proj.bonus_applied = True

    def completed_projects(self) -> List[GreatProject]:
        return [p for p in self.projects if p.is_complete()]

    def get_victory_points(self) -> int:
        total = sum(b.victory_points for b in self.buildings)
        total += sum(p.victory_points for p in self.completed_projects())
        return total

    def build_structure(self, building: Building) -> None:
        """
        Pay the required resources (assumed to be a dict mapping ResourceType to amounts)
        and add the Building instance to this faction.
        """
        cost: Dict[ResourceType, int] = building.construction_cost
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to build {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        self.buildings.append(building)

    def upgrade_structure(self, building: Building) -> None:
        """
        Pay the upgrade cost and then call the building's internal upgrade() method.
        Assumes building.upgrade_cost() returns a dict keyed by ResourceType.
        """
        cost: Dict[ResourceType, int] = building.upgrade_cost()
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to upgrade {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        building.upgrade()


class Map:
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
        """Return axial hex distance between two positions."""
        dq = pos1.x - pos2.x
        dr = pos1.y - pos2.y
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    def add_faction(self, faction: Faction):
        if not self.is_occupied(faction.settlement.position):
            self.factions.append(faction)
        else:
            raise ValueError("Position is already occupied")

    def spawn_ai_factions(self, player_settlement: Settlement) -> List[Faction]:
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
    def __init__(self, state: GameState | None = None, world: World | None = None):
        # Initialize map and world
        self.map = Map(*settings.MAP_SIZE)
        self.world = world or World(*settings.MAP_SIZE)

        # Prepare state container; actual saved data will be loaded in begin()
        self.state = state or GameState(
            timestamp=time.time(), resources={}, population=0
        )

        # Initialize resource manager with whatever data we have so far
        self.resources = ResourceManager(self.world, self.state.resources)

        self.population = self.state.population
        self.player_faction: Faction | None = None
        self.player_buildings: List[Building] = []
        self.turn = 0

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)
        # Register resources for the new faction
        self.resources.register(self.player_faction)

    def add_building(self, building: Building):
        """Add a defensive building to the player's settlement."""
        self.player_buildings.append(building)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        ai_factions = self.map.spawn_ai_factions(self.player_faction.settlement)
        for faction in self.map.factions:
            self.resources.register(faction)

        # Load saved state now that all factions exist
        self.state = load_state(world=self.world, factions=self.map.factions)
        self.resources = ResourceManager(self.world, self.state.resources)

        for faction in self.map.factions:
            self.resources.register(faction)
            saved = self.state.resources.get(faction.name)
            if saved is not None:
                faction.resources.update(saved)
        if self.player_faction:
            self.player_faction.citizens.count = self.state.population
            self.population = self.state.population

        print("Game started with factions:")
        for faction in self.map.factions:
            pos = faction.settlement.position
            print(f"- {faction.name} at ({pos.x}, {pos.y})")

        # Show initial resource and population state
        print(f"Resources: {self.state.resources}")
        print(f"Population: {self.state.population}")

        # Simulate a sample event demonstrating defensive buildings
        self.simulate_events()

    def simulate_events(self):
        """Run sample attack and disaster to show defensive buildings."""
        if not self.player_faction:
            return
        base_pop_loss = 100
        base_damage = 50
        pop_loss = mitigate_population_loss(self.player_buildings, base_pop_loss)
        damage = mitigate_building_damage(self.player_buildings, base_damage)
        print(f"Population loss mitigated from {base_pop_loss} to {pop_loss}")
        print(f"Building damage mitigated from {base_damage} to {damage}")

    def build_for_player(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.build_structure(building)

    def upgrade_player_building(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.upgrade_structure(building)

    def tick(self) -> None:
        """
        Advance the game state by one tick. This includes:
          1. Population growth
          2. Basic resource generation (food from population)
          3. Building-based resource bonuses
        """
        for faction in self.map.factions:
            # 1. Population growth
            faction.citizens.count += 1

            # 2. Generate base food from population
            food_gain = faction.citizens.count // 2
            faction.resources[ResourceType.FOOD] = (
                faction.resources.get(ResourceType.FOOD, 0) + food_gain
            )

            # 3. Building effects
            for building in faction.buildings:
                if isinstance(building, ProcessingBuilding):
                    building.process(faction)
                    continue
                if building.resource_type is not None:
                    current = faction.resources.get(building.resource_type, 0)
                    faction.resources[building.resource_type] = (
                    current + building.resource_bonus
                )

        # After all factions have been processed, update overall population and
        # ResourceManager data
        self.population = sum(f.citizens.count for f in self.map.factions)
        self.resources.tick(self.map.factions)

        # Debug output for the player faction
        if self.player_faction:
            res = self.player_faction.resources
            pop = self.player_faction.citizens.count
            print(f"Resources: {res} | Population: {pop}")

    def save(self) -> None:
        self.state.resources = self.resources.data
        self.state.population = self.population
        save_state(self.state)

    def advance_turn(self) -> None:
        """Progress construction on all ongoing projects."""
        self.turn += 1
        for faction in self.map.factions:
            faction.progress_projects()

    def calculate_scores(self) -> Dict[str, int]:
        """Return victory points for all factions."""
        return {f.name: f.get_victory_points() for f in self.map.factions}


def main():
    game = Game()
    # Example: player places settlement at (0, 0)
    game.place_initial_settlement(0, 0)
    game.begin()
    game.save()


if __name__ == "__main__":
    main()
