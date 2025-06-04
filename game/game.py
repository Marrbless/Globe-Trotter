import random
from dataclasses import dataclass, field
from typing import List, Dict

from .buildings import Building
from . import settings
from .world import World
from .resources import ResourceManager


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Settlement:
    name: str
    position: Position


@dataclass
class Faction:
    name: str
    settlement: Settlement

    # Population and resource tracking
    population: int = 10
    resources: Dict[str, int] = field(
        default_factory=lambda: {"food": 100, "wood": 50, "stone": 30}
    )
    buildings: List[Building] = field(default_factory=list)
    workers: Dict[str, int] = field(default_factory=lambda: {"food": 10, "wood": 0, "stone": 0})

    def build_structure(self, building: Building) -> None:
        """
        Pay the required resources (assumed to be a dict mapping resource types to amounts)
        and add the Building instance to this faction.
        """
        cost: Dict[str, int] = building.construction_cost  # e.g. {"wood": 20, "stone": 10}
        # Check if we have enough of each resource
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to build {building.name}")
        # Subtract the cost
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        # Add building
        self.buildings.append(building)

    def upgrade_structure(self, building: Building) -> None:
        """
        Pay the upgrade cost and then call the building's internal upgrade() method.
        Assumes building.upgrade_cost() returns a dict like construction_cost.
        """
        cost: Dict[str, int] = building.upgrade_cost()
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

    def distance(self, pos1: Position, pos2: Position) -> float:
        return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5

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
                    population=random.randint(8, 15),
                )
                self.add_faction(ai)
                new_factions.append(ai)
                spawned += 1
        return new_factions


class Game:
    def __init__(self, world: World | None = None):
        self.map = Map(*settings.MAP_SIZE)
        self.world = world or World(*settings.MAP_SIZE)
        self.resources = ResourceManager(self.world)
        self.player_faction: Faction | None = None

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)
        self.resources.register(self.player_faction)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        ai_factions = self.map.spawn_ai_factions(self.player_faction.settlement)
        for faction in self.map.factions:
            self.resources.register(faction)
        print("Game started with factions:")
        for faction in self.map.factions:
            pos = faction.settlement.position
            print(f"- {faction.name} at ({pos.x}, {pos.y})")

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
          3. Building‐based resource bonuses
        """
        for faction in self.map.factions:
            # 1. Population growth
            faction.population += 1

            # 2. Generate base food from population
            food_gain = faction.population // 2
            faction.resources["food"] = faction.resources.get("food", 0) + food_gain

            # 3. Building effects
            for building in faction.buildings:
                # Assume each Building instance has a `name` attribute indicating its type.
                b_type = getattr(building, "name", None)
                if b_type == "farm":
                    faction.resources["food"] = faction.resources.get("food", 0) + 5
                elif b_type == "lumber_mill":
                    faction.resources["wood"] = faction.resources.get("wood", 0) + 3
                elif b_type == "quarry":
                    faction.resources["stone"] = faction.resources.get("stone", 0) + 2

        # Debug output for the player faction
        if self.player_faction:
            res = self.player_faction.resources
            pop = self.player_faction.population
            print(f"Resources: {res} | Population: {pop}")


def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()


if __name__ == "__main__":
    main()
