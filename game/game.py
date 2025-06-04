import random
from dataclasses import dataclass, field
from typing import List

from .buildings import Building

from . import settings

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
    resources: int = 500
    buildings: List[Building] = field(default_factory=list)

    def build_structure(self, building: Building) -> None:
        if self.resources < building.construction_cost:
            raise ValueError("Not enough resources to build")
        self.resources -= building.construction_cost
        self.buildings.append(building)

    def upgrade_structure(self, building: Building) -> None:
        cost = building.upgrade_cost()
        if self.resources < cost:
            raise ValueError("Not enough resources to upgrade")
        self.resources -= cost
        building.upgrade()

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.factions = []

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

    def spawn_ai_factions(self, player_settlement: Settlement):
        count = settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
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
                ai = Faction(name=f"AI #{spawned + 1}", settlement=Settlement(name=f"AI Town {spawned + 1}", position=pos))
                self.add_faction(ai)
                spawned += 1

class Game:
    def __init__(self):
        self.map = Map(*settings.MAP_SIZE)
        self.player_faction: Faction | None = None

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        self.map.spawn_ai_factions(self.player_faction.settlement)
        print("Game started with factions:")
        for faction in self.map.factions:
            print(f"- {faction.name} at {faction.settlement.position}")

    def build_for_player(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.build_structure(building)

    def upgrade_player_building(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.upgrade_structure(building)

def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()

if __name__ == "__main__":
    main()
