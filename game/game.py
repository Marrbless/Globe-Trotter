import random
from dataclasses import dataclass, field
from typing import Dict

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
    population: int = 10
    workers: Dict[str, int] = field(default_factory=lambda: {"food": 10, "wood": 0, "stone": 0})

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.factions: list[Faction] = []

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

    def spawn_ai_factions(self, player_settlement: Settlement) -> list[Faction]:
        count = settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
        factions: list[Faction] = []
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
                factions.append(ai)
                spawned += 1
        return factions

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
            print(f"- {faction.name} at {faction.settlement.position}")

    def tick(self) -> None:
        """Advance the game by one tick."""
        self.resources.tick(self.map.factions)

def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()

if __name__ == "__main__":
    main()
