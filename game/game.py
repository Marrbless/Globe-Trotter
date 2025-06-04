import random
from dataclasses import dataclass

from .persistence import GameState, load_state, save_state

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
    def __init__(self, state: GameState | None = None):
        self.map = Map(*settings.MAP_SIZE)
        self.player_faction: Faction | None = None
        self.state = state or load_state()
        self.resources = self.state.resources
        self.population = self.state.population

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
        print(f"Resources: {self.state.resources}")
        print(f"Population: {self.state.population}")

    def save(self) -> None:
        self.state.resources = self.resources
        self.state.population = self.population
        save_state(self.state)

def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()
    game.save()

if __name__ == "__main__":
    main()
