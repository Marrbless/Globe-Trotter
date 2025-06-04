import random
from dataclasses import dataclass

from . import settings
from .buildings import (
    Building,
    mitigate_building_damage,
    mitigate_population_loss,
)


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
    def __init__(self):
        self.map = Map(*settings.MAP_SIZE)
        self.player_faction: Faction | None = None
        self.player_buildings: list[Building] = []

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)

    def add_building(self, building: Building):
        """Add a defensive building to the player's settlement."""
        self.player_buildings.append(building)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        self.map.spawn_ai_factions(self.player_faction.settlement)
        print("Game started with factions:")
        for faction in self.map.factions:
            print(f"- {faction.name} at {faction.settlement.position}")
        self.simulate_events()

    def simulate_events(self):
        """Run sample attack and disaster to show defensive buildings."""
        if not self.player_faction:
            return
        base_pop_loss = 100
        base_damage = 50
        pop_loss = mitigate_population_loss(self.player_buildings, base_pop_loss)
        damage = mitigate_building_damage(self.player_buildings, base_damage)
        print(
            f"Population loss mitigated from {base_pop_loss} to {pop_loss}"
        )
        print(
            f"Building damage mitigated from {base_damage} to {damage}"
        )

def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()

if __name__ == "__main__":
    main()
