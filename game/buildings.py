from dataclasses import dataclass
from typing import List

# Categories for defensive structures
FACTION_DEFENSE = "faction"
WORLD_DEFENSE = "world"

@dataclass
class DefensiveBuilding:
    """Building that provides defense."""
    name: str
    category: str
    defense_value: float = 0.0

# Pre-defined defensive buildings
WALLS = DefensiveBuilding("Walls", FACTION_DEFENSE, defense_value=0.2)
FORT = DefensiveBuilding("Fort", FACTION_DEFENSE, defense_value=0.4)
FLOOD_BARRIER = DefensiveBuilding("Flood Barrier", WORLD_DEFENSE, defense_value=0.3)
FIREBREAK = DefensiveBuilding("Firebreak", WORLD_DEFENSE, defense_value=0.3)

# Convenience list
ALL_DEFENSIVE_BUILDINGS = [WALLS, FORT, FLOOD_BARRIER, FIREBREAK]


def mitigate_population_loss(buildings: List[DefensiveBuilding], loss: int) -> int:
    """Reduce population loss from attacks using faction defense structures."""
    factor = 1.0
    for b in buildings:
        if b.category == FACTION_DEFENSE:
            factor *= 1 - b.defense_value
    return max(0, int(loss * factor))


def mitigate_building_damage(buildings: List[DefensiveBuilding], damage: int) -> int:
    """Reduce building damage from disasters using world defense structures."""
    factor = 1.0
    for b in buildings:
        if b.category == WORLD_DEFENSE:
            factor *= 1 - b.defense_value
    return max(0, int(damage * factor))


@dataclass
class Building:
    """Base class for all buildings."""
    name: str
    construction_cost: int
    upkeep: int
    resource_bonus: int = 0
    population_bonus: int = 0
    level: int = 1

    def upgrade_cost(self) -> int:
        """Cost required to upgrade this building."""
        return int(self.construction_cost * 1.5 * self.level)

    def upgrade(self) -> None:
        """Upgrade this building, improving bonuses and upkeep."""
        self.level += 1
        self.resource_bonus = int(self.resource_bonus * 1.5)
        self.population_bonus = int(self.population_bonus * 1.5)
        self.upkeep = int(self.upkeep * 1.2)


@dataclass
class Farm(Building):
    name: str = "Farm"
    construction_cost: int = 100
    upkeep: int = 10
    resource_bonus: int = 5


@dataclass
class Mine(Building):
    name: str = "Mine"
    construction_cost: int = 150
    upkeep: int = 15
    resource_bonus: int = 10


@dataclass
class House(Building):
    name: str = "House"
    construction_cost: int = 50
    upkeep: int = 5
    population_bonus: int = 2
