from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from world.world import ResourceType

if TYPE_CHECKING:
    from .game import Faction

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
    resource_type: Optional[ResourceType] = None
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
    resource_type: ResourceType = ResourceType.FOOD


@dataclass
class Mine(Building):
    name: str = "Mine"
    construction_cost: int = 150
    upkeep: int = 15
    resource_bonus: int = 10
    resource_type: ResourceType = ResourceType.ORE


@dataclass
class IronMine(Building):
    name: str = "IronMine"
    construction_cost: int = 180
    upkeep: int = 18
    resource_bonus: int = 2
    resource_type: ResourceType = ResourceType.IRON


@dataclass
class GoldMine(Building):
    name: str = "GoldMine"
    construction_cost: int = 200
    upkeep: int = 20
    resource_bonus: int = 1
    resource_type: ResourceType = ResourceType.GOLD


@dataclass
class House(Building):
    name: str = "House"
    construction_cost: int = 50
    upkeep: int = 5
    population_bonus: int = 2
    resource_type: Optional[ResourceType] = None


@dataclass
class LumberMill(Building):
    name: str = "LumberMill"
    construction_cost: int = 120
    upkeep: int = 12
    resource_bonus: int = 3
    resource_type: ResourceType = ResourceType.WOOD


@dataclass
class Quarry(Building):
    name: str = "Quarry"
    construction_cost: int = 130
    upkeep: int = 14
    resource_bonus: int = 2
    resource_type: ResourceType = ResourceType.STONE


@dataclass
class ProcessingBuilding(Building):
    """Building that converts one resource into another each tick."""
    input_resource: ResourceType = ResourceType.ORE
    output_resource: ResourceType = ResourceType.METAL
    conversion_rate: int = 1

    def process(self, faction: Faction) -> None:
        available = faction.resources.get(self.input_resource, 0)
        to_convert = min(self.conversion_rate, available)
        if to_convert > 0:
            faction.resources[self.input_resource] -= to_convert
            faction.resources[self.output_resource] = faction.resources.get(
                self.output_resource, 0
            ) + to_convert


@dataclass
class Smeltery(ProcessingBuilding):
    name: str = "Smeltery"
    construction_cost: int = 200
    upkeep: int = 20
    input_resource: ResourceType = ResourceType.ORE
    output_resource: ResourceType = ResourceType.METAL
    conversion_rate: int = 2


@dataclass
class TextileMill(ProcessingBuilding):
    name: str = "TextileMill"
    construction_cost: int = 160
    upkeep: int = 15
    input_resource: ResourceType = ResourceType.WOOD
    output_resource: ResourceType = ResourceType.CLOTH
    conversion_rate: int = 1
