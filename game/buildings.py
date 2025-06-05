from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from world.world import ResourceType
from . import settings
from .technology import TechLevel


if TYPE_CHECKING:
    from .models import Faction

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
    construction_cost: Dict[ResourceType, int]
    upkeep: int
    resource_bonus: int = 0
    population_bonus: int = 0
    victory_points: int = 0
    resource_type: Optional[ResourceType] = None
    level: int = 1
    tech_level: TechLevel = TechLevel.PRIMITIVE

    def upgrade_cost(self) -> Dict[ResourceType, int]:
        """Cost required to upgrade this building."""
        return {
            res: int(amount * 1.5 * self.level)
            for res, amount in self.construction_cost.items()
        }

    def upgrade(self) -> None:
        """Upgrade this building, improving bonuses and upkeep."""
        self.level += 1
        self.resource_bonus = int(self.resource_bonus * 1.5)
        self.population_bonus = int(self.population_bonus * 1.5)
        self.upkeep = int(self.upkeep * 1.2)


@dataclass
class Farm(Building):
    CLASS_ID = "Farm"
    name: str = "Farm"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(100 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 10
    resource_bonus: int = 5
    resource_type: ResourceType = ResourceType.FOOD
    tech_level: TechLevel = TechLevel.PRIMITIVE


@dataclass
class Mine(Building):
    CLASS_ID = "Mine"
    name: str = "Mine"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(150 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 15
    resource_bonus: int = 10
    resource_type: ResourceType = ResourceType.ORE
    tech_level: TechLevel = TechLevel.PRIMITIVE


@dataclass
class IronMine(Building):
    CLASS_ID = "IronMine"
    name: str = "IronMine"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(180 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 18
    resource_bonus: int = 2
    resource_type: ResourceType = ResourceType.IRON
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class GoldMine(Building):
    CLASS_ID = "GoldMine"
    name: str = "GoldMine"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(200 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 20
    resource_bonus: int = 1
    resource_type: ResourceType = ResourceType.GOLD
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class House(Building):
    CLASS_ID = "House"
    name: str = "House"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(50 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 5
    population_bonus: int = 2
    resource_type: Optional[ResourceType] = None
    tech_level: TechLevel = TechLevel.PRIMITIVE


@dataclass
class LumberMill(Building):
    CLASS_ID = "LumberMill"
    name: str = "LumberMill"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(120 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 12
    resource_bonus: int = 3
    resource_type: ResourceType = ResourceType.WOOD
    tech_level: TechLevel = TechLevel.PRIMITIVE


@dataclass
class Quarry(Building):
    CLASS_ID = "Quarry"
    name: str = "Quarry"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(130 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 14
    resource_bonus: int = 2
    resource_type: ResourceType = ResourceType.STONE
    tech_level: TechLevel = TechLevel.PRIMITIVE


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
    CLASS_ID = "Smeltery"
    name: str = "Smeltery"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(200 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 20
    input_resource: ResourceType = ResourceType.ORE
    output_resource: ResourceType = ResourceType.METAL
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class TextileMill(ProcessingBuilding):
    CLASS_ID = "TextileMill"
    name: str = "TextileMill"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(160 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 15
    input_resource: ResourceType = ResourceType.WOOD
    output_resource: ResourceType = ResourceType.CLOTH
    conversion_rate: int = 1
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Mill(ProcessingBuilding):
    CLASS_ID = "Mill"
    """Grinds wheat into flour."""

    name: str = "Mill"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(120 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 10
    input_resource: ResourceType = ResourceType.WHEAT
    output_resource: ResourceType = ResourceType.FLOUR
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Bakery(ProcessingBuilding):
    CLASS_ID = "Bakery"
    """Bakes flour into bread."""

    name: str = "Bakery"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(150 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 12
    input_resource: ResourceType = ResourceType.FLOUR
    output_resource: ResourceType = ResourceType.BREAD
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Forge(ProcessingBuilding):
    CLASS_ID = "Forge"
    """Forges iron into weapons."""

    name: str = "Forge"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(220 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 20
    input_resource: ResourceType = ResourceType.IRON
    output_resource: ResourceType = ResourceType.WEAPON
    conversion_rate: int = 1
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Tailor(ProcessingBuilding):
    CLASS_ID = "Tailor"
    """Turns wool into clothes."""

    name: str = "Tailor"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(160 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 15
    input_resource: ResourceType = ResourceType.WOOL
    output_resource: ResourceType = ResourceType.CLOTHES
    conversion_rate: int = 1
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Sawmill(ProcessingBuilding):
    CLASS_ID = "Sawmill"
    """Cuts wood into planks."""

    name: str = "Sawmill"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(140 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 12
    input_resource: ResourceType = ResourceType.WOOD
    output_resource: ResourceType = ResourceType.PLANK
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class Mason(ProcessingBuilding):
    CLASS_ID = "Mason"
    """Cuts stone into blocks."""

    name: str = "Mason"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(160 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 14
    input_resource: ResourceType = ResourceType.STONE
    output_resource: ResourceType = ResourceType.STONE_BLOCK
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.MEDIEVAL


@dataclass
class SoupKitchen(ProcessingBuilding):
    CLASS_ID = "SoupKitchen"
    """Turns vegetables into soup."""

    name: str = "SoupKitchen"
    construction_cost: Dict[ResourceType, int] = field(
        default_factory=lambda: {ResourceType.WOOD: int(110 * settings.SCALE_FACTOR)}
    )
    upkeep: int = 8
    input_resource: ResourceType = ResourceType.VEGETABLE
    output_resource: ResourceType = ResourceType.SOUP
    conversion_rate: int = 2
    tech_level: TechLevel = TechLevel.PRIMITIVE


# ---------------------------------------------------------------------------
# Convenience list of all buildable structures
# ---------------------------------------------------------------------------

ALL_BUILDING_CLASSES: List[type[Building]] = [
    Farm,
    Mine,
    IronMine,
    GoldMine,
    House,
    LumberMill,
    Quarry,
    Smeltery,
    TextileMill,
    Mill,
    Bakery,
    Forge,
    Tailor,
    Sawmill,
    Mason,
    SoupKitchen,
]

# Map CLASS_ID strings to the implementing class for easy lookup
BUILDING_ID_TO_CLASS: Dict[str, type[Building]] = {
    cls.CLASS_ID: cls for cls in ALL_BUILDING_CLASSES
}

