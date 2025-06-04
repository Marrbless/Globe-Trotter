from dataclasses import dataclass
from typing import List

# Categories for defensive structures
FACTION_DEFENSE = "faction"
WORLD_DEFENSE = "world"

@dataclass
class Building:
    """Basic building with a defensive value."""
    name: str
    category: str
    defense_value: float = 0.0

# Pre-defined defensive buildings
WALLS = Building("Walls", FACTION_DEFENSE, defense_value=0.2)
FORT = Building("Fort", FACTION_DEFENSE, defense_value=0.4)
FLOOD_BARRIER = Building("Flood Barrier", WORLD_DEFENSE, defense_value=0.3)
FIREBREAK = Building("Firebreak", WORLD_DEFENSE, defense_value=0.3)

# Convenience list
ALL_BUILDINGS = [WALLS, FORT, FLOOD_BARRIER, FIREBREAK]


def mitigate_population_loss(buildings: List[Building], loss: int) -> int:
    """Reduce population loss from attacks using faction defense structures."""
    factor = 1.0
    for b in buildings:
        if b.category == FACTION_DEFENSE:
            factor *= 1 - b.defense_value
    return max(0, int(loss * factor))


def mitigate_building_damage(buildings: List[Building], damage: int) -> int:
    """Reduce building damage from disasters using world defense structures."""
    factor = 1.0
    for b in buildings:
        if b.category == WORLD_DEFENSE:
            factor *= 1 - b.defense_value
    return max(0, int(damage * factor))
