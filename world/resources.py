from __future__ import annotations

"""Resource generation rules and helpers."""

import random
from typing import Dict, List, Tuple

from .resource_types import ResourceType


# Terrain -> list of (resource, min, max, probability)
RESOURCE_RULES: Dict[str, List[Tuple[ResourceType, int, int, float]]] = {
    "forest": [
        (ResourceType.WOOD, 5, 15, 1.0),
        (ResourceType.STONE, 1, 4, 0.3),
        (ResourceType.WOOL, 1, 3, 0.1),
        (ResourceType.PIGS, 1, 3, 0.2),
        (ResourceType.CHICKENS, 1, 3, 0.15),
        (ResourceType.CLAY, 1, 2, 0.1),
        (ResourceType.SPICE, 1, 1, 0.05),
        (ResourceType.TEA, 1, 1, 0.05),
        (ResourceType.COPPER, 1, 1, 0.05),
        (ResourceType.VEGETABLE, 1, 2, 0.05),
        (ResourceType.COFFEE, 1, 1, 0.05),
    ],
    "mountains": [
        (ResourceType.STONE, 5, 15, 1.0),
        (ResourceType.ORE, 1, 5, 0.7),
        (ResourceType.IRON, 1, 3, 0.4),
        (ResourceType.GOLD, 1, 2, 0.2),
        (ResourceType.GEMS, 1, 2, 0.2),
        (ResourceType.CLAY, 1, 2, 0.1),
        (ResourceType.COAL, 1, 3, 0.3),
        (ResourceType.COPPER, 1, 3, 0.4),
        (ResourceType.SILVER, 1, 2, 0.2),
    ],
    "hills": [
        (ResourceType.WOOD, 1, 5, 0.5),
        (ResourceType.STONE, 1, 4, 0.6),
        (ResourceType.ORE, 1, 3, 0.4),
        (ResourceType.IRON, 1, 2, 0.2),
        (ResourceType.GOLD, 1, 1, 0.05),
        (ResourceType.CLAY, 1, 3, 0.1),
        (ResourceType.HORSES, 1, 2, 0.05),
        (ResourceType.GEMS, 1, 1, 0.05),
        (ResourceType.COAL, 1, 2, 0.2),
        (ResourceType.COPPER, 1, 3, 0.3),
        (ResourceType.SILVER, 1, 1, 0.1),
    ],
    "plains": [
        (ResourceType.WOOD, 1, 5, 0.5),
        (ResourceType.STONE, 1, 4, 0.4),
        (ResourceType.WHEAT, 1, 4, 0.3),
        (ResourceType.WOOL, 1, 2, 0.2),
        (ResourceType.RICE, 1, 3, 0.4),
        (ResourceType.CATTLE, 1, 3, 0.25),
        (ResourceType.HORSES, 1, 2, 0.15),
        (ResourceType.PIGS, 1, 2, 0.2),
        (ResourceType.CHICKENS, 1, 3, 0.25),
        (ResourceType.CLAY, 1, 2, 0.1),
        (ResourceType.ELEPHANTS, 1, 1, 0.05),
        (ResourceType.SALT, 1, 1, 0.05),
        (ResourceType.COPPER, 1, 2, 0.1),
        (ResourceType.VEGETABLE, 1, 2, 0.05),
        (ResourceType.TEA, 1, 1, 0.05),
        (ResourceType.COFFEE, 1, 1, 0.05),
    ],
    "desert": [
        (ResourceType.STONE, 1, 3, 0.2),
        (ResourceType.ORE, 1, 2, 0.1),
        (ResourceType.GOLD, 1, 1, 0.05),
        (ResourceType.SPICE, 1, 2, 0.1),
        (ResourceType.CLAY, 1, 2, 0.05),
        (ResourceType.SALT, 1, 2, 0.15),
    ],
    "tundra": [
        (ResourceType.STONE, 1, 4, 0.3),
        (ResourceType.WOOD, 1, 3, 0.2),
        (ResourceType.WOOL, 1, 3, 0.25),
        (ResourceType.CATTLE, 1, 2, 0.05),
    ],
    "rainforest": [
        (ResourceType.WOOD, 8, 20, 1.0),
        (ResourceType.VEGETABLE, 1, 3, 0.3),
        (ResourceType.WHEAT, 1, 2, 0.15),
        (ResourceType.WOOL, 1, 2, 0.1),
        (ResourceType.SPICE, 1, 2, 0.25),
        (ResourceType.TEA, 1, 2, 0.2),
        (ResourceType.ELEPHANTS, 1, 1, 0.1),
        (ResourceType.PIGS, 1, 2, 0.15),
        (ResourceType.CHICKENS, 1, 2, 0.1),
        (ResourceType.CLAY, 1, 2, 0.1),
        (ResourceType.COFFEE, 1, 2, 0.2),
    ],
    "water": [
        (ResourceType.FISH, 1, 5, 0.5),
        (ResourceType.CRABS, 1, 3, 0.3),
        (ResourceType.PEARLS, 1, 1, 0.05),
    ],
    "floating_island": [
        (ResourceType.GEMS, 1, 3, 0.2),
        (ResourceType.GOLD, 1, 2, 0.1),
        (ResourceType.WOOD, 1, 4, 0.3),
    ],
    "crystal_forest": [
        (ResourceType.GEMS, 1, 4, 0.4),
        (ResourceType.WOOD, 2, 6, 0.8),
        (ResourceType.SPICE, 1, 2, 0.1),
    ],
}


def generate_resources(rng: random.Random, terrain: str) -> Dict[ResourceType, int]:
    """Return resources for a given terrain using RESOURCE_RULES."""
    return {
        res: rng.randint(lo, hi)
        for res, lo, hi, p in RESOURCE_RULES.get(terrain, [])
        if rng.random() < p
    }


__all__ = ["generate_resources", "RESOURCE_RULES"]
