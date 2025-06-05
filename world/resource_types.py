# coding: utf-8
from __future__ import annotations

"""Resource type enumeration and categories."""

from enum import Enum
from typing import Set


class ResourceType(Enum):
    """Supported resource types found on hexes."""

    FOOD = "food"
    WHEAT = "wheat"
    FLOUR = "flour"
    BREAD = "bread"
    WOOD = "wood"
    STONE = "stone"
    ORE = "ore"
    METAL = "metal"
    CLOTH = "cloth"
    WOOL = "wool"
    CLOTHES = "clothes"
    PLANK = "plank"
    STONE_BLOCK = "stone_block"
    VEGETABLE = "vegetable"
    SOUP = "soup"
    GOLD = "gold"
    IRON = "iron"
    WEAPON = "weapon"
    RICE = "rice"
    CRABS = "crabs"
    FISH = "fish"
    CATTLE = "cattle"
    HORSES = "horses"
    PIGS = "pigs"
    CLAY = "clay"
    CHICKENS = "chickens"
    PEARLS = "pearls"
    SPICE = "spice"
    GEMS = "gems"
    TEA = "tea"
    ELEPHANTS = "elephants"


STRATEGIC_RESOURCES: Set[ResourceType] = {
    ResourceType.IRON,
    ResourceType.WEAPON,
    ResourceType.HORSES,
    ResourceType.ELEPHANTS,
}

LUXURY_RESOURCES: Set[ResourceType] = {
    ResourceType.GOLD,
    ResourceType.GEMS,
    ResourceType.PEARLS,
    ResourceType.SPICE,
    ResourceType.TEA,
}

__all__ = [
    "ResourceType",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
]
