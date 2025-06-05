from __future__ import annotations

"""Simple diplomacy data models for trade and conflict."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, TYPE_CHECKING

from world.world import ResourceType

if TYPE_CHECKING:
    from .models import Faction


@dataclass
class TradeDeal:
    """Represents a bilateral trade agreement."""

    faction_a: "Faction"
    faction_b: "Faction"
    resources_a_to_b: Dict[ResourceType, int] = field(default_factory=dict)
    resources_b_to_a: Dict[ResourceType, int] = field(default_factory=dict)
    duration: int = 0  # 0 means infinite


@dataclass
class Truce:
    """Represents a temporary cessation of hostilities."""

    factions: Tuple["Faction", "Faction"]
    duration: int


@dataclass
class DeclarationOfWar:
    """Represents an active war between two factions."""

    factions: Tuple["Faction", "Faction"]

