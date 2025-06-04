"""Game package exposing core classes."""

from .game import Game, Faction
from .models import Settlement, Position
from .buildings import (
    Building,
    Farm,
    Mine,
    House,
    LumberMill,
    Quarry,
    WALLS,
    FORT,
    FLOOD_BARRIER,
    FIREBREAK,
    mitigate_population_loss,
    mitigate_building_damage,
)

__all__ = [
    "Game",
    "Faction",
    "Settlement",
    "Position",
    "Building",
    "Farm",
    "Mine",
    "House",
    "LumberMill",
    "Quarry",
    "WALLS",
    "FORT",
    "FLOOD_BARRIER",
    "FIREBREAK",
    "mitigate_population_loss",
    "mitigate_building_damage",
]
