"""Simple game package exposing common classes."""

from .game import Game, Position, Settlement, Faction
from .buildings import (
    Building,
    WALLS,
    FORT,
    FLOOD_BARRIER,
    FIREBREAK,
    mitigate_population_loss,
    mitigate_building_damage,
)

__all__ = [
    "Game",
    "Position",
    "Settlement",
    "Faction",
    "Building",
    "WALLS",
    "FORT",
    "FLOOD_BARRIER",
    "FIREBREAK",
    "mitigate_population_loss",
    "mitigate_building_damage",
]
