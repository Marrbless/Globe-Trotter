"""Game package exposing core classes."""

from .game import Game
from .models import Faction, Settlement, Position
from .diplomacy import TradeDeal, Truce, DeclarationOfWar, Alliance
from .technology import TechLevel
from .buildings import (
    Building,
    Farm,
    Mine,
    IronMine,
    GoldMine,
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
    "IronMine",
    "GoldMine",
    "House",
    "LumberMill",
    "Quarry",
    "WALLS",
    "FORT",
    "FLOOD_BARRIER",
    "FIREBREAK",
    "mitigate_population_loss",
    "mitigate_building_damage",
    "TradeDeal",
    "Truce",
    "DeclarationOfWar",
    "Alliance",
    "TechLevel",
]

# Expose the world package as a submodule to allow `import game.world`
import sys as _sys
import world as _world
_sys.modules[__name__ + ".world"] = _world
