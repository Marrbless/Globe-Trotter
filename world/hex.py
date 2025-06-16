from __future__ import annotations

"""
Data model for a single world hex tile with improved type safety, water‐state consolidation,
resource encapsulation, and debug‐friendly repre­sentation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union

from .resource_types import ResourceType

Coordinate = Tuple[int, int]


class TerrainType(Enum):
    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAINS = "mountains"
    HILLS = "hills"
    DESERT = "desert"
    TUNDRA = "tundra"
    RAINFOREST = "rainforest"
    WATER = "water"
    FLOATING_ISLAND = "floating_island"
    CRYSTAL_FOREST = "crystal_forest"


class WaterState(Enum):
    NONE = 0
    RIVER = 1
    LAKE = 2
    PERSISTENT_LAKE = 3


@dataclass
class Hex:
    """
    Represents a single hex tile in the world.

    Core Attributes:
      coord: Axial grid coordinate of this tile (q, r).
      terrain: One of TerrainType. Defaults to PLAINS.
      elevation: Elevation at this tile (sea level = 0.0).
      moisture: Moisture level 0.0–1.0.
      temperature: Temperature in degrees Celsius.
      resources: Mapping of ResourceType to integer quantity.
      ruined: If True, this tile cannot be used or traversed normally.
      ley_line: If True, a magical ley line crosses this tile.
      water_state: One of WaterState indicating whether this tile has water.
      water_flow: Volume of water passing this tile per time unit.
    """

    coord: Coordinate
    terrain: TerrainType = TerrainType.PLAINS
    elevation: float = 0.0
    moisture: float = 0.0
    temperature: float = 0.0
    resources: Dict[ResourceType, int] = field(default_factory=dict)
    ruined: bool = False
    ley_line: bool = False
    water_state: WaterState = WaterState.NONE
    water_flow: float = 0.0
    # Legacy water flags used by tests and persistence
    river: bool = False
    lake: bool = False
    persistent_lake: bool = False
    flooded: bool = False

    def __post_init__(self):
        # Ensure resources is a fresh dict, not a shared reference
        if type(self.resources) is not dict:
            raise TypeError("`resources` must be a plain dict, not shared or a subclass.")
        # Validate terrain type
        if not isinstance(self.terrain, TerrainType):
            raise TypeError(f"terrain must be a TerrainType, not {type(self.terrain)}")
        # Validate water_flow
        if self.water_flow < 0:
            raise ValueError("water_flow cannot be negative.")
        # If tile is ruined, force water_state to NONE and water_flow to 0
        if self.ruined:
            self.water_state = WaterState.NONE
            self.water_flow = 0.0

    def __getitem__(self, key: str) -> Union[str, float, bool, WaterState, Dict[ResourceType, int]]:
        """
        Allows attribute access via indexing, e.g., tile['terrain'].

        Raises:
            AttributeError if the key is invalid.
        """
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        """
        Allows setting attributes via indexing, e.g., tile['terrain'] = TerrainType.FOREST.

        Raises:
            AttributeError if the key is invalid.
        """
        setattr(self, key, value)

    @property
    def is_watered(self) -> bool:
        """
        True if this tile has any water feature (river, lake, or persistent lake).
        """
        return self.water_state != WaterState.NONE

    @property
    def has_any_resources(self) -> bool:
        """
        True if this tile contains any quantity of resources.
        """
        return bool(self.resources)

    @property
    def resource_list(self) -> List[Tuple[ResourceType, int]]:
        """
        Returns a list of (ResourceType, quantity), sorted by quantity descending.
        """
        return sorted(self.resources.items(), key=lambda kv: kv[1], reverse=True)

    def set_terrain_by_name(self, name: str) -> None:
        """
        Sets terrain based on a case-insensitive string lookup. Raises ValueError if invalid.
        """
        name_lower = name.strip().lower()
        for t in TerrainType:
            if t.value == name_lower:
                self.terrain = t
                return
        valid = ", ".join(t.value for t in TerrainType)
        raise ValueError(f"Invalid terrain '{name}'. Valid values: {valid}")

    def __repr__(self) -> str:
        """
        Debug-friendly representation focusing on key attributes.
        Shows coord, terrain, water_state (if not NONE), and top 3 resources.
        """
        base = f"Hex(coord={self.coord}, terrain={self.terrain.value}"
        if self.water_state != WaterState.NONE:
            base += f", water_state={self.water_state.name}"
        if self.ruined:
            base += ", RUINED"
        if self.ley_line:
            base += ", LEY_LINE"
        if self.resources:
            top_resources = self.resource_list[:3]
            res_str = ", ".join(f"{rt.name}:{qty}" for rt, qty in top_resources)
            base += f", resources=[{res_str}]"
        base += ")"
        return base

    def to_json(self) -> Dict[str, Union[str, float, bool, Dict[str, int]]]:
        """
        Serializes core attributes to a JSON‐friendly dict.
        """
        return {
            "coord": {"q": self.coord[0], "r": self.coord[1]},
            "terrain": self.terrain.value,
            "elevation": self.elevation,
            "moisture": self.moisture,
            "temperature": self.temperature,
            "resources": {rt.name: qty for rt, qty in self.resources.items()},
            "ruined": self.ruined,
            "ley_line": self.ley_line,
            "water_state": self.water_state.name,
            "water_flow": self.water_flow,
            "river": self.river,
            "lake": self.lake,
            "persistent_lake": self.persistent_lake,
            "flooded": self.flooded,
        }


__all__ = ["Hex", "Coordinate", "TerrainType", "WaterState"]
