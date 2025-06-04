from __future__ import annotations

"""World generation and management utilities."""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

Coordinate = Tuple[int, int]


class ResourceType(Enum):
    """Supported resource types found on hexes."""

    WOOD = "wood"
    STONE = "stone"
    ORE = "ore"


@dataclass(frozen=True)
class Road:
    start: Coordinate
    end: Coordinate


@dataclass
class WorldSettings:
    """Configuration values for world generation."""

    seed: int = 0
    width: int = 50
    height: int = 50
    biome_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "plains": 0.4,
            "forest": 0.3,
            "desert": 0.2,
            "mountain": 0.1,
        }
    )
    weather_patterns: Dict[str, float] = field(
        default_factory=lambda: {"rain": 0.3, "dry": 0.5, "snow": 0.2}
    )
    moisture: float = 0.5
    elevation: float = 0.5
    temperature: float = 0.5


@dataclass
class Hex:
    """Represents a single hex tile in the world."""

    coord: Coordinate
    terrain: str = "plains"
    elevation: float = 0.0
    moisture: float = 0.0
    temperature: float = 0.0
    resources: Dict[ResourceType, int] = field(default_factory=dict)
    flooded: bool = False
    ruined: bool = False

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)


def initialize_random(settings: WorldSettings) -> random.Random:
    """Create a random generator based on the provided seed."""
    return random.Random(settings.seed)


def generate_terrain_type(rng: random.Random, settings: WorldSettings) -> str:
    """Choose a biome based on distribution weights."""
    biomes = list(settings.biome_distribution.keys())
    weights = list(settings.biome_distribution.values())
    return rng.choices(biomes, weights=weights, k=1)[0]


def generate_resources(rng: random.Random, terrain: str) -> Dict[ResourceType, int]:
    """Generate resources for a hex based on its terrain."""
    resources: Dict[ResourceType, int] = {}

    if terrain == "forest":
        resources[ResourceType.WOOD] = rng.randint(5, 15)
        if rng.random() < 0.3:
            resources[ResourceType.STONE] = rng.randint(1, 4)
    elif terrain == "mountain":
        resources[ResourceType.STONE] = rng.randint(5, 15)
        if rng.random() < 0.7:
            resources[ResourceType.ORE] = rng.randint(1, 5)
    elif terrain == "plains":
        if rng.random() < 0.5:
            resources[ResourceType.WOOD] = rng.randint(1, 5)
        if rng.random() < 0.4:
            resources[ResourceType.STONE] = rng.randint(1, 4)
    elif terrain == "desert":
        if rng.random() < 0.2:
            resources[ResourceType.STONE] = rng.randint(1, 3)
    return resources


class World:
    """Collection of hexes with optional road network."""

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        *,
        seed: int = 0,
        settings: Optional[WorldSettings] = None,
    ) -> None:
        self.settings = settings or WorldSettings(seed=seed, width=width, height=height)
        self.hexes: List[List[Hex]] = []
        self.roads: List[Road] = []
        self.rng = initialize_random(self.settings)
        self._generate_hexes()

    @property
    def width(self) -> int:
        return self.settings.width

    @property
    def height(self) -> int:
        return self.settings.height

    def _generate_hexes(self) -> None:
        for r in range(self.settings.height):
            row: List[Hex] = []
            for q in range(self.settings.width):
                row.append(self._generate_hex(q, r))
            self.hexes.append(row)

    def _generate_hex(self, q: int, r: int) -> Hex:
        terrain = generate_terrain_type(self.rng, self.settings)
        elevation = self.rng.random() * self.settings.elevation
        moisture = self.rng.random() * self.settings.moisture
        temperature = self.rng.random() * self.settings.temperature
        resources = generate_resources(self.rng, terrain)
        return Hex(
            coord=(q, r),
            terrain=terrain,
            elevation=elevation,
            moisture=moisture,
            temperature=temperature,
            resources=resources,
        )

    def get(self, q: int, r: int) -> Optional[Hex]:
        if 0 <= q < self.settings.width and 0 <= r < self.settings.height:
            return self.hexes[r][q]
        return None

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        totals: Dict[ResourceType, int] = {r: 0 for r in ResourceType}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                hex_ = self.get(x + dx, y + dy)
                if hex_:
                    for rtype, amt in hex_.resources.items():
                        totals[rtype] += amt
        return {r: amt for r, amt in totals.items() if amt > 0}

    def has_road(self, start: Coordinate, end: Coordinate) -> bool:
        for road in self.roads:
            if (road.start == start and road.end == end) or (
                road.start == end and road.end == start
            ):
                return True
        return False

    def add_road(self, start: Coordinate, end: Coordinate) -> None:
        if not self.get(*start) or not self.get(*end):
            raise ValueError("Invalid road endpoints")
        if start == end:
            raise ValueError("Road must connect two different hexes")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Coordinate, end: Coordinate) -> float:
        base = 1.0
        if self.has_road(start, end):
            return base * 1.5
        return base


def adjust_settings(
    settings: WorldSettings,
    *,
    moisture: float | None = None,
    elevation: float | None = None,
    temperature: float | None = None,
) -> None:
    """Adjust world sliders before final generation."""
    if moisture is not None:
        settings.moisture = max(0.0, min(1.0, moisture))
    if elevation is not None:
        settings.elevation = max(0.0, min(1.0, elevation))
    if temperature is not None:
        settings.temperature = max(0.0, min(1.0, temperature))


__all__ = [
    "ResourceType",
    "WorldSettings",
    "Hex",
    "Road",
    "World",
    "adjust_settings",
]
