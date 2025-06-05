from __future__ import annotations

"""World generation and management utilities."""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

from .generation import (
    perlin_noise,
    terrain_from_elevation,
    determine_biome,
    generate_elevation_map,
    generate_temperature_map,
    generate_rainfall,
    generate_biome_map,
)

Coordinate = Tuple[int, int]


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


# Categorization helpers for resource types
STRATEGIC_RESOURCES = {
    ResourceType.IRON,
    ResourceType.WEAPON,
    ResourceType.HORSES,
    ResourceType.ELEPHANTS,
}

LUXURY_RESOURCES = {
    ResourceType.GOLD,
    ResourceType.GEMS,
    ResourceType.PEARLS,
    ResourceType.SPICE,
    ResourceType.TEA,
}


@dataclass(frozen=True)
class Road:
    start: Coordinate
    end: Coordinate


@dataclass(frozen=True)
class RiverSegment:
    """A start/end pair describing a single river edge."""
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
            "plains": 0.3,
            "forest": 0.25,
            "hills": 0.2,
            "desert": 0.15,
            "mountains": 0.05,
            "water": 0.05,
        }
    )
    weather_patterns: Dict[str, float] = field(
        default_factory=lambda: {"rain": 0.3, "dry": 0.5, "snow": 0.2}
    )
    moisture: float = 0.5
    elevation: float = 0.5
    temperature: float = 0.5
    rainfall_intensity: float = 0.5
    disaster_intensity: float = 0.0
    sea_level: float = 0.3
    plate_activity: float = 0.5
    base_height: float = 0.5
    world_changes: bool = True


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
    river: bool = False
    lake: bool = False

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
        if rng.random() < 0.1:
            resources[ResourceType.WOOL] = rng.randint(1, 3)
        if rng.random() < 0.2:
            resources[ResourceType.PIGS] = rng.randint(1, 3)
        if rng.random() < 0.15:
            resources[ResourceType.CHICKENS] = rng.randint(1, 3)
        if rng.random() < 0.1:
            resources[ResourceType.CLAY] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.SPICE] = rng.randint(1, 1)
        if rng.random() < 0.05:
            resources[ResourceType.TEA] = rng.randint(1, 1)
    elif terrain == "mountains":
        resources[ResourceType.STONE] = rng.randint(5, 15)
        if rng.random() < 0.7:
            resources[ResourceType.ORE] = rng.randint(1, 5)
        if rng.random() < 0.4:
            resources[ResourceType.IRON] = rng.randint(1, 3)
        if rng.random() < 0.2:
            resources[ResourceType.GOLD] = rng.randint(1, 2)
        if rng.random() < 0.2:
            resources[ResourceType.GEMS] = rng.randint(1, 2)
        if rng.random() < 0.1:
            resources[ResourceType.CLAY] = rng.randint(1, 2)
    elif terrain == "hills":
        if rng.random() < 0.5:
            resources[ResourceType.WOOD] = rng.randint(1, 5)
        if rng.random() < 0.6:
            resources[ResourceType.STONE] = rng.randint(1, 4)
        if rng.random() < 0.4:
            resources[ResourceType.ORE] = rng.randint(1, 3)
        if rng.random() < 0.2:
            resources[ResourceType.IRON] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.GOLD] = rng.randint(1, 1)
        if rng.random() < 0.1:
            resources[ResourceType.CLAY] = rng.randint(1, 3)
        if rng.random() < 0.05:
            resources[ResourceType.HORSES] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.GEMS] = rng.randint(1, 1)
    elif terrain == "plains":
        if rng.random() < 0.5:
            resources[ResourceType.WOOD] = rng.randint(1, 5)
        if rng.random() < 0.4:
            resources[ResourceType.STONE] = rng.randint(1, 4)
        if rng.random() < 0.3:
            resources[ResourceType.WHEAT] = rng.randint(1, 4)
        if rng.random() < 0.2:
            resources[ResourceType.WOOL] = rng.randint(1, 2)
        if rng.random() < 0.4:
            resources[ResourceType.RICE] = rng.randint(1, 3)
        if rng.random() < 0.25:
            resources[ResourceType.CATTLE] = rng.randint(1, 3)
        if rng.random() < 0.15:
            resources[ResourceType.HORSES] = rng.randint(1, 2)
        if rng.random() < 0.2:
            resources[ResourceType.PIGS] = rng.randint(1, 2)
        if rng.random() < 0.25:
            resources[ResourceType.CHICKENS] = rng.randint(1, 3)
        if rng.random() < 0.1:
            resources[ResourceType.CLAY] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.ELEPHANTS] = rng.randint(1, 1)
    elif terrain == "desert":
        if rng.random() < 0.2:
            resources[ResourceType.STONE] = rng.randint(1, 3)
        if rng.random() < 0.1:
            resources[ResourceType.ORE] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.GOLD] = rng.randint(1, 1)
        if rng.random() < 0.1:
            resources[ResourceType.SPICE] = rng.randint(1, 2)
        if rng.random() < 0.05:
            resources[ResourceType.CLAY] = rng.randint(1, 2)
    elif terrain == "tundra":
        if rng.random() < 0.3:
            resources[ResourceType.STONE] = rng.randint(1, 4)
        if rng.random() < 0.2:
            resources[ResourceType.WOOD] = rng.randint(1, 3)
        if rng.random() < 0.25:
            resources[ResourceType.WOOL] = rng.randint(1, 3)
        if rng.random() < 0.05:
            resources[ResourceType.CATTLE] = rng.randint(1, 2)
    elif terrain == "rainforest":
        resources[ResourceType.WOOD] = rng.randint(8, 20)
        if rng.random() < 0.3:
            resources[ResourceType.VEGETABLE] = rng.randint(1, 3)
        if rng.random() < 0.15:
            resources[ResourceType.WHEAT] = rng.randint(1, 2)
        if rng.random() < 0.1:
            resources[ResourceType.WOOL] = rng.randint(1, 2)
        if rng.random() < 0.25:
            resources[ResourceType.SPICE] = rng.randint(1, 2)
        if rng.random() < 0.2:
            resources[ResourceType.TEA] = rng.randint(1, 2)
        if rng.random() < 0.1:
            resources[ResourceType.ELEPHANTS] = rng.randint(1, 1)
        if rng.random() < 0.15:
            resources[ResourceType.PIGS] = rng.randint(1, 2)
        if rng.random() < 0.1:
            resources[ResourceType.CHICKENS] = rng.randint(1, 2)
        if rng.random() < 0.1:
            resources[ResourceType.CLAY] = rng.randint(1, 2)
    elif terrain == "water":
        if rng.random() < 0.5:
            resources[ResourceType.FISH] = rng.randint(1, 5)
        if rng.random() < 0.3:
            resources[ResourceType.CRABS] = rng.randint(1, 3)
        if rng.random() < 0.05:
            resources[ResourceType.PEARLS] = rng.randint(1, 1)

    return resources


class World:
    """Collection of hexes with optional road network."""

    CHUNK_SIZE = 10

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        *,
        seed: int = 0,
        settings: Optional[WorldSettings] = None,
    ) -> None:
        self.settings = settings or WorldSettings(seed=seed, width=width, height=height)
        self.chunks: Dict[Tuple[int, int], List[List[Hex]]] = {}
        self.hexes: List[List[Hex]] = []
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []
        self.rng = initialize_random(self.settings)

        # Precompute world data maps
        self.elevation_map = generate_elevation_map(
            self.settings.width, self.settings.height, self.settings
        )
        self.temperature_map = generate_temperature_map(self.settings, self.rng)
        self.rainfall_map = generate_rainfall(
            self.elevation_map, self.settings, self.rng
        )
        self.biome_map = generate_biome_map(
            self.elevation_map, self.temperature_map, self.rainfall_map
        )

        self._initialize_base_area()
        self._generate_rivers()

    @property
    def width(self) -> int:
        return self.settings.width

    @property
    def height(self) -> int:
        return self.settings.height

    def _initialize_base_area(self) -> None:
        """Populate hexes for the entire world by lazily generating each chunk."""
        for r in range(self.settings.height):
            row: List[Hex] = []
            for q in range(self.settings.width):
                row.append(self.get(q, r))
            self.hexes.append(row)

    def _generate_hex(self, q: int, r: int) -> Hex:
        """Generate a single hex tile using precomputed climate maps."""
        rng = random.Random(hash((q, r, self.settings.seed)))

        elevation = self.elevation_map[r][q]
        temperature = self.temperature_map[r][q]
        moisture = self.rainfall_map[r][q]
        terrain = determine_biome(elevation, temperature, moisture)
        resources = generate_resources(rng, terrain)

        return Hex(
            coord=(q, r),
            terrain=terrain,
            elevation=elevation,
            moisture=moisture,
            temperature=temperature,
            resources=resources,
        )

    def _generate_chunk(self, cx: int, cy: int) -> None:
        """Generate a CHUNK_SIZE × CHUNK_SIZE block of hexes on demand."""
        chunk: List[List[Hex]] = []
        base_q = cx * self.CHUNK_SIZE
        base_r = cy * self.CHUNK_SIZE
        y_limit = min(self.CHUNK_SIZE, self.height - base_r)
        x_limit = min(self.CHUNK_SIZE, self.width - base_q)
        for r_off in range(y_limit):
            row: List[Hex] = []
            for q_off in range(x_limit):
                q = base_q + q_off
                r = base_r + r_off
                if 0 <= q < self.width and 0 <= r < self.height:
                    row.append(self._generate_hex(q, r))
                else:
                    row.append(Hex(coord=(q, r)))
            chunk.append(row)
        self.chunks[(cx, cy)] = chunk

    def _neighbors(self, q: int, r: int) -> List[Coordinate]:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        return [
            (q + dq, r + dr)
            for dq, dr in directions
            if 0 <= q + dq < self.settings.width and 0 <= r + dr < self.settings.height
        ]

    def _downhill_neighbor(self, q: int, r: int) -> Optional[Coordinate]:
        current = self.get(q, r)
        if current is None:
            return None
        best: Optional[Coordinate] = None
        best_elev = current.elevation
        for nq, nr in self._neighbors(q, r):
            neighbor = self.get(nq, nr)
            if neighbor and neighbor.elevation < best_elev:
                best_elev = neighbor.elevation
                best = (nq, nr)
        return best

    def _generate_rivers(self) -> None:
        """Create simple rivers flowing downhill based on precomputed elevation."""
        density = max(0.0, min(1.0, self.settings.rainfall_intensity))
        seeds = max(1, int(density * 5))
        avg_elev = sum(sum(row) for row in self.elevation_map) / (
            self.width * self.height
        )
        threshold = max(self.settings.sea_level, avg_elev)
        for _ in range(seeds):
            # choose a random high-elevation starting hex
            for _ in range(100):
                q = self.rng.randint(0, self.width - 1)
                r = self.rng.randint(0, self.height - 1)
                h = self.get(q, r)
                if h and h.elevation > threshold:
                    break
            else:
                continue

            current = (q, r)
            visited: set[Coordinate] = set()
            while current and current not in visited:
                visited.add(current)
                nxt = self._downhill_neighbor(*current)
                if (
                    not nxt
                    or nxt == current
                    or not (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height)
                ):
                    self.lakes.append(current)
                    self.get(*current).lake = True
                    break
                self.rivers.append(RiverSegment(current, nxt))
                self.get(*current).river = True
                current = nxt

    def get(self, q: int, r: int) -> Optional[Hex]:
        """Retrieve a hex at (q, r), generating its chunk if necessary."""
        if not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
            return None
        cx = q // self.CHUNK_SIZE
        cy = r // self.CHUNK_SIZE
        if (cx, cy) not in self.chunks:
            self._generate_chunk(cx, cy)
        chunk = self.chunks.get((cx, cy))
        if chunk is None:
            return None
        return chunk[r % self.CHUNK_SIZE][q % self.CHUNK_SIZE]

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        """Sum up resources of all hexes within a given radius."""
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
    rainfall_intensity: float | None = None,
    disaster_intensity: float | None = None,
    sea_level: float | None = None,
    plate_activity: float | None = None,
    base_height: float | None = None,
) -> None:
    """Adjust world sliders before final generation."""
    if moisture is not None:
        settings.moisture = max(0.0, min(1.0, moisture))
    if elevation is not None:
        settings.elevation = max(0.0, min(1.0, elevation))
    if temperature is not None:
        settings.temperature = max(0.0, min(1.0, temperature))
    if rainfall_intensity is not None:
        settings.rainfall_intensity = max(0.0, min(1.0, rainfall_intensity))
    if disaster_intensity is not None:
        settings.disaster_intensity = max(0.0, min(1.0, disaster_intensity))
    if sea_level is not None:
        settings.sea_level = max(0.0, min(1.0, sea_level))
    if plate_activity is not None:
        settings.plate_activity = max(0.0, min(1.0, plate_activity))
    if base_height is not None:
        settings.base_height = max(0.0, min(1.0, base_height))


__all__ = [
    "ResourceType",
    "WorldSettings",
    "Hex",
    "Road",
    "RiverSegment",
    "World",
    "adjust_settings",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
]
