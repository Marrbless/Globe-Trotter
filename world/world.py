from __future__ import annotations

"""World generation and management utilities."""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Iterable

from .generation import (
    perlin_noise,
    determine_biome,
    generate_elevation_map,
    generate_temperature_map,
    generate_rainfall,
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
    start: Coordinate
    end: Coordinate


@dataclass
class WorldSettings:
    seed: int = 0
    width: int = 50
    height: int = 50
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
    noise_scale: float = 0.1
    noise_octaves: int = 4


@dataclass
class Hex:
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
    return random.Random(settings.seed)


# --- Resource generation rules by terrain ---
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
    ],
    "mountains": [
        (ResourceType.STONE, 5, 15, 1.0),
        (ResourceType.ORE, 1, 5, 0.7),
        (ResourceType.IRON, 1, 3, 0.4),
        (ResourceType.GOLD, 1, 2, 0.2),
        (ResourceType.GEMS, 1, 2, 0.2),
        (ResourceType.CLAY, 1, 2, 0.1),
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
    ],
    "desert": [
        (ResourceType.STONE, 1, 3, 0.2),
        (ResourceType.ORE, 1, 2, 0.1),
        (ResourceType.GOLD, 1, 1, 0.05),
        (ResourceType.SPICE, 1, 2, 0.1),
        (ResourceType.CLAY, 1, 2, 0.05),
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
    ],
    "water": [
        (ResourceType.FISH, 1, 5, 0.5),
        (ResourceType.CRABS, 1, 3, 0.3),
        (ResourceType.PEARLS, 1, 1, 0.05),
    ],
}


def generate_resources(rng: random.Random, terrain: str) -> Dict[ResourceType, int]:
    return {
        r: rng.randint(lo, hi)
        for r, lo, hi, p in RESOURCE_RULES.get(terrain, [])
        if rng.random() < p
    }


class World:
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
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []
        self.rng = initialize_random(self.settings)

        self.elevation_map = generate_elevation_map(width, height, self.settings)
        self.temperature_map = generate_temperature_map(self.settings, self.rng)
        self.rainfall_map = generate_rainfall(self.elevation_map, self.settings, self.rng)

        self._generate_rivers()

    @property
    def width(self) -> int:
        return self.settings.width

    @property
    def height(self) -> int:
        return self.settings.height

    def _generate_hex(self, q: int, r: int) -> Hex:
        elev = self.elevation_map[r][q]
        temp = self.temperature_map[r][q]
        moist = self.rainfall_map[r][q]
        terrain = determine_biome(elev, temp, moist)
        rng = random.Random(hash((q, r, self.settings.seed)))
        resources = generate_resources(rng, terrain)

        return Hex(
            coord=(q, r),
            terrain=terrain,
            elevation=elev,
            temperature=temp,
            moisture=moist,
            resources=resources,
        )

    def _generate_chunk(self, cx: int, cy: int) -> None:
        chunk = []
        base_q, base_r = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        for r_off in range(min(self.CHUNK_SIZE, self.height - base_r)):
            row = []
            for q_off in range(min(self.CHUNK_SIZE, self.width - base_q)):
                q, r = base_q + q_off, base_r + r_off
                row.append(self._generate_hex(q, r))
            chunk.append(row)
        self.chunks[(cx, cy)] = chunk

    def get(self, q: int, r: int) -> Optional[Hex]:
        if not (0 <= q < self.width and 0 <= r < self.height):
            return None
        cx, cy = q // self.CHUNK_SIZE, r // self.CHUNK_SIZE
        if (cx, cy) not in self.chunks:
            self._generate_chunk(cx, cy)
        chunk = self.chunks.get((cx, cy))
        row_idx, col_idx = r % self.CHUNK_SIZE, q % self.CHUNK_SIZE
        if not chunk or row_idx >= len(chunk) or col_idx >= len(chunk[row_idx]):
            return None
        return chunk[row_idx][col_idx]

    def _neighbors(self, q: int, r: int) -> List[Coordinate]:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        return [(q + dq, r + dr) for dq, dr in directions if 0 <= q + dq < self.width and 0 <= r + dr < self.height]

    def _downhill_neighbor(self, q: int, r: int) -> Optional[Coordinate]:
        current = self.get(q, r)
        if not current:
            return None
        best = None
        best_elev = current.elevation
        for nq, nr in self._neighbors(q, r):
            neighbor = self.get(nq, nr)
            if neighbor and neighbor.elevation < best_elev:
                best_elev = neighbor.elevation
                best = (nq, nr)
        return best

    def _generate_rivers(self) -> None:
        seeds = max(1, int(self.settings.rainfall_intensity * 5))
        avg_elev = sum(sum(row) for row in self.elevation_map) / (self.width * self.height)
        threshold = max(self.settings.sea_level, avg_elev)

        for _ in range(seeds):
            for _ in range(100):
                q, r = self.rng.randint(0, self.width - 1), self.rng.randint(0, self.height - 1)
                h = self.get(q, r)
                if h and h.elevation > threshold:
                    break
            else:
                continue
            current = (q, r)
            visited = set()
            while current and current not in visited:
                visited.add(current)
                nxt = self._downhill_neighbor(*current)
                if not nxt or nxt == current:
                    self.lakes.append(current)
                    self.get(*current).lake = True
                    break
                self.rivers.append(RiverSegment(current, nxt))
                self.get(*current).river = True
                current = nxt

    def all_hexes(self) -> Iterable[Hex]:
        for r in range(self.height):
            for q in range(self.width):
                h = self.get(q, r)
                if h:
                    yield h

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        totals = {r: 0 for r in ResourceType}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                h = self.get(x + dx, y + dy)
                if h:
                    for rtype, amt in h.resources.items():
                        totals[rtype] += amt
        return {rtype: amt for rtype, amt in totals.items() if amt > 0}

    def has_road(self, start: Coordinate, end: Coordinate) -> bool:
        return any((r.start, r.end) == (start, end) or (r.start, r.end) == (end, start) for r in self.roads)

    def add_road(self, start: Coordinate, end: Coordinate) -> None:
        if start == end or not self.get(*start) or not self.get(*end):
            raise ValueError("Invalid road endpoints")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Coordinate, end: Coordinate) -> float:
        return 1.5 if self.has_road(start, end) else 1.0


def adjust_settings(settings: WorldSettings, **kwargs) -> None:
    for key, val in kwargs.items():
        if val is not None and hasattr(settings, key):
            setattr(settings, key, max(0.0, min(1.0, val)))


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
