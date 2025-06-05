from __future__ import annotations

"""World generation and management utilities."""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

from .generation import (
    perlin_noise,
    determine_biome,
    generate_elevation_map,
    generate_temperature_map,
    generate_rainfall,
)
from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate
from .settings import WorldSettings

@dataclass(frozen=True)
class Road:
    start: Coordinate
    end: Coordinate


@dataclass(frozen=True)
class RiverSegment:
    """A start/end pair describing a single river edge."""
    start: Coordinate
    end: Coordinate


def initialize_random(settings: WorldSettings) -> random.Random:
    return random.Random(settings.seed)


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
        elevation = self.elevation_map[r][q]
        temperature = self.temperature_map[r][q]
        moisture = self.rainfall_map[r][q]
        terrain = determine_biome(
            elevation,
            temperature,
            moisture,
            mountain_elev=self.settings.mountain_elev,
            hill_elev=self.settings.hill_elev,
            tundra_temp=self.settings.tundra_temp,
            desert_rain=self.settings.desert_rain,
        )
        rng = random.Random(hash((q, r, self.settings.seed)))
        resources = generate_resources(rng, terrain)

        return Hex(
            coord=(q, r),
            terrain=terrain,
            elevation=elevation,
            temperature=temperature,
            moisture=moisture,
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
            row and chunk.append(row)
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
                    cur_hex = self.get(*current)
                    if not cur_hex:
                        break
                    merged = False
                    for n in self._neighbors(*current):
                        nh = self.get(*n)
                        if nh and (nh.lake or nh.elevation <= self.settings.sea_level):
                            self.rivers.append(RiverSegment(current, n))
                            cur_hex.river = True
                            if nh.elevation <= self.settings.sea_level:
                                nh.river = True
                            merged = True
                            break
                    if not merged:
                        self.lakes.append(current)
                        cur_hex.lake = True
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
