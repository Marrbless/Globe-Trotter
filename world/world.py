from __future__ import annotations

"""World generation and management utilities."""

import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

from .generation import (
    perlin_noise,
    determine_biome,
)
from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate
from .settings import WorldSettings
from .fantasy import apply_fantasy_overlays

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

    def _init_plates(self) -> List[Tuple[int, int, float]]:
        plates = max(2, int(3 + self.settings.plate_activity * 5))
        rng = random.Random(self.settings.seed)
        return [
            (
                rng.randint(0, self.settings.width - 1),
                rng.randint(0, self.settings.height - 1),
                rng.random(),
            )
            for _ in range(plates)
        ]

    def _plate_height(self, q: int, r: int) -> float:
        dists = sorted(
            (
                (cx - q) ** 2 + (cy - r) ** 2,
                base,
            )
            for cx, cy, base in self._plate_centers
        )
        dist0, base = dists[0]
        dist1 = dists[1][0] if len(dists) > 1 else dist0
        ratio = dist0 / (dist0 + dist1) if dist1 > 0 else 0.0
        boundary = 1.0 - abs(0.5 - ratio) * 2.0
        return base * self.settings.base_height + boundary * self.settings.plate_activity

    def _noise_value(self, q: int, r: int, seed_offset: int, setting: float) -> float:
        n = perlin_noise(q, r, self.settings.seed + seed_offset)
        amp = 0.5 + setting / 2
        offset = setting - 0.5
        return max(0.0, min(1.0, n * amp + offset))

    def _elevation(self, q: int, r: int) -> float:
        base = self._noise_value(q, r, 0, self.settings.elevation)
        plate = self._plate_height(q, r)
        return max(0.0, min(1.0, (base + plate) / 2))

    def _temperature(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        lat = r / float(self.settings.height - 1) if self.settings.height > 1 else 0.5
        base = 1.0 - abs(lat - 0.5) * 2
        base -= elevation * 0.3
        rng = random.Random(hash((r, self.settings.seed)))
        variation = rng.uniform(-0.1, 0.1) * self.settings.temperature
        wind_effect = ((q / float(self.settings.width - 1) if self.settings.width > 1 else 0.5) - 0.5) * self.settings.wind_strength * 0.2
        seasonal = math.sin(2 * math.pi * season) * self.settings.seasonal_amplitude * 0.5
        return max(0.0, min(1.0, base + variation + wind_effect + seasonal))

    def _moisture(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        rng = random.Random(hash((r, self.settings.seed, "rain")))
        base = self.settings.moisture + rng.uniform(-0.1, 0.1)
        base += math.sin(2 * math.pi * season) * self.settings.seasonal_amplitude * 0.5
        moisture = max(0.0, min(1.0, base))
        precip = 0.0
        for x in range(q + 1):
            elev = self._elevation(x, r)
            precip = max(0.0, moisture * (1.0 - elev))
            if x == q:
                break
            loss = (precip * 0.5 + elev * 0.1) * (1.0 - self.settings.wind_strength)
            moisture = max(0.0, moisture - loss)
        return max(0.0, min(1.0, precip))

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
        self.season = 0.0

        self._plate_centers = self._init_plates()
        self._generate_rivers()
        if self.settings.fantasy_level > 0:
            apply_fantasy_overlays(self.all_hexes(), self.settings.fantasy_level)

    @property
    def width(self) -> int:
        return self.settings.width

    @property
    def height(self) -> int:
        return self.settings.height

    def _generate_hex(self, q: int, r: int) -> Hex:
        elevation = self._elevation(q, r)
        temperature = self._temperature(q, r, elevation, self.season)
        moisture = self._moisture(q, r, elevation, self.season)
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
        chunk: List[List[Hex]] = []
        base_q, base_r = cx * self.CHUNK_SIZE, cy * self.CHUNK_SIZE
        rows = (
            self.CHUNK_SIZE
            if self.settings.infinite
            else min(self.CHUNK_SIZE, self.height - base_r)
        )
        cols = (
            self.CHUNK_SIZE
            if self.settings.infinite
            else min(self.CHUNK_SIZE, self.width - base_q)
        )

        for r_off in range(rows):
            row: List[Hex] = []
            for q_off in range(cols):
                q, r = base_q + q_off, base_r + r_off
                row.append(self._generate_hex(q, r))
            row and chunk.append(row)
        self.chunks[(cx, cy)] = chunk

    def get(self, q: int, r: int) -> Optional[Hex]:
        if not self.settings.infinite and not (0 <= q < self.width and 0 <= r < self.height):
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
        if self.settings.infinite:
            return [(q + dq, r + dr) for dq, dr in directions]
        return [
            (q + dq, r + dr)
            for dq, dr in directions
            if 0 <= q + dq < self.width and 0 <= r + dr < self.height
        ]

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
        rainfall: Dict[Coordinate, float] = {}
        flow: Dict[Coordinate, float] = {}
        downhill: Dict[Coordinate, Optional[Coordinate]] = {}

        for r in range(self.height):
            for q in range(self.width):
                hex_ = self.get(q, r)
                rain = hex_.moisture * self.settings.rainfall_intensity
                rainfall[(q, r)] = rain
                flow[(q, r)] = rain
                dn = self._downhill_neighbor(q, r)
                if dn and self.get(*dn).elevation < hex_.elevation:
                    downhill[(q, r)] = dn
                else:
                    downhill[(q, r)] = None

        coords = sorted(flow.keys(), key=lambda c: self.get(*c).elevation, reverse=True)
        for c in coords:
            d = downhill[c]
            if d:
                flow[d] += flow[c]

        for c, f in flow.items():
            self.get(*c).water_flow = f

        avg_flow = sum(flow.values()) / len(flow) if flow else 0.0
        river_threshold = max(0.05 * self.settings.rainfall_intensity, avg_flow * 2)
        lake_threshold = max(0.1 * self.settings.rainfall_intensity, avg_flow * 4)

        for c in coords:
            d = downhill[c]
            hex_c = self.get(*c)
            if d:
                if flow[c] >= river_threshold:
                    self.rivers.append(RiverSegment(c, d))
                    hex_c.river = True
                    self.get(*d).river = True
            else:
                if flow[c] > lake_threshold:
                    if not hex_c.lake:
                        self.lakes.append(c)
                        hex_c.lake = True

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
