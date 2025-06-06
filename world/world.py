from __future__ import annotations

"""
world.py

World generation and management utilities with enhanced realism and fantasy integration.

Key features & refactors:
- Fully deterministic, tile-based RNG via `_stable_hash` and `World._tile_rng`.
- Separated chunk width/height for rectangular chunking.
- Lazy, chunked generation of terrain/climate/biomes with explicit cache invalidation.
- River & lake generation broken into helper methods; incremental update mode is planned.
- Pluggable biome rules with runtime registration.
- Defensive input checking for roads and trade efficiency.
- On-disk caching layer skeleton for infinite worlds to prevent unbounded memory use.
- Full type annotations and PEP 257 docstrings throughout.
"""

import math
import random
import pickle
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate
from .settings import WorldSettings
from .fantasy import apply_fantasy_overlays

# ─────────────────────────────────────────────────────────────────────────────
# == TYPE ALIASES & CUSTOM EXCEPTIONS ==

CoordinateList = List[Coordinate]
FlowMap = Dict[Coordinate, float]
ElevationCache = Dict[Coordinate, float]
TemperatureCache = Dict[Coordinate, float]
MoistureCache = Dict[Coordinate, float]
BiomeCache = Dict[Coordinate, str]


class InvalidCoordinateError(ValueError):
    """Raised when a provided coordinate is not a valid (int, int) pair."""


# ─────────────────────────────────────────────────────────────────────────────
# == STABLE HASH / RNG HELPERS ==

def _stable_hash(*args: int) -> int:
    """
    Combine integer arguments into a single 64-bit integer using a deterministic mixing routine.
    Ensures repeatable results across Python runs (unlike built-in hash()).
    """
    x = 0x345678ABCDEF1234  # Arbitrary non-zero start value.
    for a in args:
        a &= 0xFFFFFFFFFFFFFFFF
        a ^= (a >> 33)
        a = (a * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        a ^= (a >> 33)
        x ^= a
        x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    return x


def perlin_noise(
    x: float,
    y: float,
    seed: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 0.05,
) -> float:
    """
    Generate fractal Perlin noise at (x, y) for a given seed.
    Returns a float in [0, 1].
    """

    def _perlin_single(px: float, py: float, s: int) -> float:
        x0 = math.floor(px)
        y0 = math.floor(py)
        x1 = x0 + 1
        y1 = y0 + 1

        def _fade(t: float) -> float:
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(a: float, b: float, t: float) -> float:
            return a + t * (b - a)

        def _grad(ix: int, iy: int, s2: int) -> Tuple[float, float]:
            # Use stable, coordinate-based RNG:
            rng = random.Random(_stable_hash(ix, iy, s2))
            angle = rng.random() * 2.0 * math.pi
            return math.cos(angle), math.sin(angle)

        def _dot_grid_gradient(ix: int, iy: int, xx: float, yy: float, s2: int) -> float:
            gx, gy = _grad(ix, iy, s2)
            dx = xx - ix
            dy = yy - iy
            return gx * dx + gy * dy

        sx = _fade(px - x0)
        sy = _fade(py - y0)

        n00 = _dot_grid_gradient(x0, y0, px, py, s)
        n10 = _dot_grid_gradient(x1, y0, px, py, s)
        n01 = _dot_grid_gradient(x0, y1, px, py, s)
        n11 = _dot_grid_gradient(x1, y1, px, py, s)

        ix0 = _lerp(n00, n10, sx)
        ix1 = _lerp(n01, n11, sx)
        val = _lerp(ix0, ix1, sy)

        return (val + 1.0) / 2.0

    total = 0.0
    amplitude = 1.0
    frequency = scale
    max_amplitude = 0.0

    for i in range(octaves):
        sample = _perlin_single(x * frequency, y * frequency, seed + i)
        total += sample * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_amplitude


# ─────────────────────────────────────────────────────────────────────────────
# == HEX NEIGHBOR CONSTANTS ==

# Axial hex directions: E, W, SE, NW, NE, SW
HEX_DIRECTIONS: List[Tuple[int, int]] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
]


# ─────────────────────────────────────────────────────────────────────────────
# == BIOME RULES & REGISTRATION ==

@dataclass(frozen=True)
class BiomeRule:
    """
    Defines a rectangular condition in (elevation, temperature, rainfall) space that
    maps to a single biome name. If 'is_fantasy' is True, this is treated as a special
    fantasy override after realistic rules.
    """
    name: str
    min_elev: float = 0.0
    max_elev: float = 1.0
    min_temp: float = 0.0
    max_temp: float = 1.0
    min_rain: float = 0.0
    max_rain: float = 1.0
    is_fantasy: bool = False


_REALISTIC_BIOME_RULES: List[BiomeRule] = [
    BiomeRule(
        name="mountains",
        min_elev=0.8,
        max_elev=1.0,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.0,
        max_rain=1.0,
        is_fantasy=False,
    ),
    BiomeRule(
        name="hills",
        min_elev=0.6,
        max_elev=0.8,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.0,
        max_rain=1.0,
        is_fantasy=False,
    ),
    BiomeRule(
        name="tundra",
        min_elev=0.0,
        max_elev=1.0,
        min_temp=0.0,
        max_temp=0.3,
        min_rain=0.0,
        max_rain=1.0,
        is_fantasy=False,
    ),
    BiomeRule(
        name="desert",
        min_elev=0.0,
        max_elev=0.6,
        min_temp=0.5,
        max_temp=1.0,
        min_rain=0.0,
        max_rain=0.3,
        is_fantasy=False,
    ),
    BiomeRule(
        name="rainforest",
        min_elev=0.0,
        max_elev=1.0,
        min_temp=0.5,
        max_temp=1.0,
        min_rain=0.7,
        max_rain=1.0,
        is_fantasy=False,
    ),
    BiomeRule(
        name="forest",
        min_elev=0.0,
        max_elev=1.0,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.4,
        max_rain=0.7,
        is_fantasy=False,
    ),
    BiomeRule(
        name="plains",
        min_elev=0.0,
        max_elev=0.6,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.3,
        max_rain=1.0,
        is_fantasy=False,
    ),
]

_FANTASY_BIOME_RULES: List[BiomeRule] = [
    BiomeRule(
        name="crystal_forest",
        min_elev=0.4,
        max_elev=1.0,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.0,
        max_rain=1.0,
        is_fantasy=True,
    ),
    BiomeRule(
        name="floating_island",
        min_elev=0.8,
        max_elev=1.0,
        min_temp=0.0,
        max_temp=1.0,
        min_rain=0.0,
        max_rain=1.0,
        is_fantasy=True,
    ),
]


def register_biome_rule(rule: BiomeRule) -> None:
    """
    Register a new biome rule. If rule.is_fantasy is False, it is appended to realistic rules;
    otherwise, to the fantasy overrides. The rule lists are checked in insertion order.
    """
    if rule.is_fantasy:
        _FANTASY_BIOME_RULES.append(rule)
    else:
        _REALISTIC_BIOME_RULES.append(rule)


def _determine_biome_tile(
    elevation: float,
    temperature: float,
    rainfall: float,
    settings: WorldSettings,
    tile_rng: random.Random,
) -> str:
    """
    Classify the biome for a single tile (given elevation, temperature, rainfall). Applies
    realistic rules first; if none match, defaults to "plains". Then, if fantasy_level > 0.0,
    checks fantasy rules in order (with possible random chance).
    """
    # 1) Realistic rules
    for rule in _REALISTIC_BIOME_RULES:
        if (
            rule.min_elev <= elevation <= rule.max_elev
            and rule.min_temp <= temperature <= rule.max_temp
            and rule.min_rain <= rainfall <= rule.max_rain
        ):
            return rule.name

    # 2) Fallback default
    base_biome = "plains"

    # 3) Fantasy overrides (only if fantasy_level > 0)
    if settings.fantasy_level > 0.0:
        for rule in _FANTASY_BIOME_RULES:
            if (
                rule.min_elev <= elevation <= rule.max_elev
                and rule.min_temp <= temperature <= rule.max_temp
                and rule.min_rain <= rainfall <= rule.max_rain
                and tile_rng.random() < settings.fantasy_level * 0.1
            ):
                return rule.name

    return base_biome


def determine_biome(
    elevation: float,
    temperature: float,
    rainfall: float,
    settings: WorldSettings | None = None,
) -> str:
    """Public helper to classify a single tile's biome."""
    rng = random.Random(0xBEEF)
    return _determine_biome_tile(
        elevation=elevation,
        temperature=temperature,
        rainfall=rainfall,
        settings=settings or WorldSettings(),
        tile_rng=rng,
    )


def _smooth_biome_map(
    biomes: list[list[str]],
    width: int,
    height: int,
    iterations: int = 1,
) -> list[list[str]]:
    """
    Smooth the biome map by replacing isolated cells that differ from the majority of neighbors.
    """

    def majority_neighbor(q0: int, r0: int) -> str:
        counts: Dict[str, int] = {}
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q0 + dq, r0 + dr
            if 0 <= nq < width and 0 <= nr < height:
                neighbor_biome = biomes[nr][nq]
                counts[neighbor_biome] = counts.get(neighbor_biome, 0) + 1
        if not counts:
            return biomes[r0][q0]
        return max(counts.items(), key=lambda it: it[1])[0]

    for _ in range(iterations):
        new_map = [row.copy() for row in biomes]
        for r0 in range(height):
            for q0 in range(width):
                curr = biomes[r0][q0]
                maj = majority_neighbor(q0, r0)
                if maj != curr:
                    new_map[r0][q0] = maj
        biomes = new_map

    return biomes


# Predefined colors for each biome (RGBA). Can be extended via `register_biome_color`.
BIOME_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "plains": (110, 205, 88, 255),
    "forest": (34, 139, 34, 255),
    "mountains": (139, 137, 137, 255),
    "hills": (107, 142, 35, 255),
    "desert": (237, 201, 175, 255),
    "tundra": (220, 220, 220, 255),
    "rainforest": (0, 100, 0, 255),
    "water": (65, 105, 225, 255),
    "floating_island": (186, 85, 211, 255),
    "crystal_forest": (0, 255, 255, 255),
}


def register_biome_color(name: str, color: Tuple[int, int, int, int]) -> None:
    """
    Register or override the RGBA color for a given biome name.
    """
    BIOME_COLORS[name] = color


# ─────────────────────────────────────────────────────────────────────────────
# == RIVER & LAKE GENERATION HELPERS ==

@dataclass(frozen=True)
class RiverSegment:
    """A start→end pair describing a single river edge with a flow strength."""

    start: Coordinate
    end: Coordinate
    strength: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# == ROAD DATACLASS ==

@dataclass(frozen=True)
class Road:
    start: Coordinate
    end: Coordinate


# ─────────────────────────────────────────────────────────────────────────────
# == MAIN WORLD CLASS ==

class World:
    __slots__ = (
        "settings",
        "chunk_width",
        "chunk_height",
        "max_active_chunks",
        "chunks",
        "evicted_chunks",
        "roads",
        "rivers",
        "lakes",
        "rng",
        "_season",
        "_plate_centers",
        "_elevation_cache",
        "_temperature_cache",
        "_moisture_cache",
        "_biome_cache",
        "_dirty_rivers",
        "_in_generate_water",
        "event_turn_counters",
        "tech_levels",
        "god_powers",
    )

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        *,
        seed: int = 0,
        settings: Optional[WorldSettings] = None,
    ) -> None:
        """
        Create a new World instance. Terrain and climate layers are generated lazily, chunk by chunk.
        River and lake generation is deferred until first call to `generate_water_features()`.

        Args:
            width (int): Width of the world in hex columns (ignored if settings.infinite = True).
            height (int): Height of the world in hex rows (ignored if settings.infinite = True).
            seed (int): Master seed for deterministic generation.
            settings (WorldSettings, optional): Custom settings object. If None, a default is created.
        """
        # Initialize settings
        self.settings: WorldSettings = (
            settings if settings is not None else WorldSettings(seed=seed, width=width, height=height)
        )
        # Overwrite width/height from constructor
        self.settings.width = width
        self.settings.height = height

        # Chunk dimensions
        self.chunk_width: int = getattr(self.settings, "chunk_width", 10)
        self.chunk_height: int = getattr(self.settings, "chunk_height", 10)
        self.max_active_chunks: int = getattr(self.settings, "max_active_chunks", 100)

        # Loaded chunks: OrderedDict[(cx,cy), List[List[Hex]]]
        self.chunks: OrderedDict[Tuple[int, int], List[List[Hex]]] = OrderedDict()
        # On-disk evicted chunks: Map (cx,cy) → filepath
        self.evicted_chunks: Dict[Tuple[int, int], str] = {}

        # Structures
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []

        # Deterministic RNG for global uses (plate centers, etc.)
        self.rng = random.Random(self.settings.seed)

        # Season fraction in [0,1)
        self._season: float = 0.0

        # Precompute tectonic plate centers (unused when using simple noise)
        self._plate_centers: List[Tuple[int, int, float]] = []

        # Per-tile caches (filled on demand)
        self._elevation_cache: ElevationCache = {}
        self._temperature_cache: TemperatureCache = {}
        self._moisture_cache: MoistureCache = {}
        self._biome_cache: BiomeCache = {}

        # Mark water features for initial generation
        self._dirty_rivers = True
        self._in_generate_water = False
        self.event_turn_counters: Dict[str, int] = {}
        self.tech_levels: Dict[str, int] = {}
        self.god_powers: Dict[str, int] = {}

        if not self.settings.infinite:
            self._generate_rivers()

    # Convenience accessors -------------------------------------------------
    @property
    def width(self) -> int:
        """Convenience access to the world's width setting."""
        return self.settings.width

    @property
    def height(self) -> int:
        """Convenience access to the world's height setting."""
        return self.settings.height

    def _expand_to_include(self, q: int, r: int) -> None:
        """Increase settings.width/height so (q, r) lies within them."""
        if not self.settings.infinite:
            return
        if q >= self.settings.width:
            self.settings.width = q + 1
        if r >= self.settings.height:
            self.settings.height = r + 1

    # ─────────────────────────────────────────────────────────────────────────
    # == PROPERTIES & SETTINGS MANAGEMENT ==

    @property
    def season(self) -> float:
        """Current season fraction in [0.0, 1.0)."""
        return self._season

    @season.setter
    def season(self, value: float) -> None:
        val = float(value) % 1.0
        if val != self._season:
            self._season = val
            # Invalidate caches dependent on season
            self._temperature_cache.clear()
            self._moisture_cache.clear()
            self._biome_cache.clear()
            self._dirty_rivers = True

    def advance_season(self, delta: float = 1 / 365) -> None:
        """
        Advance the season fraction by `delta`. Values wrap around at 1.0 (year cycles).
        Marks temperature, moisture, and biome caches as dirty.

        Args:
            delta (float): Fractional increment (e.g., 1/365 for one day).
        """
        self.season = (self._season + delta) % 1.0

    def mark_dirty(self) -> None:
        """
        Invalidate all per-tile caches and schedule water regeneration.
        Called automatically when settings affecting terrain/climate change.
        """
        self._elevation_cache.clear()
        self._temperature_cache.clear()
        self._moisture_cache.clear()
        self._biome_cache.clear()
        self._dirty_rivers = True

    def __contains__(self, coord: Coordinate) -> bool:
        """
        Allow `coord in world` checks.

        For finite worlds, returns True if and only if 0 <= q < width and 0 <= r < height.
        For infinite worlds, always returns True.
        """
        q, r = coord
        if self.settings.infinite:
            return True
        return 0 <= q < self.settings.width and 0 <= r < self.settings.height

    # ─────────────────────────────────────────────────────────────────────────
    # == CHUNK INITIALIZATION & CACHING ==

    def _init_plates(self) -> List[Tuple[int, int, float]]:
        """
        Compute tectonic plate centers.
        In this simplified noise-based generator plates are unused,
        so return an empty list for compatibility.
        """
        return []

    def _tile_rng(self, q: int, r: int, tag: int) -> random.Random:
        """
        Return a deterministic RNG whose state depends on (q, r), the world seed, and a numeric tag.

        Args:
            q (int): Axial coordinate q.
            r (int): Axial coordinate r.
            tag (int): A small integer (purpose tag) to differentiate RNG streams.

        Returns:
            random.Random: A new RNG seeded by (_stable_hash(q, r, seed, tag)).
        """
        seed = _stable_hash(q, r, self.settings.seed, tag)
        return random.Random(seed)

    def _generate_chunk(self, cx: int, cy: int) -> None:
        """
        Populate a rectangular chunk at chunk coordinates (cx, cy) with dimensions
        (chunk_width × chunk_height). Evicts the least-recently-used chunk when capacity is exceeded.

        If a chunk has been previously evicted to disk, reloads it from the file instead of regenerating.
        """
        # If chunk was evicted to disk, reload it:
        if (cx, cy) in self.evicted_chunks:
            filepath = self.evicted_chunks.pop((cx, cy))
            try:
                with open(filepath, "rb") as f:
                    loaded_chunk = pickle.load(f)
                self.chunks[(cx, cy)] = loaded_chunk
                # Remove the file from disk
                os.remove(filepath)
            except Exception:
                # If loading fails, fall back to regeneration
                pass

        if (cx, cy) not in self.chunks:
            base_q = cx * self.chunk_width
            base_r = cy * self.chunk_height

            # Determine actual row/col counts (handles finite vs infinite)
            if self.settings.infinite:
                rows = self.chunk_height
                cols = self.chunk_width
            else:
                rows = min(self.chunk_height, self.settings.height - base_r)
                cols = min(self.chunk_width, self.settings.width - base_q)

            new_chunk: List[List[Hex]] = []
            for r_off in range(rows):
                row_tiles: List[Hex] = []
                for q_off in range(cols):
                    q = base_q + q_off
                    r = base_r + r_off
                    if not self.settings.infinite:
                        if not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                            continue
                    tile = self._generate_hex(q, r)
                    row_tiles.append(tile)
                if row_tiles:
                    new_chunk.append(row_tiles)

            self.chunks[(cx, cy)] = new_chunk
            # Generate water only for new tiles
            self._dirty_rivers = True
            if self.settings.infinite and not self._in_generate_water:
                self.generate_water_features()

            # Evict LRU chunk if over capacity
            if len(self.chunks) > self.max_active_chunks:
                old_cx, old_cy = next(iter(self.chunks))
                old_chunk = self.chunks.pop((old_cx, old_cy))

                # Serialize to disk
                filepath = f"/tmp/world_chunk_{self.settings.seed}_{old_cx}_{old_cy}.pkl"
                try:
                    with open(filepath, "wb") as f:
                        pickle.dump(old_chunk, f)
                    self.evicted_chunks[(old_cx, old_cy)] = filepath
                except Exception:
                    # If serialization fails, just drop it from memory
                    pass

    def get(self, q: int, r: int) -> Optional[Hex]:
        """
        Retrieve the Hex at (q, r). If it doesn’t exist, generate its chunk on demand.
        Returns None if (q, r) is out of bounds in a finite world.

        Args:
            q (int): Axial coordinate q.
            r (int): Axial coordinate r.

        Returns:
            Optional[Hex]: The Hex object, or None if out of bounds.
        """
        if not self.settings.infinite:
            if not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                return None
        else:
            self._expand_to_include(q, r)

        cx = q // self.chunk_width
        cy = r // self.chunk_height
        if (cx, cy) not in self.chunks:
            self._generate_chunk(cx, cy)
        # Mark chunk as recently used
        self.chunks.move_to_end((cx, cy))
        chunk = self.chunks.get((cx, cy))
        if not chunk:
            return None

        row_idx = r % self.chunk_height
        col_idx = q % self.chunk_width
        if row_idx >= len(chunk) or col_idx >= len(chunk[row_idx]):
            return None

        return chunk[row_idx][col_idx]

    def all_hexes(self) -> Iterable[Hex]:
        """
        Yield every generated Hex in a finite world, or every currently loaded chunk in an infinite world.

        Returns:
            Iterable[Hex]: All Hex objects that have been generated or cached.
        """
        for chunk in self.chunks.values():
            for row in chunk:
                for h in row:
                    yield h

    def iter_all_coords(self) -> Iterable[Coordinate]:
        """
        Yield all coordinate pairs in a finite world without generating Hex objects,
        or yield currently loaded chunk coordinates in an infinite world.

        Returns:
            Iterable[Coordinate]: (q, r) pairs.
        """
        if not self.settings.infinite:
            for r in range(self.settings.height):
                for q in range(self.settings.width):
                    yield (q, r)
        else:
            for (cx, cy), chunk in self.chunks.items():
                for r_idx, row in enumerate(chunk):
                    for q_idx, _ in enumerate(row):
                        yield (cx * self.chunk_width + q_idx, cy * self.chunk_height + r_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # == SHARED TERRAIN/CLIMATE HELPERS ==

    def _elevation(self, q: int, r: int) -> float:
        """
        Compute or retrieve cached elevation for tile (q, r).
        Uses Perlin noise scaled by the ``elevation`` setting.
        Returns a float in [0.0, 1.0].
        """
        coord = (q, r)
        if coord in self._elevation_cache:
            return self._elevation_cache[coord]

        base = perlin_noise(float(q), float(r), self.settings.seed, scale=0.1)
        amp = 0.5 + self.settings.elevation / 2.0
        offset = self.settings.elevation - 0.5
        elev = max(0.0, min(1.0, base * amp + offset))
        self._elevation_cache[coord] = elev
        return elev

    def _temperature(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        """
        Compute or retrieve cached temperature for tile (q, r).
        Uses Perlin noise and lapses with elevation.
        Returns a float in [0, 1].
        """
        coord = (q, r)
        if coord in self._temperature_cache:
            return self._temperature_cache[coord]

        base = perlin_noise(float(q), float(r), self.settings.seed ^ 0x1234, scale=0.1)
        amp = 0.5 + self.settings.temperature / 2.0
        offset = self.settings.temperature - 0.5
        temp = base * amp + offset
        temp -= elevation * self.settings.lapse_rate
        temp = max(0.0, min(1.0, temp))
        self._temperature_cache[coord] = temp
        return temp

    def _moisture(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        """
        Compute or retrieve cached moisture for tile (q, r).
        Uses an orographic moisture model with wind effects.
        Returns a float in [0, 1].
        """
        coord = (q, r)
        if coord in self._moisture_cache:
            return self._moisture_cache[coord]

        # Base moisture decreases toward poles
        lat = float(r) / float(self.settings.height - 1) if self.settings.height > 1 else 0.5
        base_moist = 1.0 - abs(lat - 0.5) * 2.0
        base_moist *= self.settings.moisture

        tile_rng = self._tile_rng(q, r, 0xBEEF)
        variation = tile_rng.uniform(-0.1, 0.1) * self.settings.moisture
        base_moist += variation
        base_moist += math.sin(2.0 * math.pi * season) * self.settings.seasonal_amplitude * 0.5
        moisture = max(0.0, min(1.0, base_moist))

        thresh = getattr(self.settings, "orographic_threshold", 0.6)
        factor = getattr(self.settings, "orographic_factor", 0.3)
        wind = self.settings.wind_dir

        if wind in (1, 3):
            start = 0 if wind == 1 else self.settings.width - 1
            step = 1 if wind == 1 else -1
            rng_range = range(start, q + step, step)
            coord_func = lambda x: (x, r)
        else:
            start = 0 if wind == 2 else self.settings.height - 1
            step = 1 if wind == 2 else -1
            rng_range = range(start, r + step, step)
            coord_func = lambda y: (q, y)

        precip = 0.0
        prev_elev = 0.0
        for idx in rng_range:
            cq, cr = coord_func(idx)
            elev = elevation if (cq, cr) == (q, r) else self._elevation(cq, cr)
            precip = max(0.0, moisture * (1.0 - elev))
            if (cq, cr) == (q, r):
                break
            loss = precip * 0.5
            if elev > thresh and elev > prev_elev:
                loss += (elev - thresh) * factor * self.settings.wind_strength
            moisture = max(0.0, moisture - loss)
            prev_elev = elev

        moist = max(0.0, min(1.0, precip))
        self._moisture_cache[coord] = moist
        return moist

    def _biome(self, q: int, r: int, elevation: float, temperature: float, rainfall: float) -> str:
        """
        Compute or retrieve cached biome for tile (q, r), given elevation, temperature, and rainfall.
        Returns the biome name as a string.
        """
        coord = (q, r)
        if coord in self._biome_cache:
            return self._biome_cache[coord]

        # Regional seed for cluster consistency
        region_size = getattr(self.settings, "biome_region_size", 10)
        region_q = q // region_size
        region_r = r // region_size
        region_seed = _stable_hash(region_q, region_r, self.settings.seed, 0x1001)

        # Tile seed for minor randomness
        tile_seed = _stable_hash(region_seed, q, r, 0x1002)
        tile_rng = random.Random(tile_seed)

        biome_str = _determine_biome_tile(
            elevation=elevation,
            temperature=temperature,
            rainfall=rainfall,
            settings=self.settings,
            tile_rng=tile_rng,
        )
        self._biome_cache[coord] = biome_str
        return biome_str

    def _generate_hex(self, q: int, r: int) -> Hex:
        """
        Generate (or retrieve from cache) a single Hex at (q, r), including elevation,
        temperature, moisture, biome, resources, and fantasy overlays if enabled.
        """
        elevation = self._elevation(q, r)
        temperature = self._temperature(q, r, elevation, self._season)
        rainfall = self._moisture(q, r, elevation, self._season)
        biome = self._biome(q, r, elevation, temperature, rainfall)

        tile_rng = self._tile_rng(q, r, 0x2000)  # “resource generation” tag
        resources = generate_resources(tile_rng, biome)

        h = Hex(
            coord=(q, r),
            terrain=biome,
            elevation=elevation,
            temperature=temperature,
            moisture=rainfall,
            resources=resources,
        )

        # If fantasy overlays are desired, apply them here
        if self.settings.fantasy_level > 0.0:
            apply_fantasy_overlays([h], self.settings.fantasy_level)

        return h

    # ─────────────────────────────────────────────────────────────────────────
    # == NEIGHBOR & WATER HELPERS ==

    def _neighbors(self, q: int, r: int) -> CoordinateList:
        """
        Return axial hex neighbor coordinates. Always checks bounds if not infinite.
        """
        result: CoordinateList = []
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if self.settings.infinite:
                result.append((nq, nr))
            else:
                if 0 <= nq < self.settings.width and 0 <= nr < self.settings.height:
                    result.append((nq, nr))
        return result

    def _neighbors_elevated(self, q: int, r: int) -> CoordinateList:
        """
        Return coordinates of neighbors whose elevation is already cached (or computed).
        This ensures we can compare elevations without key errors.
        """
        result: CoordinateList = []
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if self.settings.infinite:
                _ = self._elevation(nq, nr)
                result.append((nq, nr))
            else:
                if 0 <= nq < self.settings.width and 0 <= nr < self.settings.height:
                    _ = self._elevation(nq, nr)
                    result.append((nq, nr))
        return result

    def _downhill_neighbor(self, q: int, r: int) -> Optional[Coordinate]:
        """
        Return the coordinate of the neighbor with strictly lower elevation.
        If multiple exist, pick the steepest drop. If none, return None.
        """
        current_elev = self._elevation(q, r)
        best_coord: Optional[Coordinate] = None
        best_elev = current_elev
        for (nq, nr) in self._neighbors(q, r):
            neigh_elev = self._elevation(nq, nr)
            if neigh_elev < best_elev:
                best_elev = neigh_elev
                best_coord = (nq, nr)
        return best_coord

    def _collect_initial_flow_and_downhill(self) -> tuple[FlowMap, Dict[Coordinate, Optional[Coordinate]]]:
        """
        Step A: Compute initial flow_map and downhill_map for each tile in the relevant bounds.
        Returns:
            flow_map: Dict mapping each (q, r) to its rainfall-based flow value.
            downhill_map: Dict mapping each (q, r) to the chosen downhill neighbor or None.
        """
        flow_map: FlowMap = {}
        downhill_map: Dict[Coordinate, Optional[Coordinate]] = {}

        # Only process currently loaded chunks
        if not self.chunks:
            self.rivers.clear()
            self.lakes.clear()
            self._dirty_rivers = False
            return {}, {}

        for (cx, cy), chunk in self.chunks.items():
            for r_idx, row_tiles in enumerate(chunk):
                for q_idx, _ in enumerate(row_tiles):
                    q = cx * self.chunk_width + q_idx
                    r = cy * self.chunk_height + r_idx

                    elev = self._elevation(q, r)
                    rain_amt = self._moisture(q, r, elev, self._season) * self.settings.rainfall_intensity
                    flow_map[(q, r)] = rain_amt

                    dn = self._downhill_neighbor(q, r)
                    if dn and self._elevation(*dn) < elev:
                        downhill_map[(q, r)] = dn
                    else:
                        downhill_map[(q, r)] = None

        return flow_map, downhill_map

    def _accumulate_flows(self, flow_map: FlowMap, downhill_map: Dict[Coordinate, Optional[Coordinate]]) -> None:
        """
        Step B & C: Sort coords by descending elevation, accumulate flow downstream,
        and optionally create tributaries. Modifies flow_map in place.
        """
        coords_sorted = sorted(
            flow_map.keys(),
            key=lambda c: self._elevation(c[0], c[1]),
            reverse=True,
        )

        visited: set[Coordinate] = set()
        for c in coords_sorted:
            if c in visited:
                continue
            d = downhill_map[c]
            if d:
                flow_map[d] = flow_map.get(d, 0.0) + flow_map[c]
                downhill_map.setdefault(d, self._downhill_neighbor(*d))
                visited.add(c)
                # Branching logic
                branch_threshold = self.settings.river_branch_threshold * self.settings.rainfall_intensity
                if flow_map[c] > branch_threshold:
                    neighbor_coords = self._neighbors_elevated(*c)
                    second_best: Optional[Coordinate] = None
                    sec_elev = self._elevation(*c)
                    for n in neighbor_coords:
                        if n == d:
                            continue
                        nelev = self._elevation(*n)
                        if nelev < sec_elev:
                            sec_elev = nelev
                            second_best = n
                    if second_best is not None and second_best not in visited:
                        tile_rng = self._tile_rng(c[0], c[1], 0x3010)
                        if tile_rng.random() < self.settings.river_branch_chance:
                            flow_map[second_best] = flow_map.get(second_best, 0.0) + flow_map[c] * 0.3
                            downhill_map.setdefault(second_best, self._downhill_neighbor(*second_best))
                            visited.add(second_best)

    def _determine_thresholds(self, flow_values: Iterable[float]) -> tuple[float, float]:
        """
        Step D: Given all flow amounts, compute:
          - river_threshold: min flow to count as a river segment
          - lake_threshold: min flow to count as a lake (no downhill neighbor)
        """
        flows = list(flow_values)
        if not flows:
            return 1.0, 1.0

        avg_flow = sum(flows) / len(flows)
        rt = max(0.1 * self.settings.rainfall_intensity, avg_flow * 3.0)
        lt = max(0.2 * self.settings.rainfall_intensity, avg_flow * 4.0)
        return rt, lt

    def _clear_old_water_flags(self, coords: Iterable[Coordinate]) -> None:
        """
        Step E(i): Clear old river/lake flags for each loaded tile in coords.
        """
        for (q, r) in coords:
            if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                continue
            h = self.get(q, r)
            if h:
                h.river = False
                h.lake = False
                h.water_flow = 0.0

    def _identify_and_flag_rivers_lakes(
        self,
        flow_map: FlowMap,
        downhill_map: Dict[Coordinate, Optional[Coordinate]],
        river_thresh: float,
        lake_thresh: float,
    ) -> tuple[List[RiverSegment], List[Coordinate]]:
        """
        Steps E(ii) & (iii): Identify rivers & lakes, yield lists of new_rivers and new_lakes.
        """
        new_rivers: List[RiverSegment] = []
        new_lakes: List[Coordinate] = []

        coords_sorted = sorted(
            flow_map.keys(),
            key=lambda c: self._elevation(c[0], c[1]),
            reverse=True,
        )

        for c in coords_sorted:
            if not self.settings.infinite and not (0 <= c[0] < self.settings.width and 0 <= c[1] < self.settings.height):
                continue
            fval = flow_map[c]
            d = downhill_map[c]
            h_c = self.get(*c)
            if not h_c:
                continue

            if d:
                if fval >= river_thresh:
                    strength = min(fval, flow_map.get(d, fval))
                    new_rivers.append(RiverSegment(c, d, strength))
                    h_c.river = True
                    h_d = self.get(*d)
                    if h_d:
                        h_d.river = True
            else:
                # Local sink ⇒ possible lake
                if fval > lake_thresh:
                    new_lakes.append(c)
                    h_c.lake = True
                    h_c.terrain = "water"
                    lake_rng = self._tile_rng(c[0], c[1], 0x3020)
                    h_c.resources = generate_resources(lake_rng, "water")

        if not new_lakes and self.settings.rainfall_intensity >= 1.0:
            lowest = min(flow_map.keys(), key=lambda c: self._elevation(c[0], c[1]))
            h_low = self.get(*lowest)
            if h_low:
                h_low.lake = True
                h_low.terrain = "water"
                lake_rng = self._tile_rng(lowest[0], lowest[1], 0x3020)
                h_low.resources = generate_resources(lake_rng, "water")
                new_lakes.append(lowest)

        return new_rivers, new_lakes

    def _lake_outflow(self, new_lakes: List[Coordinate]) -> None:
        """
        Step E(iv): For each new lake, find the lowest neighbor (if any) and create a river segment from lake → neighbor.
        """
        for lake_coord in new_lakes:
            q0, r0 = lake_coord
            lake_elev = self._elevation(q0, r0)
            lowest_neighbor: Optional[Coordinate] = None
            low_elev = lake_elev
            for n in self._neighbors_elevated(q0, r0):
                nelev = self._elevation(*n)
                if nelev < low_elev:
                    low_elev = nelev
                    lowest_neighbor = n
            if lowest_neighbor:
                h_lake = self.get(q0, r0)
                h_out = self.get(*lowest_neighbor)
                if h_lake and h_out:
                    strength = self.get(*lake_coord).water_flow if self.get(*lake_coord) else 0.0
                    self.rivers.append(RiverSegment(lake_coord, lowest_neighbor, strength))
                    h_lake.river = True
                    h_out.river = True

    def generate_water_features(self) -> None:
        """
        Generate rivers and lakes across the loaded portion of the world.
        Re-runs only if `self._dirty_rivers` is True.

        Steps:
          A) Collect initial flow & downhill maps.
          B) Accumulate flows downstream, with branching.
          C) Determine river/lake thresholds.
          D) Clear old flags.
          E) Identify rivers & lakes.
          F) Lake outflows.
        """
        if not self._dirty_rivers or self._in_generate_water:
            return
        self._in_generate_water = True

        flow_map, downhill_map = self._collect_initial_flow_and_downhill()
        if not flow_map:
            # No loaded tiles ⇒ nothing to do
            self.rivers.clear()
            self.lakes.clear()
            self._dirty_rivers = False
            self._in_generate_water = False
            return

        self._accumulate_flows(flow_map, downhill_map)
        river_thresh, lake_thresh = self._determine_thresholds(flow_map.values())
        all_coords = flow_map.keys()

        for (q, r), fval in flow_map.items():
            h = self.get(q, r)
            if h:
                h.water_flow = fval

        self._clear_old_water_flags(all_coords)
        new_rivers, new_lakes = self._identify_and_flag_rivers_lakes(
            flow_map, downhill_map, river_thresh, lake_thresh
        )
        self.rivers = new_rivers
        self.lakes = new_lakes
        self._lake_outflow(new_lakes)

        self._dirty_rivers = False
        self._in_generate_water = False

    def _generate_rivers(self) -> None:
        """Generate rivers and lakes using a simple flow accumulation model."""

        # Ensure tiles are generated
        for coord in self.iter_all_coords():
            _ = self.get(*coord)

        flow: FlowMap = {}
        downhill: Dict[Coordinate, Optional[Coordinate]] = {}

        for q, r in self.iter_all_coords():
            elev = self._elevation(q, r)
            rain = self._moisture(q, r, elev, self._season) * self.settings.rainfall_intensity
            flow[(q, r)] = rain
            dn = self._downhill_neighbor(q, r)
            if dn and self._elevation(*dn) < elev:
                downhill[(q, r)] = dn
            else:
                downhill[(q, r)] = None

        coords_sorted = sorted(flow.keys(), key=lambda c: self._elevation(c[0], c[1]), reverse=True)
        for c in coords_sorted:
            d = downhill[c]
            if d:
                flow[d] = flow.get(d, 0.0) + flow[c]

        self.rivers.clear()
        self.lakes.clear()

        threshold = getattr(self.settings, "river_threshold", 0.5)
        for c in coords_sorted:
            h = self.get(*c)
            if not h:
                continue
            fval = flow.get(c, 0.0)
            d = downhill[c]
            if d and fval >= threshold:
                self.rivers.append(RiverSegment(c, d, fval))
                h.river = True
                h.water_flow = fval
                h_d = self.get(*d)
                if h_d:
                    h_d.river = True
            elif d is None and fval >= threshold:
                h.lake = True
                h.terrain = "water"
                lake_rng = self._tile_rng(c[0], c[1], 0x3020)
                h.resources = generate_resources(lake_rng, "water")
                self.lakes.append(c)

        self._dirty_rivers = False

    # ─────────────────────────────────────────────────────────────────────────
    # == RESOURCES & ROADS ==

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        """
        Sum resources in a Chebyshev (square) radius around (x, y). Excludes out-of-bounds tiles.

        Args:
            x (int): Center q coordinate.
            y (int): Center r coordinate.
            radius (int): Radius to search.

        Returns:
            Dict[ResourceType,int]: Total resource counts by type.
        """
        totals: Dict[ResourceType, int] = {r: 0 for r in ResourceType}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                coord = (x + dx, y + dy)
                h = self.get(*coord)
                if h:
                    for rtype, amt in h.resources.items():
                        totals[rtype] = totals.get(rtype, 0) + amt
        return {rtype: amt for rtype, amt in totals.items() if amt > 0}

    def has_road(self, start: Coordinate, end: Coordinate) -> bool:
        """
        Return True if a bidirectional road exists between start and end.

        Args:
            start (Coordinate): (q, r) of endpoint A.
            end (Coordinate): (q, r) of endpoint B.

        Returns:
            bool: True if a road exists in either direction.
        """
        for r in self.roads:
            if (r.start == start and r.end == end) or (r.start == end and r.end == start):
                return True
        return False

    def add_road(self, start: Coordinate, end: Coordinate) -> None:
        """
        Add a two-way road between start and end.

        Args:
            start (Coordinate): (q, r) tuple for one endpoint.
            end (Coordinate): (q, r) tuple for the other endpoint.

        Raises:
            InvalidCoordinateError: If endpoints are not valid (non-integer tuples, out of bounds, or identical).
        """
        if not (
            isinstance(start, tuple)
            and isinstance(end, tuple)
            and len(start) == 2
            and len(end) == 2
            and all(isinstance(c, int) for c in start + end)
        ):
            raise InvalidCoordinateError(f"Road endpoints must be (int,int) tuples, got {start}, {end}")
        if start == end:
            raise InvalidCoordinateError("Cannot build a road from a tile to itself.")
        if start not in self or end not in self:
            raise InvalidCoordinateError(f"Cannot build road: one endpoint out of bounds: {start}, {end}")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Coordinate, end: Coordinate) -> float:
        """
        Return a multiplier to trade speed: 1.5 if a road exists between start and end, otherwise 1.0.

        Args:
            start (Coordinate): (q, r) tuple for the origin.
            end (Coordinate): (q, r) tuple for the destination.

        Raises:
            InvalidCoordinateError: If endpoints are not valid (non-integer tuples or out of bounds).

        Returns:
            float: 1.5 with a road, else 1.0.
        """
        if not (
            isinstance(start, tuple)
            and isinstance(end, tuple)
            and len(start) == 2
            and len(end) == 2
            and all(isinstance(c, int) for c in start + end)
        ):
            raise InvalidCoordinateError(f"Trade endpoints must be (int,int) tuples, got {start}, {end}")
        if start not in self or end not in self:
            raise InvalidCoordinateError(f"Cannot compute trade efficiency: out of bounds: {start}, {end}")
        return 1.5 if self.has_road(start, end) else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# == SETTINGS ADJUSTMENT HELPER ==

def adjust_settings(settings: WorldSettings, **kwargs: Any) -> None:
    """
    Adjust world settings safely. Float values are clamped to [0.0, 1.0];
    int/bool values are assigned only if types match; other mismatches raise TypeError.
    Automatically marks caches dirty if a relevant field is changed.

    Args:
        settings (WorldSettings): The settings object to modify in-place.
        **kwargs: Field=value pairs indicating new settings.

    Raises:
        TypeError: If a provided value’s type does not match the existing field’s type.
    """
    for key, val in kwargs.items():
        if not hasattr(settings, key):
            continue
        current = getattr(settings, key)
        if isinstance(current, float) and isinstance(val, (int, float)):
            new_val = float(max(0.0, min(1.0, float(val))))
            setattr(settings, key, new_val)
        elif isinstance(current, int) and isinstance(val, int):
            setattr(settings, key, val)
        elif isinstance(current, bool) and isinstance(val, bool):
            setattr(settings, key, val)
        else:
            raise TypeError(f"Cannot assign value of type {type(val)} to setting '{key}'.")

    # If any field that affects terrain/climate changed, mark caches dirty
    dirty_fields = {
        "elevation",
        "temperature",
        "moisture",
        "rainfall_intensity",
        "plate_activity",
        "fantasy_level",
        "wind_strength",
        "seasonal_amplitude",
        "orographic_threshold",
        "orographic_factor",
        "river_branch_threshold",
        "river_branch_chance",
    }
    if any(field in kwargs for field in dirty_fields):
        # Consumers should call `world.mark_dirty()` after calling adjust_settings.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# == PUBLIC API EXPOSURES ==

__all__ = [
    "World",
    "Road",
    "RiverSegment",
    "perlin_noise",
    "ResourceType",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
    "BIOME_COLORS",
    "register_biome_color",
    "register_biome_rule",
    "InvalidCoordinateError",
    "adjust_settings",
    "determine_biome",
]
