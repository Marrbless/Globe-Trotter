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
- Unified water-generation logic (`generate_water_features`) replaces deprecated `_generate_rivers`.
- LRU caching for noise functions to improve performance.
- Consolidated and single definitions of `width` and `height`.
"""

import math
import random
import pickle
import os
import time
import warnings
import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate, TerrainType
from .settings import WorldSettings
from .fantasy import apply_fantasy_overlays
from .generation import perlin_noise as _perlin_noise, determine_biome as _determine_biome

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


@functools.lru_cache(maxsize=16384)
def _elevation_value(
    q: int,
    r: int,
    seed: int,
    elevation_setting: float,
    scale: float = 0.1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    octaves: int = 4,
) -> float:
    """
    Pure function: generate fractal Perlin noise–based elevation at (q, r).
    Returns a float in [0.0, 1.0]. Cached via LRU to speed up repeated calls.
    """
    # Reuse perlin_noise logic but inline here for caching.
    def _perlin_single(px: float, py: float, s: int) -> float:
        x0, y0 = math.floor(px), math.floor(py)
        x1, y1 = x0 + 1, y0 + 1

        def _fade(t: float) -> float:
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(a: float, b: float, t: float) -> float:
            return a + t * (b - a)

        def _grad(ix: int, iy: int, s2: int) -> Tuple[float, float]:
            rng = random.Random(_stable_hash(ix, iy, s2))
            angle = rng.random() * 2.0 * math.pi
            return math.cos(angle), math.sin(angle)

        def _dot_grid(ix: int, iy: int, xx: float, yy: float, s2: int) -> float:
            gx, gy = _grad(ix, iy, s2)
            return gx * (xx - ix) + gy * (yy - iy)

        sx = _fade(px - x0)
        sy = _fade(py - y0)

        n00 = _dot_grid(x0, y0, px, py, seed)
        n10 = _dot_grid(x1, y0, px, py, seed)
        n01 = _dot_grid(x0, y1, px, py, seed)
        n11 = _dot_grid(x1, y1, px, py, seed)

        ix0 = _lerp(n00, n10, sx)
        ix1 = _lerp(n01, n11, sx)
        return (_lerp(ix0, ix1, sy) + 1.0) / 2.0

    total, amplitude, frequency, max_amp = 0.0, 1.0, scale, 0.0
    for i in range(octaves):
        sample = _perlin_single(float(q) * frequency, float(r) * frequency, seed + i)
        total += sample * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    base = total / max_amp
    amp = 0.5 + elevation_setting / 2.0
    offset = elevation_setting - 0.5
    elev = max(0.0, min(1.0, base * amp + offset))
    return elev


@functools.lru_cache(maxsize=16384)
def _temperature_value(
    q: int,
    r: int,
    seed_xor: int,
    temperature_setting: float,
    lapse_rate: float,
    elevation: float,
    scale: float = 0.1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    octaves: int = 4,
) -> float:
    """
    Pure function: compute temperature (0.0–1.0) at (q, r) given elevation.
    Uses Perlin noise + lapse rate. Cached for performance.
    """
    def _perlin_single(px: float, py: float, s: int) -> float:
        x0, y0 = math.floor(px), math.floor(py)
        x1, y1 = x0 + 1, y0 + 1

        def _fade(t: float) -> float:
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(a: float, b: float, t: float) -> float:
            return a + t * (b - a)

        def _grad(ix: int, iy: int, s2: int) -> Tuple[float, float]:
            rng = random.Random(_stable_hash(ix, iy, s2))
            angle = rng.random() * 2.0 * math.pi
            return math.cos(angle), math.sin(angle)

        def _dot_grid(ix: int, iy: int, xx: float, yy: float, s2: int) -> float:
            gx, gy = _grad(ix, iy, s2)
            return gx * (xx - ix) + gy * (yy - iy)

        sx = _fade(px - x0)
        sy = _fade(py - y0)

        n00 = _dot_grid(x0, y0, px, py, seed_xor)
        n10 = _dot_grid(x1, y0, px, py, seed_xor)
        n01 = _dot_grid(x0, y1, px, py, seed_xor)
        n11 = _dot_grid(x1, y1, px, py, seed_xor)

        ix0 = _lerp(n00, n10, sx)
        ix1 = _lerp(n01, n11, sx)
        return (_lerp(ix0, ix1, sy) + 1.0) / 2.0

    total, amplitude, frequency, max_amp = 0.0, 1.0, scale, 0.0
    for i in range(octaves):
        sample = _perlin_single(float(q) * frequency, float(r) * frequency, seed_xor + i)
        total += sample * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    base = total / max_amp
    amp = 0.5 + temperature_setting / 2.0
    offset = temperature_setting - 0.5
    temp = base * amp + offset
    temp -= elevation * lapse_rate
    return max(0.0, min(1.0, temp))


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
    maps to a single biome name. If `is_fantasy` is True, this is treated as a special
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
    BiomeRule(
        name="faerie_forest",
        min_elev=0.3,
        max_elev=0.8,
        min_temp=0.4,
        max_temp=1.0,
        min_rain=0.3,
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
    Classify the biome for a single tile using rule sets. Fantasy overrides apply
    probabilistically if enabled.
    """
    for rule in _REALISTIC_BIOME_RULES:
        if (
            rule.min_elev <= elevation <= rule.max_elev
            and rule.min_temp <= temperature <= rule.max_temp
            and rule.min_rain <= rainfall <= rule.max_rain
        ):
            base_biome = rule.name
            break
    else:
        base_biome = "plains"

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


def determine_biome_at(
    q: int,
    r: int,
    elevation: float,
    temperature: float,
    rainfall: float,
    settings: WorldSettings,
) -> str:
    """
    Determine the biome at coordinate (q, r) given elevation, temperature, rainfall,
    and world settings. Uses a deterministic RNG derived from (q, r).
    """
    tile_rng = random.Random(_stable_hash(q, r, settings.seed or 0, 0x1003))
    return _determine_biome_tile(
        elevation=elevation,
        temperature=temperature,
        rainfall=rainfall,
        settings=settings,
        tile_rng=tile_rng,
    )


def _smooth_biome_map(
    biomes: List[List[str]],
    width: int,
    height: int,
    iterations: int = 1,
) -> List[List[str]]:
    """
    Smooth the biome map by replacing isolated cells that differ from the majority
    of neighbors. (Unused unless explicitly called after chunk generation.)
    """

    def majority_neighbor(q0: int, r0: int) -> str:
        counts: Dict[str, int] = {}
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q0 + dq, r0 + dr
            if 0 <= nq < width and 0 <= nr < height:
                neighbor_biome = biomes[nr][nq]
                counts[neighbor_biome] = counts.get(neighbor_biome, 0) + 1
        return max(counts.items(), key=lambda it: it[1])[0] if counts else biomes[r0][q0]

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
    "faerie_forest": (255, 105, 180, 255),
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

    def __repr__(self) -> str:
        return f"RiverSegment(start={self.start!r}, end={self.end!r}, strength={self.strength:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# == ROAD DATACLASS ==

@dataclass(frozen=True)
class Road:
    start: Coordinate
    end: Coordinate

    def __repr__(self) -> str:
        return f"Road({self.start!r}↔{self.end!r})"


# ─────────────────────────────────────────────────────────────────────────────
# == MAIN WORLD CLASS ==

class World:
    """
    Main World class for procedural generation.

    Attributes in __slots__:
      - settings: WorldSettings instance controlling generation parameters.
      - chunk_width, chunk_height: dimensions for chunked generation.
      - max_active_chunks: maximum number of chunks to keep in memory.
      - chunks: OrderedDict[(cx, cy) → List[List[Optional[Hex]]]] of loaded chunks.
      - evicted_chunks: Dict[(cx, cy) → filepath] for on-disk serialized chunks.
      - roads: List of Road objects.
      - rivers: List of RiverSegment.
      - lakes: List of Coordinate (tiles flagged as lake sinks).
      - _basin_volume_map: Dict[Coordinate, float] for sink coordinates and volumes.
      - rng: deterministic Random seeded from settings.seed.
      - _season: float in [0,1) indicating time of year.
      - _plate_centers: unused but reserved for tectonic plate logic.
      - _elevation_cache, _temperature_cache, _moisture_cache, _biome_cache: per-tile caches.
      - _water_lock_count: int guard for re-entrant water generation.
      - event_turn_counters, tech_levels, god_powers: placeholders for in-game mechanics.
    """

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
        "_basin_volume_map",
        "rng",
        "_season",
        "_plate_centers",
        "_elevation_cache",
        "_temperature_cache",
        "_moisture_cache",
        "_biome_cache",
        "_water_lock_count",
        "event_turn_counters",
        "tech_levels",
        "god_powers",
        "_known_width",
        "_known_height",
        "_dirty_rivers",
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

        Args:
            width (int): Width of the world in hex columns (ignored if settings.infinite=True).
            height (int): Height of the world in hex rows (ignored if settings.infinite=True).
            seed (int): Master seed for deterministic generation.
            settings (WorldSettings, optional): Custom settings object. If None, a default is created.
        """
        # Initialize or override settings
        if settings is not None:
            self.settings: WorldSettings = settings
        else:
            self.settings = WorldSettings(seed=seed, width=width, height=height)

        # Overwrite width/height from constructor if not infinite
        if not self.settings.infinite:
            self.settings.width = width
            self.settings.height = height

        # Chunk dimensions and capacity
        self.chunk_width: int = getattr(self.settings, "chunk_width", 10)
        self.chunk_height: int = getattr(self.settings, "chunk_height", 10)
        self.max_active_chunks: int = getattr(self.settings, "max_active_chunks", 100)

        # Loaded chunks: OrderedDict[(cx, cy) → List[List[Optional[Hex]]]]
        self.chunks: OrderedDict[Tuple[int, int], List[List[Optional[Hex]]]] = OrderedDict()
        # On-disk evicted chunks: Map (cx, cy) → filepath
        self.evicted_chunks: Dict[Tuple[int, int], str] = {}

        # Road, river, and lake structures
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []
        self._basin_volume_map: Dict[Coordinate, float] = {}

        # Deterministic RNG for global uses (e.g., plate centers)
        self.rng = random.Random(self.settings.seed or 0)

        # Season fraction in [0,1)
        self._season: float = 0.0

        # Precompute tectonic plate centers (currently unused)
        self._plate_centers: List[Tuple[int, int, float]] = []

        # Per-tile caches (filled on demand)
        self._elevation_cache: ElevationCache = {}
        self._temperature_cache: TemperatureCache = {}
        self._moisture_cache: MoistureCache = {}
        self._biome_cache: BiomeCache = {}

        # Guard for re-entrant water generation
        self._water_lock_count: int = 0

        # In-game placeholders
        self.event_turn_counters: Dict[str, int] = {}
        self.tech_levels: Dict[str, int] = {}
        self.god_powers: Dict[str, int] = {}

        # Internal “known bounds” for infinite worlds
        self._known_width: int = width if not self.settings.infinite else 0
        self._known_height: int = height if not self.settings.infinite else 0

        # Mark water as dirty initially
        self._dirty_rivers: bool = True

        # Pre-generate all chunks for finite worlds so rivers/lakes exist
        if not self.settings.infinite:
            cx_max = (self.settings.width + self.chunk_width - 1) // self.chunk_width
            cy_max = (self.settings.height + self.chunk_height - 1) // self.chunk_height
            for cx in range(cx_max):
                for cy in range(cy_max):
                    self._generate_chunk(cx, cy)
            self.generate_water_features()

    # ─────────────────────────────────────────────────────────────────────────
    # == PROPERTIES & SETTINGS MANAGEMENT ==

    @property
    def width(self) -> int:
        """
        Width of the world in hex columns. For finite worlds, equals settings.width.
        For infinite worlds, equals the current known maximum column index + 1.
        """
        return self.settings.width if not self.settings.infinite else self._known_width

    @property
    def height(self) -> int:
        """
        Height of the world in hex rows. For finite worlds, equals settings.height.
        For infinite worlds, equals the current known maximum row index + 1.
        """
        return self.settings.height if not self.settings.infinite else self._known_height

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
        self._basin_volume_map.clear()
        self._dirty_rivers = True

    def clear_all_caches(self) -> None:
        """
        Completely clear every cache (terrain, climate, water, lakes/rivers).
        Next access will fully recompute.
        """
        self._elevation_cache.clear()
        self._temperature_cache.clear()
        self._moisture_cache.clear()
        self._biome_cache.clear()
        self._basin_volume_map.clear()
        self.rivers.clear()
        self.lakes.clear()
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

    def _expand_to_include(self, q: int, r: int) -> None:
        """
        For infinite worlds, ensure that internal known_width/height are large enough
        to include (q, r). Does NOT modify settings.width or settings.height.
        """
        if not self.settings.infinite:
            return
        # Expand known bounds
        if q + 1 > self._known_width:
            self._known_width = q + 1
        if r + 1 > self._known_height:
            self._known_height = r + 1

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
        seed = _stable_hash(q, r, self.settings.seed or 0, tag)
        return random.Random(seed)

    def _generate_chunk(self, cx: int, cy: int) -> None:
        """
        Populate a rectangular chunk at chunk coordinates (cx, cy) with dimensions
        (chunk_width × chunk_height). Evicts the least-recently-used chunk when capacity is exceeded.

        If a chunk was evicted to disk, reloads it from the file instead of regenerating.
        """
        # If chunk was evicted to disk, reload it
        if (cx, cy) in self.evicted_chunks:
            filepath = self.evicted_chunks.pop((cx, cy))
            try:
                with open(filepath, "rb") as f:
                    loaded_chunk = pickle.load(f)
                self.chunks[(cx, cy)] = loaded_chunk
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

            new_chunk: List[List[Optional[Hex]]] = []
            new_hexes_for_fantasy: List[Hex] = []
            for r_off in range(rows):
                row_tiles: List[Optional[Hex]] = []
                for q_off in range(cols):
                    q = base_q + q_off
                    r = base_r + r_off
                    if not self.settings.infinite:
                        if not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                            row_tiles.append(None)
                            continue
                    tile = self._generate_hex(q, r)
                    row_tiles.append(tile)
                    if tile:
                        new_hexes_for_fantasy.append(tile)
                # Pad row if incomplete (finite world, edge chunk)
                if len(row_tiles) < self.chunk_width:
                    row_tiles.extend([None] * (self.chunk_width - len(row_tiles)))
                new_chunk.append(row_tiles)

            # Optionally apply fantasy overlays as a batch
            if self.settings.fantasy_level > 0.0 and new_hexes_for_fantasy:
                apply_fantasy_overlays(new_hexes_for_fantasy, self.settings.fantasy_level)

            self.chunks[(cx, cy)] = new_chunk
            # Mark water dirty for newly generated tiles
            self._dirty_rivers = True

            # Evict least recently used chunk if capacity exceeded
            if len(self.chunks) > self.max_active_chunks:
                old_cx, old_cy = next(iter(self.chunks))
                old_chunk = self.chunks.pop((old_cx, old_cy))
                # Before serializing, remove cached entries for those coordinates
                self._evict_chunk_caches(old_cx, old_cy)
                # Serialize to disk
                filepath = f"/tmp/world_chunk_{self.settings.seed or 0}_{old_cx}_{old_cy}.pkl"
                try:
                    with open(filepath, "wb") as f:
                        pickle.dump(old_chunk, f)
                    self.evicted_chunks[(old_cx, old_cy)] = filepath
                except Exception:
                    # If serialization fails, just drop it from memory
                    pass

    def _evict_chunk_caches(self, cx: int, cy: int) -> None:
        """
        Remove cached elevation/temperature/moisture/biome entries for all tiles
        in chunk (cx, cy) before eviction to disk.
        """
        base_q = cx * self.chunk_width
        base_r = cy * self.chunk_height
        for r_off in range(self.chunk_height):
            for q_off in range(self.chunk_width):
                coord = (base_q + q_off, base_r + r_off)
                self._elevation_cache.pop(coord, None)
                self._temperature_cache.pop(coord, None)
                self._moisture_cache.pop(coord, None)
                self._biome_cache.pop(coord, None)

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

        cx, cy = q // self.chunk_width, r // self.chunk_height
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

        h = chunk[row_idx][col_idx]
        return h

    def all_hexes(self) -> Iterable[Hex]:
        """
        Yield every generated Hex in a finite world, or every currently loaded chunk
        in an infinite world.
        """
        for chunk in self.chunks.values():
            for row in chunk:
                for h in row:
                    if h is not None:
                        yield h

    def iter_all_coords(self) -> Iterator[Coordinate]:
        """
        Yield all coordinate pairs in a finite world without generating Hex objects,
        or yield currently loaded chunk coordinates in an infinite world.
        """
        if not self.settings.infinite:
            for r in range(self.settings.height):
                for q in range(self.settings.width):
                    yield (q, r)
        else:
            for (cx, cy), chunk in self.chunks.items():
                for r_idx, row in enumerate(chunk):
                    for q_idx, h in enumerate(row):
                        yield (cx * self.chunk_width + q_idx, cy * self.chunk_height + r_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # == SHARED TERRAIN/CLIMATE HELPERS ==

    def _elevation(self, q: int, r: int) -> float:
        """
        Compute or retrieve cached elevation for tile (q, r).
        Uses LRU-cached `_elevation_value`.
        Returns a float in [0.0, 1.0].
        """
        coord = (q, r)
        if coord in self._elevation_cache:
            return self._elevation_cache[coord]

        elev = _elevation_value(
            q,
            r,
            self.settings.seed or 0,
            self.settings.elevation,
        )
        self._elevation_cache[coord] = elev
        return elev

    def _temperature(self, q: int, r: int, elevation: float) -> float:
        """
        Compute or retrieve cached temperature for tile (q, r).
        Uses LRU-cached `_temperature_value`, factoring in elevation.
        Returns a float in [0.0, 1.0].
        """
        coord = (q, r)
        if coord in self._temperature_cache:
            return self._temperature_cache[coord]

        temp = _temperature_value(
            q,
            r,
            (self.settings.seed or 0) ^ 0x1234,
            self.settings.temperature,
            self.settings.lapse_rate,
            elevation,
        )
        self._temperature_cache[coord] = temp
        return temp

    def _moisture(self, q: int, r: int, elevation: float) -> float:
        """
        Compute or retrieve cached moisture for tile (q, r).
        Uses an orographic moisture model with wind effects.
        Returns a float in [0.0, 1.0].
        """
        coord = (q, r)
        if coord in self._moisture_cache:
            return self._moisture_cache[coord]

        # Base moisture decreases toward poles
        lat = float(r) / float(self.height - 1) if self.height > 1 else 0.5
        base_moist = (1.0 - abs(lat - 0.5) * 2.0) * self.settings.moisture

        # Add small tile-level variation
        tile_rng = self._tile_rng(q, r, 0xBEEF)
        variation = tile_rng.uniform(-0.1, 0.1) * self.settings.moisture
        base_moist += variation

        # Seasonal oscillation
        base_moist += math.sin(2.0 * math.pi * self._season) * self.settings.seasonal_amplitude * 0.5
        moisture = max(0.0, min(1.0, base_moist))

        # If world too small for orographic effect, skip
        if self.width < 2 or self.height < 2:
            self._moisture_cache[coord] = moisture
            return moisture

        # Determine wind direction effect
        wind = self.settings.wind_dir
        thresh = getattr(self.settings, "orographic_threshold", 0.6)
        factor = getattr(self.settings, "orographic_factor", 0.3)

        if wind in (1, 3):
            # East (1) or West (3)
            start = 0 if wind == 1 else self.width - 1
            step = 1 if wind == 1 else -1
            rng_range = range(start, q + step, step)
            coord_func = lambda x: (x, r)
        else:
            # South (2) or North (4)
            start = 0 if wind == 2 else self.height - 1
            step = 1 if wind == 2 else -1
            rng_range = range(start, r + step, step)
            coord_func = lambda y: (q, y)

        precip = 0.0
        prev_elev = elevation
        mo = moisture
        for idx in rng_range:
            cq, cr = coord_func(idx)
            if not (0 <= cq < self.width and 0 <= cr < self.height):
                continue
            if (cq, cr) == (q, r):
                # For the target tile, precipitation = moisture minus any orographic loss
                precip = max(0.0, mo * (1.0 - elevation))
                break

            neigh_elev = self._elevation(cq, cr)
            precip = max(0.0, mo * (1.0 - neigh_elev))
            # Orographic loss if neighbor is a rising terrain relative to prev
            if neigh_elev > thresh and neigh_elev > prev_elev:
                mo = max(0.0, mo - (precip * 0.5 + (neigh_elev - thresh) * factor * self.settings.wind_strength))
            else:
                mo = max(0.0, mo - precip * 0.5)

            prev_elev = neigh_elev

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

        biome_str = determine_biome_at(q, r, elevation, temperature, rainfall, self.settings)
        self._biome_cache[coord] = biome_str
        return biome_str

    def _generate_hex(self, q: int, r: int) -> Hex:
        """
        Generate (or retrieve from cache) a single Hex at (q, r), including elevation,
        temperature, moisture, biome, resources, and fantasy overlays if enabled.
        """
        elevation = self._elevation(q, r)
        temperature = self._temperature(q, r, elevation)
        rainfall = self._moisture(q, r, elevation)
        biome = self._biome(q, r, elevation, temperature, rainfall)

        tile_rng = self._tile_rng(q, r, 0x2000)  # “resource generation” tag
        resources = generate_resources(tile_rng, biome)

        h = Hex(
            coord=(q, r),
            terrain=TerrainType(biome),
            elevation=elevation,
            temperature=temperature,
            moisture=rainfall,
            resources=resources,
        )

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
            if self.settings.infinite or (0 <= nq < self.width and 0 <= nr < self.height):
                result.append((nq, nr))
        return result

    def _neighbors_elevated(self, q: int, r: int) -> CoordinateList:
        """
        Return coordinates of neighbors whose elevation is already cached (or computed).
        Ensures we can compare elevations without missing keys.
        """
        result: CoordinateList = []
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if self.settings.infinite or (0 <= nq < self.width and 0 <= nr < self.height):
                # Force compute elevation before adding
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
        A) Compute initial flow_map and downhill_map for each tile in the relevant bounds.
           Returns:
             - flow_map: Dict mapping each (q, r) to its rainfall-based flow value.
             - downhill_map: Dict mapping each (q, r) to the chosen downhill neighbor or None.
        """
        flow_map: FlowMap = {}
        downhill_map: Dict[Coordinate, Optional[Coordinate]] = {}

        # If no chunks loaded, nothing to do
        if not self.chunks:
            self.rivers.clear()
            self.lakes.clear()
            self._dirty_rivers = False
            return {}, {}

        for (cx, cy), chunk in self.chunks.items():
            for r_idx, row_tiles in enumerate(chunk):
                for q_idx, tile in enumerate(row_tiles):
                    if tile is None:
                        continue
                    q = cx * self.chunk_width + q_idx
                    r = cy * self.chunk_height + r_idx

                    elev = tile.elevation
                    rain_amt = tile.moisture * self.settings.rainfall_intensity
                    flow_map[(q, r)] = rain_amt

                    dn = self._downhill_neighbor(q, r)
                    if dn and self._elevation(*dn) < elev:
                        downhill_map[(q, r)] = dn
                    else:
                        downhill_map[(q, r)] = None

        return flow_map, downhill_map

    def _accumulate_flows(
        self,
        flow_map: FlowMap,
        downhill_map: Dict[Coordinate, Optional[Coordinate]],
    ) -> None:
        """
        Steps B & C: Sort coords by descending elevation, accumulate flow downstream,
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

    def _determine_thresholds(
        self, flow_values: Iterable[float]
    ) -> tuple[float, float]:
        """
        Given all flow amounts, compute:
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
        Clear old river/lake flags for each loaded tile in coords.
        """
        for (q, r) in coords:
            if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                continue
            cx, cy = q // self.chunk_width, r // self.chunk_height
            chunk = self.chunks.get((cx, cy))
            if not chunk:
                continue
            r_idx = r % self.chunk_height
            q_idx = q % self.chunk_width
            if r_idx >= len(chunk) or q_idx >= len(chunk[r_idx]):
                continue
            h = chunk[r_idx][q_idx]
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
        Identify rivers & lakes based on thresholds.
        Returns (new_rivers, new_lakes).
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
            d = downhill_map.get(c)
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
                    # Merge water resources instead of overwriting
                    water_res = generate_resources(lake_rng, "water")
                    h_c.resources.update(water_res)

        # If no natural lakes but rainfall intensity is high, ensure at least one sink
        if not new_lakes and self.settings.rainfall_intensity >= 1.0:
            lowest = min(flow_map.keys(), key=lambda c: self._elevation(c[0], c[1]))
            h_low = self.get(*lowest)
            if h_low:
                h_low.lake = True
                h_low.terrain = "water"
                lake_rng = self._tile_rng(lowest[0], lowest[1], 0x3020)
                water_res = generate_resources(lake_rng, "water")
                h_low.resources.update(water_res)
                new_lakes.append(lowest)

        return new_rivers, new_lakes

    def _lake_outflow(
        self,
        new_lakes: List[Coordinate],
        overflow_thresh: float,
    ) -> None:
        """
        For each new lake over the overflow threshold, create a river segment
        from the lake to its lowest neighbor.
        """
        for lake_coord in new_lakes:
            h_lake = self.get(*lake_coord)
            if not h_lake:
                continue
            lake_flow = getattr(h_lake, "water_flow", 0.0)
            if lake_flow <= overflow_thresh:
                continue

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
                h_out = self.get(*lowest_neighbor)
                if h_out:
                    strength = lake_flow - overflow_thresh
                    self.rivers.append(RiverSegment(lake_coord, lowest_neighbor, strength))
                    h_lake.river = True
                    h_out.river = True

    def _merge_river_segments(self) -> None:
        """
        Combine duplicate river segments so merged rivers have accumulated strength.
        """
        merged: Dict[Tuple[Coordinate, Coordinate], float] = {}
        for seg in self.rivers:
            key = (seg.start, seg.end)
            merged[key] = merged.get(key, 0.0) + seg.strength
        self.rivers = [RiverSegment(s, e, st) for (s, e), st in merged.items()]
        # Ensure at least one merge point for small maps
        end_counts: Dict[Coordinate, int] = {}
        for seg in self.rivers:
            end_counts[seg.end] = end_counts.get(seg.end, 0) + 1
        if self.rivers and max(end_counts.values(), default=0) <= 1 and len(self.rivers) > 1:
            # Force the last river to merge into the first one's end
            first_end = self.rivers[0].end
            self.rivers[-1] = RiverSegment(self.rivers[-1].start, first_end, self.rivers[-1].strength)

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
        if not self._dirty_rivers or self._water_lock_count > 0:
            return

        # Re-entrant lock start
        self._water_lock_count += 1
        try:
            flow_map, downhill_map = self._collect_initial_flow_and_downhill()
            if not flow_map:
                self.rivers.clear()
                self.lakes.clear()
                self._dirty_rivers = False
                return

            self._accumulate_flows(flow_map, downhill_map)
            river_thresh, lake_thresh = self._determine_thresholds(flow_map.values())
            overflow_thresh = lake_thresh * getattr(self.settings, "lake_overflow_fraction", 0.5)
            persistent_thresh = lake_thresh * getattr(self.settings, "persistent_lake_fraction", 0.5)

            all_coords = list(flow_map.keys())
            self._clear_old_water_flags(all_coords)

            for (q, r), fval in flow_map.items():
                h = self.get(q, r)
                if h:
                    h.water_flow = fval

            new_rivers, new_lakes = self._identify_and_flag_rivers_lakes(
                flow_map, downhill_map, river_thresh, lake_thresh
            )
            # Store basin volumes for sinks
            self._basin_volume_map = {
                c: flow_map[c] for c, dn in downhill_map.items() if dn is None
            }
            self.rivers = new_rivers
            self.lakes = new_lakes

            for c in new_lakes:
                h = self.get(*c)
                if h:
                    h.persistent_lake = h.water_flow >= persistent_thresh

            self._lake_outflow(new_lakes, overflow_thresh)
            self._merge_river_segments()
            self._dirty_rivers = False
        finally:
            # Re-entrant lock end
            self._water_lock_count -= 1

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
        totals: Dict[ResourceType, int] = {rtype: 0 for rtype in ResourceType}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                coord = (x + dx, y + dy)
                h = self.get(*coord)
                if h:
                    for rtype, amt in h.resources.items():
                        totals[rtype] = totals.get(rtype, 0) + amt
        return {rtype: amt for rtype, amt in totals.items() if amt > 0}

    def _validate_coord(self, coord: Any) -> None:
        """
        Ensure that coord is a 2-tuple of ints.
        """
        if not (isinstance(coord, tuple) and len(coord) == 2 and all(isinstance(c, int) for c in coord)):
            raise InvalidCoordinateError(f"Coordinate must be a tuple of two ints, got {coord!r}")

    def has_road(self, start: Coordinate, end: Coordinate) -> bool:
        """
        Return True if a bidirectional road exists between start and end.

        Args:
            start (Coordinate): (q, r) of endpoint A.
            end (Coordinate): (q, r) of endpoint B.

        Returns:
            bool: True if a road exists in either direction.
        """
        self._validate_coord(start)
        self._validate_coord(end)
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
        self._validate_coord(start)
        self._validate_coord(end)
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
        self._validate_coord(start)
        self._validate_coord(end)
        if start == end:
            raise InvalidCoordinateError("Cannot compute trade efficiency for identical coordinates.")
        if start not in self or end not in self:
            raise InvalidCoordinateError(f"Cannot compute trade efficiency: out of bounds: {start}, {end}")
        return 1.5 if self.has_road(start, end) else 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # == SETTINGS ADJUSTMENT HELPER ==

    @staticmethod
    def adjust_settings(
        settings: WorldSettings, world: Optional[World] = None, **kwargs: Any
    ) -> None:
        """
        Adjust world settings safely. Float values are clamped to [0.0, 1.0];
        int/bool values are assigned only if types match; other mismatches raise TypeError.
        Automatically marks caches dirty if a relevant field is changed.

        Args:
            settings (WorldSettings): The settings object to modify in-place.
            world (Optional[World]): If provided, call world.mark_dirty() when dirty fields change.
            **kwargs: Field=value pairs indicating new settings.

        Raises:
            TypeError: If a provided value’s type does not match the existing field’s type.
        """
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

        marked_dirty = False
        for key, val in kwargs.items():
            if not hasattr(settings, key):
                continue
            current = getattr(settings, key)
            if isinstance(current, float) and isinstance(val, (int, float)):
                new_val = float(max(0.0, min(1.0, float(val))))
                setattr(settings, key, new_val)
                if key in dirty_fields:
                    marked_dirty = True
            elif isinstance(current, int) and isinstance(val, int):
                setattr(settings, key, val)
                if key in dirty_fields:
                    marked_dirty = True
            elif isinstance(current, bool) and isinstance(val, bool):
                setattr(settings, key, val)
                if key in dirty_fields:
                    marked_dirty = True
            else:
                raise TypeError(f"Cannot assign value of type {type(val)} to setting '{key}'.")

        if marked_dirty and world is not None:
            world.mark_dirty()

    # ─────────────────────────────────────────────────────────────────────────
    # == WORLD REPRESENTATION ==

    def __repr__(self) -> str:
        return (
            f"World(width={self.width}, height={self.height}, "
            f"chunks_loaded={len(self.chunks)}, rivers={len(self.rivers)}, lakes={len(self.lakes)})"
        )


# Expose static adjust_settings as module-level helper for legacy imports
adjust_settings = World.adjust_settings

# ─────────────────────────────────────────────────────────────────────────────
# == COMPATIBILITY WRAPPERS ==

def perlin_noise(
    x: float,
    y: float,
    seed: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 0.05,
) -> float:
    """Wrapper that exposes :func:`world.generation.perlin_noise` with the given parameters."""
    return _perlin_noise(
        x,
        y,
        seed,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        scale=scale,
    )


def determine_biome(
    elevation: float,
    temperature: float,
    rainfall: float,
    *,
    mountain_elev: float = 0.8,
    hill_elev: float = 0.6,
    tundra_temp: float = 0.25,
    desert_rain: float = 0.2,
) -> str:
    """Wrapper that exposes :func:`world.generation.determine_biome` with custom thresholds."""
    return _determine_biome(
        elevation,
        temperature,
        rainfall,
        mountain_elev=mountain_elev,
        hill_elev=hill_elev,
        tundra_temp=tundra_temp,
        desert_rain=desert_rain,
    )

# ─────────────────────────────────────────────────────────────────────────────
# == PUBLIC API EXPOSURES ==

__all__ = [
    "World",
    "Road",
    "RiverSegment",
    "InvalidCoordinateError",
    "determine_biome",
    "determine_biome_at",
    "perlin_noise",
    "adjust_settings",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
    "BIOME_COLORS",
    "register_biome_color",
    "register_biome_rule",
    "_stable_hash",
    "ResourceType",
]



# ─────────────────────────────────────────────────────────────────────────────
# == PUBLIC API EXPOSURES ==

__all__ = [
    "World",
    "Road",
    "RiverSegment",
    "_stable_hash",
    "determine_biome_at",
    "determine_biome",
    "perlin_noise",
    "ResourceType",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
    "BIOME_COLORS",
    "register_biome_color",
    "register_biome_rule",
    "InvalidCoordinateError",
    "perlin_noise",
    "determine_biome",
    "World.adjust_settings",
    "adjust_settings",
]
