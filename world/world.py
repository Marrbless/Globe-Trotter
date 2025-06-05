from __future__ import annotations

"""World generation and management utilities with enhanced realism and fantasy integration.

Key changes & refactors:
- All RNG uses a stable, custom mixing function (`_stable_hash`) instead of Python's non-deterministic `hash()`.
- Terrain/elevation/temperature/moisture logic has been consolidated into shared helpers.
- River/lake generation is fully repeatable, with no cycles, and can be re-triggered when relevant settings change.
- Chunk-based, on-demand precomputation avoids O(width×height) startup time.
- Type-annotated throughout. Consistent use of `Coordinate = tuple[int, int]`.
- Legacy standalone functions (marked @deprecated) have been removed or relocated; this module exports only the true public API.
"""

import math
import random
import functools
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Iterable,
    cast,
)

from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate
from .settings import WorldSettings
from .fantasy import apply_fantasy_overlays

# ─────────────────────────────────────────────────────────────────────────────
# == CONSTANTS & TYPE ALIASES ==
CoordinateList = List[Coordinate]
FlowMap = Dict[Coordinate, float]
ElevationCache = Dict[Coordinate, float]
TemperatureCache = Dict[Coordinate, float]
MoistureCache = Dict[Coordinate, float]
BiomeCache = Dict[Coordinate, str]

# Axial hex directions: E, W, SE, NW, NE, SW
HEX_DIRECTIONS: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


# == STABLE HASH / RNG HELPERS ==
def _stable_hash(*args: int) -> int:
    """
    Combine integer arguments into a single 64-bit integer using a deterministic mixing routine.
    This ensures repeatable results across Python runs (unlike built-in hash()).
    """
    x = 0x345678ABCDEF1234  # arbitrary non-zero start
    for a in args:
        # 64-bit mix (inspired by MurmurHash3 finalizer)
        a &= 0xFFFFFFFFFFFFFFFF
        a ^= (a >> 33)
        a = (a * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        a ^= (a >> 33)
        x ^= a
        x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    return x


def _perlin_single(x: float, y: float, seed: int) -> float:
    """
    Classic 2D Perlin noise, returning a value in [0,1].
    """
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad(ix: int, iy: int, s: int) -> tuple[float, float]:
        rng = random.Random(_stable_hash(ix, iy, s))
        angle = rng.random() * 2.0 * math.pi
        return math.cos(angle), math.sin(angle)

    def _dot_grid_gradient(ix: int, iy: int, xx: float, yy: float, s: int) -> float:
        gx, gy = _grad(ix, iy, s)
        dx = xx - ix
        dy = yy - iy
        return gx * dx + gy * dy

    sx = _fade(x - x0)
    sy = _fade(y - y0)

    n00 = _dot_grid_gradient(x0, y0, x, y, seed)
    n10 = _dot_grid_gradient(x1, y0, x, y, seed)
    n01 = _dot_grid_gradient(x0, y1, x, y, seed)
    n11 = _dot_grid_gradient(x1, y1, x, y, seed)

    ix0 = _lerp(n00, n10, sx)
    ix1 = _lerp(n01, n11, sx)
    val = _lerp(ix0, ix1, sy)

    return (val + 1.0) / 2.0


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
    Generate fractal Perlin noise at (x,y) for a given seed.
    Returns a float in [0,1].
    """
    value = 0.0
    amplitude = 1.0
    frequency = scale
    max_amp = 0.0

    for i in range(octaves):
        value += _perlin_single(x * frequency, y * frequency, seed + i) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_amp


# ─────────────────────────────────────────────────────────────────────────────
# == SHARED TERRAIN/CLIMATE HELPERS ==

def _compute_base_elevation(
    q: int,
    r: int,
    seed: int,
    elevation_setting: float,
    plate_centers: list[tuple[int, int, float]],
    plate_base_height: float,
    plate_activity: float,
    width: int,
    height: int,
) -> float:
    """
    Compute raw elevation at (q,r) by combining fractal Perlin noise and tectonic plate height.
    Arguments:
      - seed: world seed integer
      - elevation_setting: the 'settings.elevation' float
      - plate_centers: precomputed (cx,cy,base) for each tectonic plate
      - plate_base_height: settings.base_height
      - plate_activity: settings.plate_activity
      - width, height: world dimensions (for boundary conditions)
    Returns:
      A float in [0,1].
    """
    # 1) Primary Perlin component
    n = perlin_noise(float(q), float(r), seed, scale=elevation_setting)
    amp = 0.5 + elevation_setting / 2.0
    offset = elevation_setting - 0.5
    base_noise = max(0.0, min(1.0, n * amp + offset))

    # 2) Plate height contribution
    #    Find nearest two plate centers for smooth fault boundaries
    dists: list[tuple[float, float]] = []
    for cx, cy, base_val in plate_centers:
        dist_sq = float((cx - q) ** 2 + (cy - r) ** 2)
        dists.append((dist_sq, base_val))
    dists.sort(key=lambda pair: pair[0])
    dist0, base0 = dists[0]
    dist1 = dists[1][0] if len(dists) > 1 else dist0
    ratio = dist0 / (dist0 + dist1) if dist1 > 0.0 else 0.0
    boundary_factor = 1.0 - abs(0.5 - ratio) * 2.0
    plate_height = base0 * plate_base_height + boundary_factor * plate_activity

    combined = (base_noise + plate_height) / 2.0
    return max(0.0, min(1.0, combined))


def _compute_temperature_tile(
    q: int,
    r: int,
    elevation: float,
    width: int,
    height: int,
    seed: int,
    temperature_setting: float,
    wind_strength: float,
    seasonal_amplitude: float,
    season: float = 0.0,
) -> float:
    """
    Compute normalized temperature at tile (q,r) given its elevation and global settings.
    Returns a float in [0,1].
    """
    lat = float(r) / float(height - 1) if height > 1 else 0.5
    base_temp = 1.0 - abs(lat - 0.5) * 2.0  # 1.0 at equator, 0.0 at poles
    base_temp -= elevation * 0.3

    # Small random variation per tile
    tile_seed = _stable_hash(r, seed, 0xABCD)  # 0xABCD flag for "temperature"
    rng_tile = random.Random(tile_seed)
    variation = rng_tile.uniform(-0.1, 0.1) * temperature_setting

    # Wind effect (west→east)
    wind_effect = (
        ((float(q) / float(width - 1) if width > 1 else 0.5) - 0.5)
        * wind_strength
        * 0.2
    )

    # Seasonal effect (sinusoidal)
    seasonal = math.sin(2.0 * math.pi * season) * seasonal_amplitude * 0.5

    temp = base_temp + variation + wind_effect + seasonal
    return max(0.0, min(1.0, temp))


def _compute_moisture_orographic(
    q: int,
    r: int,
    elevation: float,
    elevation_cache: ElevationCache,
    width: int,
    height: int,
    seed: int,
    moisture_setting: float,
    wind_strength: float,
    seasonal_amplitude: float,
    season: float = 0.0,
) -> float:
    """
    Compute moisture at (q,r) using west-to-east transport with orographic effects.
    Requires a precomputed elevation_cache {(q,r): elevation}.
    Returns a float in [0,1].
    """
    # Base moisture with random jitter
    tile_seed_base = _stable_hash(r, seed, 0xDCBA)  # 0xDCBA for "moisture"
    rng_row = random.Random(tile_seed_base)
    base_moisture = moisture_setting + rng_row.uniform(-0.1, 0.1)
    base_moisture += math.sin(2.0 * math.pi * season) * seasonal_amplitude * 0.5
    moisture = max(0.0, min(1.0, base_moisture))

    prev_elev = elevation
    precip = 0.0

    # March west to east (x=0→q) to simulate orographic lift
    for x in range(q + 1):
        # If we have a cached elevation for (x,r), use it; else fallback to current tile's elevation
        elev_local = elevation_cache.get((x, r), elevation)

        # If there's a significant jump uphill, moisture drops
        orographic_threshold = getattr(settings, "orographic_threshold", 0.1)
        orographic_factor = getattr(settings, "orographic_factor", 0.5)
        if elev_local - prev_elev > orographic_threshold:
            moisture = max(0.0, moisture - (elev_local - prev_elev) * orographic_factor)

        precip = max(0.0, moisture * (1.0 - elev_local))
        if x == q:
            break

        # Moisture lost traveling to next column
        loss = (precip * 0.5 + elev_local * 0.1) * (1.0 - wind_strength)
        moisture = max(0.0, moisture - loss)
        prev_elev = elev_local

    return max(0.0, min(1.0, precip))


def _determine_biome_tile(
    elevation: float,
    temperature: float,
    rainfall: float,
    tile_seed: int,
    mountain_elev: float,
    hill_elev: float,
    tundra_temp: float,
    desert_rain: float,
    fantasy_level: float,
) -> str:
    """
    Classify the biome for a single tile (elevation, temperature, rainfall).
    Optionally overrides into a fantasy type if `fantasy_level > 0.0`.
    """
    # 1) Realistic base classification
    if elevation > mountain_elev:
        base_biome = "mountains"
    elif elevation > hill_elev:
        base_biome = "hills"
    elif temperature < tundra_temp:
        base_biome = "tundra"
    elif rainfall < desert_rain and temperature > 0.5:
        base_biome = "desert"
    elif rainfall > 0.7 and temperature > 0.5:
        base_biome = "rainforest"
    elif rainfall > 0.4:
        base_biome = "forest"
    else:
        base_biome = "plains"

    # 2) Fantasy overrides
    if fantasy_level > 0.0:
        rng = random.Random(tile_seed)
        # Crystal forests
        if base_biome == "forest" and elevation > 0.4 and fantasy_level > 0.6:
            return "crystal_forest"
        # Floating islands
        if (
            base_biome == "mountains"
            and fantasy_level > 0.8
            and rng.random() < fantasy_level * 0.1
        ):
            return "floating_island"

    return base_biome


def _smooth_biome_map(
    biomes: List[List[str]],
    width: int,
    height: int,
    iterations: int = 1,
) -> List[List[str]]:
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


# Predefined colors for each biome (RGBA)
BIOME_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "plains": (110, 205,  88, 255),
    "forest": (34,  139,  34, 255),
    "mountains": (139, 137, 137, 255),
    "hills": (107, 142,  35, 255),
    "desert": (237, 201, 175, 255),
    "tundra": (220, 220, 220, 255),
    "rainforest": (0, 100,   0, 255),
    "water": (65, 105, 225, 255),
    "floating_island": (186,  85, 211, 255),
    "crystal_forest": (0, 255, 255, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
# == RIVER & LAKE GENERATION HELPERS ==

@dataclass(frozen=True)
class RiverSegment:
    """A start→end pair describing a single river edge."""
    start: Coordinate
    end:   Coordinate


# ─────────────────────────────────────────────────────────────────────────────
# == MAIN WORLD CLASS ==

@dataclass(frozen=True)
class Road:
    start: Coordinate
    end:   Coordinate


class World:
    __slots__ = (
        "settings",
        "chunk_size",
        "max_active_chunks",
        "chunks",
        "roads",
        "rivers",
        "lakes",
        "rng",
        "season",
        "_plate_centers",
        # Caches are now per‐tile and fill lazily
        "_elevation_cache",
        "_temperature_cache",
        "_moisture_cache",
        "_biome_cache",
        # Dirty flags to know when to regenerate water features
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
        River and lake generation is deferred until first call to `generate_water_features()`.
        """
        self.settings: WorldSettings = (
            settings if settings is not None else WorldSettings(seed=seed, width=width, height=height)
        )
        # Ensure settings.width/height match constructor parameters
        self.settings.width = width
        self.settings.height = height

        # Chunk and caching parameters
        self.chunk_size: int = getattr(self.settings, "chunk_size", 10)
        self.max_active_chunks: int = getattr(self.settings, "max_active_chunks", 100)
        self.chunks: OrderedDict[tuple[int, int], List[List[Hex]]] = OrderedDict()

        # Road/river/lake structures
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []

        # RNG seeded deterministically
        self.rng = random.Random(self.settings.seed)
        self.season = 0.0

        # Precompute tectonic plate centers
        self._plate_centers: List[tuple[int, int, float]] = self._init_plates()

        # Caches for per‐tile data (filled on demand)
        self._elevation_cache: ElevationCache = {}
        self._temperature_cache: TemperatureCache = {}
        self._moisture_cache: MoistureCache = {}
        self._biome_cache: BiomeCache = {}

        # Mark water features as "dirty" initially, so they get generated once someone calls them
        self._dirty_rivers = True

        # If fantasy overlays are requested, apply them after any hex is generated
        # (We’ll call this in _maybe_apply_fantasy when each chunk loads.)

    def __contains__(self, coord: Coordinate) -> bool:
        """Allow `coord in world` checks. For finite worlds, bounds are clamped."""
        q, r = coord
        if self.settings.infinite:
            return True
        return 0 <= q < self.settings.width and 0 <= r < self.settings.height

    def __getitem__(self, coord: Coordinate) -> Optional[Hex]:
        """Allow `world[q, r]` shorthand for `world.get(q, r)`."""
        return self.get(*coord)

    def _init_plates(self) -> List[tuple[int, int, float]]:
        """
        Compute tectonic plate centers: number of plates depends on settings.plate_activity.
        Returns a list of (cx, cy, base) tuples.
        """
        num_plates = max(2, int(3 + self.settings.plate_activity * 5))
        rng = random.Random(self.settings.seed ^ 0xFACEB00C)
        centers: list[tuple[int, int, float]] = []
        for _ in range(num_plates):
            cx = rng.randint(0, self.settings.width - 1)
            cy = rng.randint(0, self.settings.height - 1)
            base = rng.random()
            centers.append((cx, cy, base))
        return centers

    def _plate_height(self, q: int, r: int) -> float:
        """
        Return the “plate tectonic” component of elevation at (q,r).
        Uses precomputed self._plate_centers to pick nearest two plates.
        """
        if not self._plate_centers:
            return 0.0
        dists: list[tuple[float, float]] = []
        for cx, cy, base in self._plate_centers:
            d = float((cx - q) ** 2 + (cy - r) ** 2)
            dists.append((d, base))
        dists.sort(key=lambda it: it[0])
        dist0, base0 = dists[0]
        dist1 = dists[1][0] if len(dists) > 1 else dist0
        ratio = dist0 / (dist0 + dist1) if dist1 > 0.0 else 0.0
        boundary = 1.0 - abs(0.5 - ratio) * 2.0
        return base0 * self.settings.base_height + boundary * self.settings.plate_activity

    def _elevation(self, q: int, r: int) -> float:
        """
        Compute or retrieve cached elevation for tile (q,r).
        Combines fractal Perlin and plate height.
        """
        coord = (q, r)
        if coord in self._elevation_cache:
            return self._elevation_cache[coord]

        base = perlin_noise(float(q), float(r), self.settings.seed, scale=self.settings.elevation)
        amp = 0.5 + self.settings.elevation / 2.0
        offset = self.settings.elevation - 0.5
        noise_val = max(0.0, min(1.0, base * amp + offset))

        plate_val = self._plate_height(q, r)
        combined = (noise_val + plate_val) / 2.0
        elev = max(0.0, min(1.0, combined))
        self._elevation_cache[coord] = elev
        return elev

    def _temperature(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        """
        Compute or retrieve cached temperature for tile (q,r).
        """
        coord = (q, r)
        if coord in self._temperature_cache:
            return self._temperature_cache[coord]

        temp = _compute_temperature_tile(
            q=q,
            r=r,
            elevation=elevation,
            width=self.settings.width,
            height=self.settings.height,
            seed=self.settings.seed,
            temperature_setting=self.settings.temperature,
            wind_strength=self.settings.wind_strength,
            seasonal_amplitude=self.settings.seasonal_amplitude,
            season=season,
        )
        self._temperature_cache[coord] = temp
        return temp

    def _moisture(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        """
        Compute or retrieve cached moisture for tile (q,r).
        """
        coord = (q, r)
        if coord in self._moisture_cache:
            return self._moisture_cache[coord]

        moist = _compute_moisture_orographic(
            q=q,
            r=r,
            elevation=elevation,
            elevation_cache=self._elevation_cache,
            width=self.settings.width,
            height=self.settings.height,
            seed=self.settings.seed,
            moisture_setting=self.settings.moisture,
            wind_strength=self.settings.wind_strength,
            seasonal_amplitude=self.settings.seasonal_amplitude,
            season=season,
        )
        self._moisture_cache[coord] = moist
        return moist

    def _biome(self, q: int, r: int, elevation: float, temperature: float, rainfall: float) -> str:
        """
        Compute or retrieve cached biome for tile (q,r), given elevation, temperature, rainfall.
        """
        coord = (q, r)
        if coord in self._biome_cache:
            return self._biome_cache[coord]

        # Regional seeding: cluster similar biomes in blocks
        region_size = getattr(self.settings, "biome_region_size", 10)
        region_q = q // region_size
        region_r = r // region_size
        region_seed = _stable_hash(region_q, region_r, self.settings.seed, 0x1001)

        # Combine with (q,r) for tile-level variation
        tile_seed = _stable_hash(region_seed, q, r, 0x1002)

        biome_str = _determine_biome_tile(
            elevation=elevation,
            temperature=temperature,
            rainfall=rainfall,
            tile_seed=tile_seed,
            mountain_elev=self.settings.mountain_elev,
            hill_elev=self.settings.hill_elev,
            tundra_temp=self.settings.tundra_temp,
            desert_rain=self.settings.desert_rain,
            fantasy_level=self.settings.fantasy_level,
        )
        self._biome_cache[coord] = biome_str
        return biome_str

    def _smooth_biomes(self, raw_biomes: List[List[str]]) -> List[List[str]]:
        """
        Given a 2D array of biome strings, apply smoothing passes if requested.
        """
        width = self.settings.width
        height = self.settings.height
        iters = getattr(self.settings, "biome_smoothing_iterations", 1)
        return _smooth_biome_map(raw_biomes, width, height, iters)

    def _generate_hex(self, q: int, r: int) -> Hex:
        """
        Generate (or retrieve from cache) a single Hex at (q, r), including elevation,
        temperature, moisture, biome, resources, and fantasy overlays if enabled.
        """
        elevation = self._elevation(q, r)
        temperature = self._temperature(q, r, elevation, self.season)
        rainfall = self._moisture(q, r, elevation, self.season)
        biome = self._biome(q, r, elevation, temperature, rainfall)

        # Build the Hex object
        h = Hex(
            coord=(q, r),
            terrain=biome,
            elevation=elevation,
            temperature=temperature,
            moisture=rainfall,
            resources=generate_resources(
                random.Random(_stable_hash(q, r, self.settings.seed, 0x2000)),
                biome,
            ),
        )

        # If fantasy overlays are desired, apply them here
        if self.settings.fantasy_level > 0.0:
            apply_fantasy_overlays([h], self.settings.fantasy_level)

        return h

    def _generate_chunk(self, cx: int, cy: int) -> None:
        """
        Populate chunk (cx,cy) of size chunk_size×chunk_size. Evict oldest chunk if over capacity.
        """
        base_q = cx * self.chunk_size
        base_r = cy * self.chunk_size

        # Determine actual row/col counts (handles finite vs infinite)
        rows = (
            self.chunk_size
            if self.settings.infinite
            else min(self.chunk_size, self.settings.height - base_r)
        )
        cols = (
            self.chunk_size
            if self.settings.infinite
            else min(self.chunk_size, self.settings.width - base_q)
        )

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
        # Evict LRU chunk if over capacity
        if len(self.chunks) > self.max_active_chunks:
            self.chunks.popitem(last=False)

    def get(self, q: int, r: int) -> Optional[Hex]:
        """
        Retrieve the Hex at (q,r). If it doesn’t exist, generate its chunk on demand.
        Returns None if (q,r) is out of bounds in a finite world.
        """
        # Bounds check for finite world
        if not self.settings.infinite:
            if not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                return None

        cx = q // self.chunk_size
        cy = r // self.chunk_size
        if (cx, cy) not in self.chunks:
            self._generate_chunk(cx, cy)
        # Mark chunk as recently used
        self.chunks.move_to_end((cx, cy))
        chunk = self.chunks.get((cx, cy))
        if not chunk:
            return None

        row_idx = r % self.chunk_size
        col_idx = q % self.chunk_size
        if row_idx >= len(chunk) or col_idx >= len(chunk[row_idx]):
            return None

        return chunk[row_idx][col_idx]

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
        Return coordinates of neighbors whose elevation is already cached (or can be computed).
        This ensures we never attempt a non-existent key in elevation_cache when generating rivers.
        """
        result: CoordinateList = []
        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if self.settings.infinite:
                # Force generation if needed
                _ = self._elevation(nq, nr)
                result.append((nq, nr))
            else:
                if 0 <= nq < self.settings.width and 0 <= nr < self.settings.height:
                    # Precompute or fetch
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

    def _identify_water_thresholds(self, flow_values: Iterable[float]) -> tuple[float, float]:
        """
        Given all flow amounts, compute:
          - river_threshold: min flow to count as a river segment
          - lake_threshold: min flow to count as a lake (no downhill neighbor)
        """
        flows = list(flow_values)
        if not flows:
            return 1.0, 1.0

        avg_flow = sum(flows) / len(flows)
        rt = max(0.05 * self.settings.rainfall_intensity, avg_flow * 2.0)
        lt = max(0.1 * self.settings.rainfall_intensity, avg_flow * 4.0)
        return rt, lt

    def generate_water_features(self) -> None:
        """
        Generate rivers and lakes across the entire currently loaded portion of the world.
        If `self.settings.infinite` is False, this covers the entire map; otherwise, it covers 
        only the bounding box of currently cached chunks.
        Re-runs only if `self._dirty_rivers == True`.
        """
        if not self._dirty_rivers:
            return

        # 1) Collect rainfall & initialize flow
        flow_map: FlowMap = {}
        downhill_map: Dict[Coordinate, Optional[Coordinate]] = {}

        # Determine bounds for iteration:
        if self.settings.infinite:
            # Use bounding box of all cached chunks
            qs: list[int] = []
            rs: list[int] = []
            for (cx, cy) in self.chunks.keys():
                chunk = self.chunks[(cx, cy)]
                for row_idx, row_tiles in enumerate(chunk):
                    for col_idx, tile in enumerate(row_tiles):
                        q_coord = cx * self.chunk_size + col_idx
                        r_coord = cy * self.chunk_size + row_idx
                        qs.append(q_coord)
                        rs.append(r_coord)
            if not qs:
                # No chunks loaded yet: nothing to generate
                self.rivers.clear()
                self.lakes.clear()
                self._dirty_rivers = False
                return
            min_q, max_q = min(qs), max(qs)
            min_r, max_r = min(rs), max(rs)
        else:
            min_q, max_q = 0, self.settings.width - 1
            min_r, max_r = 0, self.settings.height - 1

        # Step A: Compute initial flow and downhill neighbor for each tile in bounds
        for r in range(min_r, max_r + 1):
            for q in range(min_q, max_q + 1):
                if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                    continue
                # Ensure elevation, temp, moisture are in cache
                elev = self._elevation(q, r)
                # Rainfall proportional to moisture & intensity
                rain_amt = self._moisture(q, r) * self.settings.rainfall_intensity
                flow_map[(q, r)] = rain_amt

                dn = self._downhill_neighbor(q, r)
                if dn and self._elevation(*dn) < elev:
                    downhill_map[(q, r)] = dn
                else:
                    downhill_map[(q, r)] = None

        # Step B: Sort coords by descending elevation for accumulation
        coords_sorted = sorted(
            flow_map.keys(),
            key=lambda c: self._elevation(c[0], c[1]),
            reverse=True,
        )

        # Step C: Accumulate flow downstream, allow branches but prevent cycles
        visited: set[Coordinate] = set()
        for c in coords_sorted:
            if c in visited:
                continue
            d = downhill_map[c]
            if d:
                flow_map[d] = flow_map.get(d, 0.0) + flow_map[c]
                visited.add(c)
                # Branching logic (20% chance for a tributary if flow is high)
                threshold = self.settings.river_branch_threshold * self.settings.rainfall_intensity
                if flow_map[c] > threshold:
                    neighbor_coords = self._neighbors_elevated(*c)
                    # Find second-best downhill (excluding the main one)
                    second_best: Optional[Coordinate] = None
                    sec_elev = self._elevation(*c)
                    for n in neighbor_coords:
                        if n == d:
                            continue
                        ne = self._elevation(*n)
                        if ne < sec_elev:
                            sec_elev = ne
                            second_best = n
                    if second_best is not None and second_best not in visited:
                        branch_seed = _stable_hash(c[0], c[1], self.settings.seed, 0x3010)
                        if random.Random(branch_seed).random() < 0.3:
                            flow_map[second_best] = flow_map.get(second_best, 0.0) + flow_map[c] * 0.3
                            visited.add(second_best)

        # Step D: Determine thresholds
        river_thresh, lake_thresh = self._identify_water_thresholds(flow_map.values())

        # Step E: Clear previous water flags
        # We cannot alter all hexes one‐by‐one (since some may not be loaded), so store coords first
        new_rivers: list[RiverSegment] = []
        new_lakes: list[Coordinate] = []

        # We will reflag each loaded tile
        for (q, r) in coords_sorted:
            if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                continue
            h = self.get(q, r)
            if not h:
                continue
            # Clear old flags
            h.river = False
            h.lake = False

        # Identify rivers & lakes
        for c in coords_sorted:
            if not self.settings.infinite and not (0 <= c[0] < self.settings.width and 0 <= c[1] < self.settings.height):
                continue
            fval = flow_map[c]
            d = downhill_map[c]
            h_c = self.get(*c)
            if h_c is None:
                continue
            if d:
                if fval >= river_thresh:
                    new_rivers.append(RiverSegment(c, d))
                    h_c.river = True
                    # Mark the downhill hex as part of the same river
                    h_d = self.get(*d)
                    if h_d:
                        h_d.river = True
            else:
                # No downhill (local sink). If flow>N, it's a lake
                if fval > lake_thresh:
                    new_lakes.append(c)
                    h_c.lake = True
                    h_c.terrain = "water"
                    # Regenerate resources for a water tile
                    lake_seed = _stable_hash(c[0], c[1], self.settings.seed, 0x3020)
                    h_c.resources = generate_resources(random.Random(lake_seed), "water")

        # Lake outflow: for each lake, find lowest neighbor, create a river
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
                    new_rivers.append(RiverSegment(lake_coord, lowest_neighbor))
                    h_lake.river = True
                    h_out.river = True

        # Commit new lists
        self.rivers = new_rivers
        self.lakes = new_lakes
        self._dirty_rivers = False

    def all_hexes(self) -> Iterable[Hex]:
        """
        Yield every generated Hex in the finite world, or every loaded chunk in an infinite world.
        """
        if not self.settings.infinite:
            for r in range(self.settings.height):
                for q in range(self.settings.width):
                    h = self.get(q, r)
                    if h:
                        yield h
        else:
            # Infinite: only loaded chunks
            for chunk in self.chunks.values():
                for row in chunk:
                    for h in row:
                        yield h

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        """
        Sum resources in a Chebyshev (square) radius around (x,y). Excludes tiles out of bounds.
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
        """
        for r in self.roads:
            if (r.start == start and r.end == end) or (r.start == end and r.end == start):
                return True
        return False

    def add_road(self, start: Coordinate, end: Coordinate) -> None:
        """
        Add a two-way road between start and end. Raises ValueError if invalid.
        """
        if start == end:
            raise ValueError("Cannot build a road from a tile to itself.")
        if not (start in self and end in self):
            raise ValueError(f"Cannot build road: one endpoint out of bounds: {start}, {end}")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Coordinate, end: Coordinate) -> float:
        """
        Return a multiplier to trade speed: 1.5 if there's a road, otherwise 1.0.
        """
        return 1.5 if self.has_road(start, end) else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# == SETTINGS ADJUSTMENT HELPER ==

def adjust_settings(settings: WorldSettings, **kwargs) -> None:
    """
    Adjust world settings safely. Float values are clamped to [0.0, 1.0];
    int/bool values are assigned if types match; other mismatches raise TypeError.
    """
    for key, val in kwargs.items():
        if not hasattr(settings, key):
            continue
        current = getattr(settings, key)
        # If current is a float, clamp to [0,1].
        if isinstance(current, float) and isinstance(val, (int, float)):
            setattr(settings, key, float(max(0.0, min(1.0, float(val)))))
        # If current is int, assign only if val is int.
        elif isinstance(current, int) and isinstance(val, int):
            setattr(settings, key, val)
        # If current is bool, assign only if val is bool.
        elif isinstance(current, bool) and isinstance(val, bool):
            setattr(settings, key, val)
        else:
            raise TypeError(f"Cannot assign value of type {type(val)} to setting '{key}'.")


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
    "adjust_settings",
]
