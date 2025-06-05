from __future__ import annotations

"""World generation and management utilities with enhanced realism and fantasy integration."""

import random
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import functools

from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .resources import generate_resources
from .hex import Hex, Coordinate
from .settings import WorldSettings
from .fantasy import apply_fantasy_overlays

# -- Visualization -----------------------------------------------------------

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

# -- Noise utilities ---------------------------------------------------------

def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

@functools.lru_cache(maxsize=None)
def _grad(ix: int, iy: int, seed: int) -> Tuple[float, float]:
    """Return a pseudo-random gradient vector for the grid point."""
    rnd = random.Random(hash((ix, iy, seed)))
    angle = rnd.random() * 2 * math.pi
    return math.cos(angle), math.sin(angle)

def _dot_grid_gradient(ix: int, iy: int, x: float, y: float, seed: int) -> float:
    gx, gy = _grad(ix, iy, seed)
    dx = x - ix
    dy = y - iy
    return gx * dx + gy * dy

def _perlin(x: float, y: float, seed: int) -> float:
    """Classic 2D Perlin noise in range [0, 1]."""
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = _fade(x - x0)
    sy = _fade(y - y0)

    n00 = _dot_grid_gradient(x0, y0, x, y, seed)
    n10 = _dot_grid_gradient(x1, y0, x, y, seed)
    n01 = _dot_grid_gradient(x0, y1, x, y, seed)
    n11 = _dot_grid_gradient(x1, y1, x, y, seed)

    ix0 = _lerp(n00, n10, sx)
    ix1 = _lerp(n01, n11, sx)
    value = _lerp(ix0, ix1, sy)
    return (value + 1) / 2

def perlin_noise(
    x: float,
    y: float,
    seed: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    scale: float = 0.05,
) -> float:
    """Generate fractal Perlin noise value for given coordinates."""
    value = 0.0
    amplitude = 1.0
    frequency = scale
    max_amp = 0.0

    for i in range(octaves):
        value += _perlin(x * frequency, y * frequency, seed + i) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_amp

# -- Elevation map generation (legacy, kept for external tools) --------------

def generate_elevation_map(
    width: int,
    height: int,
    settings: WorldSettings,
) -> List[List[float]]:
    """Return a 2D list of elevation values in range [0, 1], using same logic as World."""
    elev: List[List[float]] = []
    for y in range(height):
        row: List[float] = []
        for x in range(width):
            n = perlin_noise(x, y, settings.seed, scale=settings.elevation)
            amp = 0.5 + settings.elevation / 2
            offset = settings.elevation - 0.5
            val = max(0.0, min(1.0, n * amp + offset))
            row.append(val)
        elev.append(row)
    apply_tectonic_plates(elev, settings)
    return elev

def apply_tectonic_plates(
    elev: List[List[float]],
    settings: WorldSettings,
) -> None:
    """Modify elevation map in place to create continents and mountains."""
    width = len(elev[0])
    height = len(elev)
    rng = random.Random(settings.seed)

    plates = max(2, int(3 + settings.plate_activity * 5))
    centers = [
        (rng.randint(0, width - 1), rng.randint(0, height - 1), rng.random())
        for _ in range(plates)
    ]

    for y in range(height):
        for x in range(width):
            dists = sorted(
                (
                    (cx - x) ** 2 + (cy - y) ** 2,
                    base,
                )
                for cx, cy, base in centers
            )
            dist0, base = dists[0]
            dist1 = dists[1][0] if len(dists) > 1 else dist0
            ratio = dist0 / (dist0 + dist1) if dist1 > 0 else 0.0
            boundary = 1.0 - abs(0.5 - ratio) * 2.0
            plate_height = (
                base * settings.base_height + boundary * settings.plate_activity
            )
            elev[y][x] = min(1.0, max(0.0, (elev[y][x] + plate_height) / 2))

def terrain_from_elevation(
    value: float,
    settings: WorldSettings,
) -> str:
    """Convert elevation value to a terrain type."""
    if value < settings.sea_level:
        return "water"
    if value < settings.sea_level + 0.2:
        return "plains"
    if value < settings.sea_level + 0.4:
        return "hills"
    return "mountains"

# -- Climate and biome utilities ---------------------------------------------

def _latitude(row: int, height: int) -> float:
    """Return normalized latitude (0 south pole -> 1 north pole)."""
    if height <= 1:
        return 0.5
    return row / float(height - 1)

def compute_temperature(
    row: int,
    col: int,
    elevation: float,
    settings: WorldSettings,
    rng: random.Random,
    *,
    season: float = 0.0,
) -> float:
    """Compute temperature influenced by latitude, elevation, winds and season."""
    lat = _latitude(row, settings.height)
    base = 1.0 - abs(lat - 0.5) * 2
    base -= elevation * 0.3
    variation = rng.uniform(-0.1, 0.1) * settings.temperature
    wind_effect = (
        (col / float(settings.width - 1) if settings.width > 1 else 0.5) - 0.5
    ) * settings.wind_strength * 0.2
    seasonal = math.sin(2 * math.pi * season) * settings.seasonal_amplitude * 0.5
    return max(0.0, min(1.0, base + variation + wind_effect + seasonal))

def _compute_moisture_with_orographic(
    q: int,
    r: int,
    elevation: float,
    settings: WorldSettings,
    season: float = 0.0,
) -> float:
    """
    Compute moisture using west-to-east transport with orographic (windward) effects.
    """
    rng_tile = random.Random(hash((r, settings.seed, "rain")))
    base = settings.moisture + rng_tile.uniform(-0.1, 0.1)
    base += math.sin(2 * math.pi * season) * settings.seasonal_amplitude * 0.5
    moisture = max(0.0, min(1.0, base))

    prev_elev = elevation  # placeholder, overwritten for x=0
    precip = 0.0
    for x in range(q + 1):
        elev_local = elevation if x == q else None
        # For earlier x, fetch elevation via caching if available; placeholder behavior in precompute
        if x != q:
            elev_local = settings._elevation_cache.get((x, r), None)
            if elev_local is None:
                elev_local = elevation  # fallback

        # Orographic blocking: if current elev_local is significantly higher than prev_elev
        orographic_threshold = getattr(settings, "orographic_threshold", 0.1)
        orographic_factor = getattr(settings, "orographic_factor", 0.5)
        if elev_local is not None and elev_local - prev_elev > orographic_threshold:
            moisture = max(0.0, moisture - (elev_local - prev_elev) * orographic_factor)

        precip = max(0.0, moisture * (1.0 - elev_local))
        if x == q:
            break
        loss = (precip * 0.5 + elev_local * 0.1) * (1.0 - settings.wind_strength)
        moisture = max(0.0, moisture - loss)
        prev_elev = elev_local

    return max(0.0, min(1.0, precip))

def generate_temperature_map(
    elevation_map: List[List[float]],
    settings: WorldSettings,
    rng: random.Random,
    *,
    season: float = 0.0,
) -> List[List[float]]:
    """Generate a temperature value for each hex."""
    temps: List[List[float]] = []
    for r in range(settings.height):
        row: List[float] = []
        for q in range(settings.width):
            elev = elevation_map[r][q]
            rng_tile = random.Random(hash((q, r, settings.seed, "temp")))
            row.append(compute_temperature(r, q, elev, settings, rng_tile, season=season))
        temps.append(row)
    return temps

def generate_rainfall(
    elevation_map: List[List[float]],
    settings: WorldSettings,
    rng: random.Random,
    *,
    season: float = 0.0,
) -> List[List[float]]:
    """Create rainfall map using orographic moisture transport from west to east."""
    rain: List[List[float]] = [
        [0.0 for _ in range(settings.width)] for _ in range(settings.height)
    ]

    for r in range(settings.height):
        base = settings.moisture + rng.uniform(-0.1, 0.1)
        base += math.sin(2 * math.pi * season) * settings.seasonal_amplitude * 0.5
        moisture = max(0.0, min(1.0, base))
        for q in range(settings.width):
            elev = elevation_map[r][q]
            # Leverage orographic moisture computation
            precip = _compute_moisture_with_orographic(q, r, elev, settings, season=season)
            rain[r][q] = precip
    return rain

def determine_biome(
    elevation: float,
    temperature: float,
    rainfall: float,
    rng: random.Random,
    *,
    mountain_elev: float = 0.8,
    hill_elev: float = 0.6,
    tundra_temp: float = 0.25,
    desert_rain: float = 0.2,
    fantasy_level: float = 0.0,
) -> str:
    """
    Classify biome from elevation, temperature, rainfall, and fantasy influence.
    Uses provided rng for deterministic outcomes.
    """
    # Realistic base biome
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

    # Fantasy overrides
    if fantasy_level > 0:
        # Chance to become a crystal forest if fantasy_level high and elevation moderate
        if base_biome == "forest" and elevation > 0.4 and fantasy_level > 0.6:
            return "crystal_forest"
        # Floating islands in mountainous areas if fantasy is extreme
        if base_biome == "mountains" and fantasy_level > 0.8 and rng.random() < fantasy_level * 0.1:
            return "floating_island"
    return base_biome

def smooth_biome_map(
    biomes: List[List[str]],
    width: int,
    height: int,
    iterations: int = 1,
) -> List[List[str]]:
    """
    Smooth the biome map by replacing isolated cells that differ from the majority of their neighbors.
    """
    def majority_neighbor(q: int, r: int) -> str:
        counts: Dict[str, int] = {}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if 0 <= nq < width and 0 <= nr < height:
                neighbor_biome = biomes[nr][nq]
                counts[neighbor_biome] = counts.get(neighbor_biome, 0) + 1
        if not counts:
            return biomes[r][q]
        majority = max(counts.items(), key=lambda item: item[1])[0]
        return majority

    for _ in range(iterations):
        new_map = [row.copy() for row in biomes]
        for r in range(height):
            for q in range(width):
                current = biomes[r][q]
                majority = majority_neighbor(q, r)
                if majority != current:
                    new_map[r][q] = majority
        biomes = new_map
    return biomes

def generate_biome_map(
    elevation_map: List[List[float]],
    temperature_map: List[List[float]],
    rainfall_map: List[List[float]],
    settings: WorldSettings,
) -> List[List[str]]:
    """Return biome classification for each hex, including fantasy influence and regional variation."""
    height = len(elevation_map)
    width = len(elevation_map[0]) if height else 0
    biomes: List[List[str]] = []

    region_size = getattr(settings, "biome_region_size", 10)
    for r in range(height):
        row: List[str] = []
        for q in range(width):
            # Regional seed ensures clusters of similar biome variation
            region_coord = (q // region_size, r // region_size)
            region_seed = hash((region_coord[0], region_coord[1], settings.seed, "region"))
            # Local tile seed combines region_seed for regional cohesion
            tile_seed = hash((region_seed, q, r))
            rng_tile = random.Random(tile_seed)
            biome = determine_biome(
                elevation_map[r][q],
                temperature_map[r][q],
                rainfall_map[r][q],
                rng_tile,
                mountain_elev=settings.mountain_elev,
                hill_elev=settings.hill_elev,
                tundra_temp=settings.tundra_temp,
                desert_rain=settings.desert_rain,
                fantasy_level=settings.fantasy_level,
            )
            row.append(biome)
        biomes.append(row)

    # Apply smoothing pass(s) to create gradual transitions
    smoothing_iterations = getattr(settings, "biome_smoothing_iterations", 1)
    if smoothing_iterations > 0:
        biomes = smooth_biome_map(biomes, width, height, smoothing_iterations)

    return biomes

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
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        *,
        seed: int = 0,
        settings: Optional[WorldSettings] = None,
    ) -> None:
        self.settings = settings or WorldSettings(seed=seed, width=width, height=height)
        # Allow chunk size override in settings (fallback to 10)
        self.chunk_size = getattr(self.settings, "chunk_size", 10)
        # Maximum active chunks to keep in memory (oldest evicted first)
        self.max_active_chunks = getattr(self.settings, "max_active_chunks", 100)
        self.chunks: OrderedDict[Tuple[int, int], List[List[Hex]]] = OrderedDict()
        self.roads: List[Road] = []
        self.rivers: List[RiverSegment] = []
        self.lakes: List[Coordinate] = []
        self.rng = initialize_random(self.settings)
        self.season = 0.0

        # Precompute tectonic plate centers for plate height calculations
        self._plate_centers = self._init_plates()
        # Precompute and cache terrain layers (elevation, temperature, moisture, biome)
        self._precompute_terrain_layers()
        # Generate rivers & lakes
        self._generate_rivers()
        # Apply fantasy overlays if enabled
        if self.settings.fantasy_level > 0:
            apply_fantasy_overlays(self.all_hexes(), self.settings.fantasy_level)

    def __contains__(self, coord: Coordinate) -> bool:
        """Allow `coord in world` checks."""
        q, r = coord
        if self.settings.infinite:
            return True
        return 0 <= q < self.settings.width and 0 <= r < self.settings.height

    def __getitem__(self, coord: Coordinate) -> Optional[Hex]:
        """Allow `world[q, r]` access as shorthand for `world.get(q, r)`."""
        return self.get(*coord)

    def hex_neighbors(self, hex_tile: Hex) -> List[Hex]:
        """Return actual Hex neighbors for a given Hex."""
        q, r = hex_tile.coord
        neighbors: List[Hex] = []
        for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            neighbor = self.get(q + dq, r + dr)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors

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
        """Compute elevation combining Perlin noise and plate tectonics."""
        base = self._noise_value(q, r, 0, self.settings.elevation)
        plate = self._plate_height(q, r)
        return max(0.0, min(1.0, (base + plate) / 2))

    def _temperature(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        lat = r / float(self.settings.height - 1) if self.settings.height > 1 else 0.5
        base = 1.0 - abs(lat - 0.5) * 2
        base -= elevation * 0.3
        rng_tile = random.Random(hash((r, self.settings.seed, "temp_local")))
        variation = rng_tile.uniform(-0.1, 0.1) * self.settings.temperature
        wind_effect = ((q / float(self.settings.width - 1) if self.settings.width > 1 else 0.5) - 0.5) * self.settings.wind_strength * 0.2
        seasonal = math.sin(2 * math.pi * season) * self.settings.seasonal_amplitude * 0.5
        return max(0.0, min(1.0, base + variation + wind_effect + seasonal))

    def _moisture(self, q: int, r: int, elevation: float, season: float = 0.0) -> float:
        """
        Compute moisture using west-to-east transport with orographic effects.
        Relies on cached elevation values to avoid recomputing elevation during river generation.
        """
        return _compute_moisture_with_orographic(q, r, elevation, self.settings, season=season)

    def _precompute_terrain_layers(self) -> None:
        """
        Generate and cache elevation, temperature, moisture, and biome for each tile
        before rivers and lakes. Stored in dicts to avoid re-calculation.
        """
        self._elevation_cache: Dict[Coordinate, float] = {}
        self._temperature_cache: Dict[Coordinate, float] = {}
        self._moisture_cache: Dict[Coordinate, float] = {}
        self._biome_cache: Dict[Coordinate, str] = {}

        # First pass: compute elevation, temperature, moisture
        for r in range(self.settings.height):
            for q in range(self.settings.width):
                elev = self._elevation(q, r)
                temp = self._temperature(q, r, elev, self.season)
                moist = self._moisture(q, r, elev, self.season)
                self._elevation_cache[(q, r)] = elev
                self._temperature_cache[(q, r)] = temp
                self._moisture_cache[(q, r)] = moist

        # Generate biome map based on cached values
        elev_map = [
            [self._elevation_cache[(q, r)] for q in range(self.settings.width)]
            for r in range(self.settings.height)
        ]
        temp_map = [
            [self._temperature_cache[(q, r)] for q in range(self.settings.width)]
            for r in range(self.settings.height)
        ]
        rain_map = [
            [self._moisture_cache[(q, r)] for q in range(self.settings.width)]
            for r in range(self.settings.height)
        ]

        biomes = generate_biome_map(elev_map, temp_map, rain_map, self.settings)
        for r in range(self.settings.height):
            for q in range(self.settings.width):
                self._biome_cache[(q, r)] = biomes[r][q]

    def _generate_hex(self, q: int, r: int) -> Hex:
        """Generate or retrieve a hex tile using cached terrain layers."""
        elev = self._elevation_cache[(q, r)]
        temp = self._temperature_cache[(q, r)]
        moist = self._moisture_cache[(q, r)]
        terrain = self._biome_cache[(q, r)]
        rng_tile = random.Random(hash((q, r, self.settings.seed)))
        resources = generate_resources(rng_tile, terrain)

        return Hex(
            coord=(q, r),
            terrain=terrain,
            elevation=elev,
            temperature=temp,
            moisture=moist,
            resources=resources,
        )

    def _generate_chunk(self, cx: int, cy: int) -> None:
        """Populate a chunk of size self.chunk_size × self.chunk_size, then manage cache."""
        chunk: List[List[Hex]] = []
        base_q, base_r = cx * self.chunk_size, cy * self.chunk_size
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

        for r_off in range(rows):
            row: List[Hex] = []
            for q_off in range(cols):
                q, r = base_q + q_off, base_r + r_off
                if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
                    continue
                row.append(self._generate_hex(q, r))
            if row:
                chunk.append(row)

        # Insert at end as most recently used
        self.chunks[(cx, cy)] = chunk
        # If exceeding max_active_chunks, evict the oldest
        if len(self.chunks) > self.max_active_chunks:
            self.chunks.popitem(last=False)

    def get(self, q: int, r: int) -> Optional[Hex]:
        """Retrieve a Hex if within bounds; generate its chunk if needed."""
        if not self.settings.infinite and not (0 <= q < self.settings.width and 0 <= r < self.settings.height):
            return None
        cx, cy = q // self.chunk_size, r // self.chunk_size
        if (cx, cy) not in self.chunks:
            self._generate_chunk(cx, cy)
        # Move accessed chunk to the end to mark as recently used
        self.chunks.move_to_end((cx, cy))
        chunk = self.chunks.get((cx, cy))
        row_idx, col_idx = r % self.chunk_size, q % self.chunk_size
        if not chunk or row_idx >= len(chunk) or col_idx >= len(chunk[row_idx]):
            return None
        return chunk[row_idx][col_idx]

    def _neighbors(self, q: int, r: int) -> List[Coordinate]:
        """
        Return axial hex neighbor coordinates (pointy-topped layout).
        Directions: E, W, SE, NW, NE, SW as (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        if self.settings.infinite:
            return [(q + dq, r + dr) for dq, dr in directions]
        return [
            (q + dq, r + dr)
            for dq, dr in directions
            if 0 <= q + dq < self.settings.width and 0 <= r + dr < self.settings.height
        ]

    def _downhill_neighbor(self, q: int, r: int) -> Optional[Coordinate]:
        """Return the neighbor coordinate with strictly lower elevation, if any."""
        current_elev = self._elevation_cache.get((q, r), None)
        if current_elev is None:
            return None
        best_coord: Optional[Coordinate] = None
        best_elev = current_elev
        for nq, nr in self._neighbors(q, r):
            neighbor_elev = self._elevation_cache.get((nq, nr), None)
            if neighbor_elev is not None and neighbor_elev < best_elev:
                best_elev = neighbor_elev
                best_coord = (nq, nr)
        return best_coord

    def _generate_rivers(self) -> None:
        """
        Generate rivers with branching and lake outflow:
        1. Compute initial flow from cached moisture.
        2. Accumulate downstream, allowing tributaries to merge.
        3. Introduce random branching with cycle prevention.
        4. Mark river segments and lakes, then handle lake outflow.
        """
        rainfall: Dict[Coordinate, float] = {}
        flow: Dict[Coordinate, float] = {}
        downhill: Dict[Coordinate, Optional[Coordinate]] = {}

        # Initial pass: assign rainfall & downhill neighbor
        for r in range(self.settings.height):
            for q in range(self.settings.width):
                coord = (q, r)
                rain_amount = self._moisture_cache[coord] * self.settings.rainfall_intensity
                rainfall[coord] = rain_amount
                flow[coord] = rain_amount
                dn = self._downhill_neighbor(q, r)
                if dn and self._elevation_cache.get(dn, 1.0) < self._elevation_cache[coord]:
                    downhill[coord] = dn
                else:
                    downhill[coord] = None

        # Sort coordinates by descending elevation for accumulation
        coords_sorted = sorted(
            flow.keys(),
            key=lambda c: self._elevation_cache[c],
            reverse=True,
        )

        visited: set[Coordinate] = set()
        for c in coords_sorted:
            if c in visited:
                continue
            d = downhill[c]
            if d and d in flow:
                flow[d] += flow[c]
                visited.add(c)
                # 20% chance for a tributary branch if flow is high
                if flow[c] > self.settings.river_branch_threshold * self.settings.rainfall_intensity:
                    neighbor_list = self._neighbors(*c)
                    second_best: Optional[Coordinate] = None
                    second_elev = self._elevation_cache[c]
                    for n in neighbor_list:
                        if downhill[c] != n and self._elevation_cache.get(n, 1.0) < second_elev:
                            second_elev = self._elevation_cache[n]
                            second_best = n
                    if second_best:
                        branch_chance = random.Random(hash((c, self.settings.seed, "branch"))).random()
                        if branch_chance < 0.3 and second_best not in visited:
                            flow[second_best] += flow[c] * 0.3
                            # Mark second_best as visited for this pass to avoid cycles
                            visited.add(second_best)

        # Assign water_flow to hexes
        for coord, f in flow.items():
            h = self.get(*coord)
            if h:
                h.water_flow = f

        avg_flow = sum(flow.values()) / len(flow) if flow else 0.0
        river_threshold = max(0.05 * self.settings.rainfall_intensity, avg_flow * 2)
        lake_threshold = max(0.1 * self.settings.rainfall_intensity, avg_flow * 4)

        # Identify rivers and lakes
        for coord in coords_sorted:
            d = downhill[coord]
            hex_c = self.get(*coord)
            if not hex_c:
                continue
            if d:
                hex_d = self.get(*d)
                if hex_d and flow[coord] >= river_threshold:
                    self.rivers.append(RiverSegment(coord, d))
                    hex_c.river = True
                    hex_d.river = True
            else:
                if flow[coord] > lake_threshold and not hex_c.lake:
                    self.lakes.append(coord)
                    hex_c.lake = True
                    hex_c.terrain = "water"
                    rng_tile = random.Random(hash((coord, self.settings.seed, "water")))
                    hex_c.resources = generate_resources(rng_tile, "water")

        # Lake outflow: for each lake, if neighbor lower, create a river from lake
        for lake_coord in list(self.lakes):
            q, r = lake_coord
            lake_elev = self._elevation_cache[lake_coord]
            lowest_neighbor: Optional[Coordinate] = None
            lowest_elev = lake_elev
            for n in self._neighbors(q, r):
                neigh_elev = self._elevation_cache.get(n, None)
                if neigh_elev is not None and neigh_elev < lowest_elev:
                    lowest_elev = neigh_elev
                    lowest_neighbor = n
            if lowest_neighbor:
                hex_lake = self.get(*lake_coord)
                hex_out = self.get(*lowest_neighbor)
                if hex_lake and hex_out:
                    self.rivers.append(RiverSegment(lake_coord, lowest_neighbor))
                    hex_lake.river = True
                    hex_out.river = True

    def all_hexes(self) -> Iterable[Hex]:
        """Yield every generated Hex in the world (useful for overlays or iteration)."""
        for r in range(self.settings.height):
            for q in range(self.settings.width):
                h = self.get(q, r)
                if h:
                    yield h

    def resources_near(self, x: int, y: int, radius: int = 1) -> Dict[ResourceType, int]:
        """Sum resources of nearby hexes within a given radius."""
        totals = {r: 0 for r in ResourceType}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                coord = (x + dx, y + dy)
                h = self.get(*coord)
                if h:
                    for rtype, amt in h.resources.items():
                        totals[rtype] += amt
        return {rtype: amt for rtype, amt in totals.items() if amt > 0}

    def has_road(self, start: Coordinate, end: Coordinate) -> bool:
        return any(
            (r.start, r.end) == (start, end) or (r.start, r.end) == (end, start)
            for r in self.roads
        )

    def add_road(self, start: Coordinate, end: Coordinate) -> None:
        if start == end or not (start in self and end in self):
            raise ValueError("Invalid road endpoints")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Coordinate, end: Coordinate) -> float:
        return 1.5 if self.has_road(start, end) else 1.0

def adjust_settings(settings: WorldSettings, **kwargs) -> None:
    """
    Adjust world settings safely by clamping any float values to [0.0, 1.0].
    """
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
    "perlin_noise",
    "generate_elevation_map",
    "apply_tectonic_plates",
    "terrain_from_elevation",
    "compute_temperature",
    "generate_temperature_map",
    "generate_rainfall",
    "determine_biome",
    "generate_biome_map",
    "BIOME_COLORS",
    "STRATEGIC_RESOURCES",
    "LUXURY_RESOURCES",
]
