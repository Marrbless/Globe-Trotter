from __future__ import annotations

"""Simplified world generation helpers with Perlin noise, orographic moisture, and biome determination."""

import math
import random
from typing import Dict, List, Tuple

from .settings import WorldSettings

# Type aliases
Coordinate = Tuple[int, int]
ElevationCache = Dict[Coordinate, float]

# Predefined colors for visualizing biomes (RGBA)
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


def _stable_hash(*args: int) -> int:
    """
    Deterministic 64-bit hash used for RNG seeding.
    Combines integer inputs into a reproducible 64-bit result.
    """
    x = 0x345678ABCDEF1234
    for a in args:
        a &= 0xFFFFFFFFFFFFFFFF
        a ^= a >> 33
        a = (a * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        a ^= a >> 33
        x ^= a
        x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    return x


def _compute_moisture_orographic(
    *,
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
    season: float,
    settings: WorldSettings,
) -> float:
    """
    Approximate rainfall using a simple west-to-east orographic model.

    Starts with a base moisture (with random jitter and seasonal offset),
    then simulates moisture loss as air rises over elevation from west (x=0) toward current tile (x=q).
    """
    rng = random.Random(_stable_hash(r, seed, 0xCAFE))
    # Base moisture with random jitter
    base = moisture_setting + rng.uniform(-0.1, 0.1)
    # Add seasonal fluctuation
    base += math.sin(2.0 * math.pi * season) * seasonal_amplitude * 0.5
    moisture = max(0.0, min(1.0, base))

    precip = 0.0
    for x in range(q + 1):
        # Look up cached elevation for column x in the same row r
        elev = elevation_cache.get((x, r))
        if elev is None:
            elev = elevation if x == q else 0.0
        # Precipitation at this column
        precip = max(0.0, moisture * (1.0 - elev))
        if x == q:
            break
        # Moisture loss depends on precipitation and elevation, modulated by wind
        loss = (precip * 0.5 + elev * 0.1) * (1.0 - wind_strength)
        moisture = max(0.0, moisture - loss)

    return max(0.0, min(1.0, precip))


def _fade(t: float) -> float:
    """Fade function for Perlin noise interpolation."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t."""
    return a + t * (b - a)


def _grad(ix: int, iy: int, seed: int) -> Tuple[float, float]:
    """
    Generate a pseudorandom gradient vector for integer grid point (ix, iy) using a stable hash.
    """
    rng = random.Random(_stable_hash(ix, iy, seed))
    angle = rng.random() * 2.0 * math.pi
    return math.cos(angle), math.sin(angle)


def _dot_grid_gradient(ix: int, iy: int, x: float, y: float, seed: int) -> float:
    """
    Compute the dot product between the gradient vector at the integer grid point (ix, iy)
    and the distance vector from that grid point to (x, y).
    """
    gx, gy = _grad(ix, iy, seed)
    dx = x - ix
    dy = y - iy
    return gx * dx + gy * dy


def _perlin(x: float, y: float, seed: int) -> float:
    """
    Single-octave Perlin noise at coordinates (x, y) with given seed.
    Returns a value in [-1, 1], which is shifted to [0, 1] by the caller.
    """
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
    # Shift from [-1,1] to [0,1]
    return (value + 1.0) / 2.0


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
    Generate fractal Perlin noise at (x, y) using multiple octaves.
    Returns a normalized value in [0, 1].
    """
    value = 0.0
    amplitude = 1.0
    frequency = scale
    max_amp = 0.0

    for i in range(octaves):
        value += _perlin(x * frequency, y * frequency, seed + i) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return value / max_amp if max_amp > 0 else 0.0


def _apply_tectonic_plates(elev: List[List[float]], settings: WorldSettings) -> None:
    """
    Modify the elevation map in-place to simulate tectonic plate boundaries.
    Creates random plate centers, then adjusts height based on distance to nearest plates.
    """
    width = len(elev[0])
    height = len(elev)
    rng = random.Random(settings.seed)
    plates = max(2, int(3 + settings.plate_activity * 5))
    # Each plate: (center_x, center_y, base_height)
    centers = [
        (rng.randint(0, width - 1), rng.randint(0, height - 1), rng.random())
        for _ in range(plates)
    ]

    for y in range(height):
        for x in range(width):
            # Compute squared distances to all plate centers
            dists = sorted(
                ((cx - x) ** 2 + (cy - y) ** 2, base) for cx, cy, base in centers
            )
            dist0, base = dists[0]
            dist1 = dists[1][0] if len(dists) > 1 else dist0
            ratio = dist0 / (dist0 + dist1) if dist1 > 0 else 0.0
            boundary = 1.0 - abs(0.5 - ratio) * 2.0
            plate_height = base * settings.base_height + boundary * settings.plate_activity
            # Blend existing elevation with plate-influenced height
            elev[y][x] = min(1.0, max(0.0, (elev[y][x] + plate_height) / 2.0))


def terrain_from_elevation(value: float, settings: WorldSettings) -> str:
    """
    Classify a tile as 'water', 'plains', 'hills', or 'mountains' solely based on elevation.
    """
    if value < settings.sea_level:
        return "water"
    if value < settings.sea_level + 0.2:
        return "plains"
    if value < settings.sea_level + 0.4:
        return "hills"
    return "mountains"


def _latitude(row: int, height: int) -> float:
    """
    Compute a normalized latitude (0 at top edge, 1 at bottom edge) for row index.
    """
    return row / float(height - 1) if height > 1 else 0.5


def compute_temperature(
    row: int,
    col: int,
    elevation: float,
    settings: WorldSettings,
    rng: random.Random,
    *,
    season: float = 0.0,
) -> float:
    """
    Compute a normalized temperature [0,1] at grid position (row, col).
    Accounts for latitude, elevation penalty, random variation, wind effect, and seasonal shift.
    """
    lat_norm = _latitude(row, settings.height)
    base = 1.0 - abs(lat_norm - 0.5) * 2.0       # Equator is warmest (0.5)
    base -= elevation * 0.3                     # Higher elevation is colder

    variation = rng.uniform(-0.1, 0.1) * settings.temperature
    wind_effect = 0.0
    if settings.width > 1:
        wind_effect = ((col / float(settings.width - 1)) - 0.5) * settings.wind_strength * 0.2

    seasonal = math.sin(2.0 * math.pi * season) * settings.seasonal_amplitude * 0.5
    temp = base + variation + wind_effect + seasonal
    return max(0.0, min(1.0, temp))





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
    """
    Determine a biome string from elevation, temperature, and rainfall thresholds.
    Order of checks:
      1. High elevation → mountains
      2. Moderate elevation → hills
      3. Low temperature → tundra
      4. Low rainfall + warm → desert
      5. High rainfall + warm → rainforest
      6. Moderate rainfall → forest
      7. Otherwise → plains
    """
    if elevation > mountain_elev:
        return "mountains"
    if elevation > hill_elev:
        return "hills"
    if temperature < tundra_temp:
        return "tundra"
    if rainfall < desert_rain and temperature > 0.5:
        return "desert"
    if rainfall > 0.7 and temperature > 0.5:
        return "rainforest"
    if rainfall > 0.4:
        return "forest"
    return "plains"


__all__ = [
    "_stable_hash",
    "_compute_moisture_orographic",
    "_fade",
    "_lerp",
    "_grad",
    "_dot_grid_gradient",
    "_perlin",
    "perlin_noise",
    "_apply_tectonic_plates",
    "terrain_from_elevation",
    "compute_temperature",
    "determine_biome",
    "BIOME_COLORS",
]
