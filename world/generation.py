from __future__ import annotations

"""Procedural world generation helpers including climate and biome classification."""

import random
import math
from typing import List, TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .settings import WorldSettings


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
}


# -- Noise utilities ---------------------------------------------------------


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(ix: int, iy: int, seed: int) -> tuple[float, float]:
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


# -- Elevation map generation ------------------------------------------------


def generate_elevation_map(
    width: int,
    height: int,
    settings: WorldSettings,
) -> List[List[float]]:
    """Return a 2D list of elevation values in range [0, 1]."""
    elev: List[List[float]] = []

    for y in range(height):
        row: List[float] = []
        for x in range(width):
            n = perlin_noise(x, y, settings.seed)
            # Adjust elevation based on the requested overall level.
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


# -- Climate and biome utilities ------------------------------------------------


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
            row.append(compute_temperature(r, q, elev, settings, rng, season=season))
        temps.append(row)
    return temps


def generate_rainfall(
    elevation_map: List[List[float]],
    settings: WorldSettings,
    rng: random.Random,
    *,
    season: float = 0.0,
) -> List[List[float]]:
    """Create rainfall map using simple west-to-east moisture transport."""
    rain: List[List[float]] = [
        [0.0 for _ in range(settings.width)] for _ in range(settings.height)
    ]

    for r in range(settings.height):
        # initial moisture with slight randomness per row and season adjustment
        base = settings.moisture + rng.uniform(-0.1, 0.1)
        base += math.sin(2 * math.pi * season) * settings.seasonal_amplitude * 0.5
        moisture = max(0.0, min(1.0, base))
        for q in range(settings.width):
            elev = elevation_map[r][q]
            precip = max(0.0, moisture * (1.0 - elev))
            rain[r][q] = precip
            # moisture lost proportional to rainfall and elevation blocking
            loss = (precip * 0.5 + elev * 0.1) * (1.0 - settings.wind_strength)
            moisture = max(0.0, moisture - loss)

    return rain


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
    """Classify biome from elevation, temperature, and rainfall values."""
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


def generate_biome_map(
    elevation_map: List[List[float]],
    temperature_map: List[List[float]],
    rainfall_map: List[List[float]],
) -> List[List[str]]:
    """Return biome classification for each hex."""
    height = len(elevation_map)
    width = len(elevation_map[0]) if height else 0
    biomes: List[List[str]] = []

    for r in range(height):
        row: List[str] = []
        for q in range(width):
            row.append(
                determine_biome(
                    elevation_map[r][q],
                    temperature_map[r][q],
                    rainfall_map[r][q],
                )
            )
        biomes.append(row)

    return biomes


__all__ = [
    "generate_elevation_map",
    "apply_tectonic_plates",
    "terrain_from_elevation",
    "perlin_noise",
    "compute_temperature",
    "generate_temperature_map",
    "generate_rainfall",
    "determine_biome",
    "generate_biome_map",
    "BIOME_COLORS",
]
