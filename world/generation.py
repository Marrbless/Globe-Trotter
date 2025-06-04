from __future__ import annotations

"""Advanced world generation helpers."""

import random
from typing import List

from .world import WorldSettings


def _latitude(row: int, height: int) -> float:
    """Return normalized latitude (0 south pole -> 1 north pole)."""
    if height <= 1:
        return 0.5
    return row / float(height - 1)


def compute_temperature(row: int, settings: WorldSettings, rng: random.Random) -> float:
    """Compute temperature influenced by latitude and random climate variation."""
    lat = _latitude(row, settings.height)
    # Base temp peaks at equator (lat=0.5) and drops towards poles
    base = 1.0 - abs(lat - 0.5) * 2
    variation = rng.uniform(-0.1, 0.1) * settings.temperature
    return max(0.0, min(1.0, base + variation))


def generate_temperature_map(settings: WorldSettings, rng: random.Random) -> List[List[float]]:
    """Generate a temperature value for each hex row."""
    temps: List[List[float]] = []
    for r in range(settings.height):
        row = [compute_temperature(r, settings, rng) for _ in range(settings.width)]
        temps.append(row)
    return temps


def generate_rainfall(
    elevation_map: List[List[float]],
    settings: WorldSettings,
    rng: random.Random,
) -> List[List[float]]:
    """Create rainfall map using simple west-to-east moisture transport."""
    rain = [[0.0 for _ in range(settings.width)] for _ in range(settings.height)]
    for r in range(settings.height):
        # initial moisture with slight randomness per row
        moisture = settings.moisture + rng.uniform(-0.1, 0.1)
        for q in range(settings.width):
            elev = elevation_map[r][q]
            precip = max(0.0, moisture * (1.0 - elev))
            rain[r][q] = precip
            # moisture lost proportional to rainfall and elevation blocking
            moisture = max(0.0, moisture - precip * 0.5 - elev * 0.1)
    return rain


def determine_biome(elevation: float, temperature: float, rainfall: float) -> str:
    """Classify biome from elevation, temperature and rainfall values."""
    if elevation > 0.8:
        return "mountains"
    if elevation > 0.6:
        return "hills"

    if temperature < 0.25:
        return "tundra"

    if rainfall < 0.2 and temperature > 0.5:
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
    "compute_temperature",
    "generate_temperature_map",
    "generate_rainfall",
    "determine_biome",
    "generate_biome_map",
]
