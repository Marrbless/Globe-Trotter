"""Procedural hex map generation utilities."""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import random

# Type alias for coordinates in the hex grid
Coordinate = Tuple[int, int]


@dataclass
class WorldSettings:
    """Configuration values for world generation."""

    seed: int = 0
    width: int = 50
    height: int = 50
    biome_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "plains": 0.4,
            "forest": 0.3,
            "desert": 0.2,
            "mountain": 0.1,
        }
    )
    weather_patterns: Dict[str, float] = field(
        default_factory=lambda: {"rain": 0.3, "dry": 0.5, "snow": 0.2}
    )
    moisture: float = 0.5
    elevation: float = 0.5
    temperature: float = 0.5


@dataclass
class Hex:
    """Represents a single hex tile in the world."""

    coord: Coordinate
    biome: str = "plains"
    elevation: float = 0.0
    moisture: float = 0.0
    temperature: float = 0.0


@dataclass
class World:
    """Collection of hexes generated for a world."""

    settings: WorldSettings
    hexes: Dict[Coordinate, Hex] = field(default_factory=dict)

    def get_hex(self, x: int, y: int) -> Hex:
        return self.hexes.get((x, y))


def initialize_random(settings: WorldSettings) -> random.Random:
    """Create a random generator based on the provided seed."""
    return random.Random(settings.seed)


def generate_terrain_type(rng: random.Random, settings: WorldSettings) -> str:
    """Choose a biome based on distribution weights."""
    biomes = list(settings.biome_distribution.keys())
    weights = list(settings.biome_distribution.values())
    return rng.choices(biomes, weights=weights, k=1)[0]


def generate_hexes(settings: WorldSettings) -> World:
    """Generate all hexes for the world according to settings."""
    rng = initialize_random(settings)
    world = World(settings)

    for y in range(settings.height):
        for x in range(settings.width):
            biome = generate_terrain_type(rng, settings)
            elevation = rng.random() * settings.elevation
            moisture = rng.random() * settings.moisture
            temperature = rng.random() * settings.temperature
            world.hexes[(x, y)] = Hex(
                coord=(x, y),
                biome=biome,
                elevation=elevation,
                moisture=moisture,
                temperature=temperature,
            )
    return world


def adjust_settings(
    settings: WorldSettings,
    *,
    moisture: float | None = None,
    elevation: float | None = None,
    temperature: float | None = None,
) -> None:
    """Adjust world sliders before final generation."""
    if moisture is not None:
        settings.moisture = max(0.0, min(1.0, moisture))
    if elevation is not None:
        settings.elevation = max(0.0, min(1.0, elevation))
    if temperature is not None:
        settings.temperature = max(0.0, min(1.0, temperature))


__all__ = [
    "WorldSettings",
    "Hex",
    "World",
    "generate_hexes",
    "adjust_settings",
]

