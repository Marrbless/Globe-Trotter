from __future__ import annotations

"""Configuration dataclass for world generation."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WorldSettings:
    seed: int = 0
    width: int = 50
    height: int = 50
    biome_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "plains": 0.3,
            "forest": 0.25,
            "hills": 0.2,
            "desert": 0.15,
            "mountains": 0.05,
            "water": 0.05,
        }
    )
    weather_patterns: Dict[str, float] = field(
        default_factory=lambda: {"rain": 0.3, "dry": 0.5, "snow": 0.2}
    )
    moisture: float = 0.5
    elevation: float = 0.5
    temperature: float = 0.5
    rainfall_intensity: float = 0.5
    disaster_intensity: float = 0.0
    sea_level: float = 0.3
    plate_activity: float = 0.5
    base_height: float = 0.5
    world_changes: bool = True
    mountain_elev: float = 0.8
    hill_elev: float = 0.6
    tundra_temp: float = 0.25
    desert_rain: float = 0.2


__all__ = ["WorldSettings"]
