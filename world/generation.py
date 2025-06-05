from __future__ import annotations

"""Helper generation utilities for terrain features."""

import math
import random
from typing import Dict

from .settings import WorldSettings


def _compute_moisture_orographic(
    *,
    q: int,
    r: int,
    elevation: float,
    elevation_cache: Dict[tuple[int, int], float],
    width: int,
    height: int,
    seed: int,
    moisture_setting: float,
    wind_strength: float,
    seasonal_amplitude: float,
    season: float = 0.0,
    settings: WorldSettings | None = None,
) -> float:
    """Return a simple moisture value influenced by elevation and wind."""
    lat = float(r) / float(height - 1) if height > 1 else 0.5
    moist = 1.0 - abs(lat - 0.5) * 2.0

    rng = random.Random((q * 73856093) ^ (r * 19349663) ^ seed ^ 0xBADC0DE)
    moist += (rng.random() - 0.5) * 0.2
    moist *= moisture_setting

    wind_dir = 1 if wind_strength >= 0 else -1
    neighbor = (q - wind_dir, r)
    neigh_elev = elevation_cache.get(neighbor, elevation)
    moist += (neigh_elev - elevation) * 0.3 * abs(wind_strength)

    moist += math.sin(2.0 * math.pi * season) * seasonal_amplitude * 0.3
    return max(0.0, min(1.0, moist))


__all__ = ["_compute_moisture_orographic"]
