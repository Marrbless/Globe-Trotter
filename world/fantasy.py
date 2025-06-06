from __future__ import annotations

"""Fantasy terrain generators like floating islands and crystal forests."""

import random
from typing import Iterable

from .hex import Hex
from .resources import generate_resources


def add_floating_islands(hexes: Iterable[Hex], level: float, *, rng: random.Random | None = None) -> None:
    """Convert some high elevation tiles into floating islands."""
    if level <= 0:
        return
    rng = rng or random.Random(42)
    candidates = [h for h in hexes if h.elevation > 0.6 and h.terrain != "water"]
    if not candidates:
        candidates = [h for h in hexes if h.terrain != "water"]
    count = max(1, int(len(candidates) * 0.05 * level))
    for h in rng.sample(candidates, min(len(candidates), count)):
        h.terrain = "floating_island"
        h.resources = generate_resources(random.Random(h.coord[0] * 73856093 ^ h.coord[1] * 19349663 ^ 0xF17A57), h.terrain)


def add_crystal_forests(hexes: Iterable[Hex], level: float, *, rng: random.Random | None = None) -> None:
    """Transform some forests or plains into crystal forests."""
    if level <= 0:
        return
    rng = rng or random.Random(99)
    candidates = [h for h in hexes if h.terrain in {"forest", "plains"}]
    if not candidates:
        candidates = [h for h in hexes if h.terrain != "water"]
    count = max(1, int(len(candidates) * 0.1 * level))
    for h in rng.sample(candidates, min(len(candidates), count)):
        h.terrain = "crystal_forest"
        h.resources = generate_resources(random.Random(h.coord[0] * 73856093 ^ h.coord[1] * 19349663 ^ 0xC5A1CE), h.terrain)


def add_ley_lines(hexes: Iterable[Hex], level: float, *, rng: random.Random | None = None) -> None:
    """Mark some tiles as intersecting magical ley lines."""
    if level <= 0:
        return
    rng = rng or random.Random(123)
    candidates = [h for h in hexes if h.terrain != "water"]
    if not candidates:
        return
    count = max(1, int(len(candidates) * 0.03 * level))
    for h in rng.sample(candidates, min(len(candidates), count)):
        h.ley_line = True


def add_mythic_biomes(hexes: Iterable[Hex], level: float, *, rng: random.Random | None = None) -> None:
    """Transform some forests or plains into faerie forests."""
    if level <= 0:
        return
    rng = rng or random.Random(777)
    candidates = [h for h in hexes if h.terrain in {"forest", "plains"}]
    if not candidates:
        candidates = [h for h in hexes if h.terrain != "water"]
    count = max(1, int(len(candidates) * 0.05 * level))
    for h in rng.sample(candidates, min(len(candidates), count)):
        h.terrain = "faerie_forest"
        h.resources = generate_resources(random.Random(h.coord[0] * 73856093 ^ h.coord[1] * 19349663 ^ 0xFAE1), h.terrain)


def apply_fantasy_overlays(hexes: Iterable[Hex], level: float) -> None:
    """Apply all fantasy overlays based on the given level."""
    if level <= 0:
        return
    hex_list = list(hexes)
    add_floating_islands(hex_list, level)
    add_crystal_forests(hex_list, level)
    add_ley_lines(hex_list, level)
    add_mythic_biomes(hex_list, level)


__all__ = [
    "add_floating_islands",
    "add_crystal_forests",
    "add_ley_lines",
    "add_mythic_biomes",
    "apply_fantasy_overlays",
]
