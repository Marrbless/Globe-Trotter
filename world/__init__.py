from __future__ import annotations

from .export import export_resources_json, export_resources_xml
from .fantasy import (
    add_crystal_forests,
    add_floating_islands,
    add_ley_lines,
    add_mythic_biomes,
    apply_fantasy_overlays,
)
from .generation import (
    _compute_moisture_orographic,
    compute_temperature,
    determine_biome,
    terrain_from_elevation,
)
from .hex import Hex
from .resource_types import ResourceType, STRATEGIC_RESOURCES, LUXURY_RESOURCES
from .settings import WorldSettings
from .world import (
    BIOME_COLORS,
    Road,
    RiverSegment,
    World,
)

# ``adjust_settings`` is provided for backward compatibility as a module-level
# helper that forwards to ``World.adjust_settings``.
def adjust_settings(settings: WorldSettings, world: "World" | None = None, **kwargs):
    return World.adjust_settings(settings, world, **kwargs)

__all__ = [
    "BIOME_COLORS",
    "Hex",
    "LUXURY_RESOURCES",
    "ResourceType",
    "Road",
    "RiverSegment",
    "STRATEGIC_RESOURCES",
    "World",
    "WorldSettings",
    "add_crystal_forests",
    "add_floating_islands",
    "add_ley_lines",
    "add_mythic_biomes",
    "adjust_settings",
    "apply_fantasy_overlays",
    "compute_temperature",
    "determine_biome",
    "export_resources_json",
    "export_resources_xml",
    "terrain_from_elevation",
    "_compute_moisture_orographic",
]
