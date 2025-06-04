# Resource management utilities
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from world.world import World, Hex, ResourceType

if TYPE_CHECKING:
    from .game import Position, Faction


@dataclass
class ResourceManager:
    """Tracks resource quantities for each faction and handles gathering."""

    world: World
    data: Dict[str, Dict[ResourceType, int]] = field(default_factory=dict)

    def register(self, faction: "Faction") -> None:
        """Add a faction to be tracked."""
        if faction.name not in self.data:
            self.data[faction.name] = {
                ResourceType.FOOD: 0,
                ResourceType.WOOD: 0,
                ResourceType.STONE: 0,
                ResourceType.ORE: 0,
                ResourceType.METAL: 0,
                ResourceType.CLOTH: 0,
            }

    def adjacent_tiles(self, pos: "Position") -> List[Hex]:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        tiles: List[Hex] = []
        for dq, dr in directions:
            tile = self.world.get(pos.x + dq, pos.y + dr)
            if tile is not None:
                tiles.append(tile)
        return tiles

    def gather_for_faction(self, faction: "Faction") -> None:
        self.register(faction)
        resources = self.data[faction.name]
        tiles = self.adjacent_tiles(faction.settlement.position)

        # Count how many tiles yield each resource type
        counts: Dict[ResourceType, int] = {}
        for tile in tiles:
            for res_type in tile.resources.keys():
                counts[res_type] = counts.get(res_type, 0) + 1

        # Determine how many workers can gather (limited by assigned workers and available citizens)
        workers_available = min(faction.workers.assigned, faction.citizens.count)

        # Gather from each resource type up to the number of available workers
        for res_type, tile_count in counts.items():
            gathered = min(tile_count, workers_available)
            if res_type not in resources:
                resources[res_type] = 0
            resources[res_type] += gathered

    def tick(self, factions: List["Faction"]) -> None:
        for faction in factions:
            self.gather_for_faction(faction)
