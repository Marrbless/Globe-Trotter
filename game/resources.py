# Resource management utilities
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from .world import World, Hex, ResourceType

if TYPE_CHECKING:
    from .game import Position, Faction


@dataclass
class ResourceManager:
    """Tracks resource quantities for each faction and handles gathering."""

    world: World
    data: Dict[str, Dict[ResourceType, int]] = field(default_factory=dict)

    def register(self, faction: 'Faction') -> None:
        """Add a faction to be tracked."""
        if faction.name not in self.data:
            self.data[faction.name] = {
                ResourceType.FOOD: 0,
                ResourceType.WOOD: 0,
                ResourceType.STONE: 0,
            }

    def adjacent_tiles(self, pos: Position) -> List[Hex]:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        tiles: List[Hex] = []
        for dq, dr in directions:
            tile = self.world.get(pos.x + dq, pos.y + dr)
            if tile is not None:
                tiles.append(tile)
        return tiles

    def gather_for_faction(self, faction: 'Faction') -> None:
        self.register(faction)
        resources = self.data[faction.name]
        tiles = self.adjacent_tiles(faction.settlement.position)
        terrain_map = {
            "plains": ResourceType.FOOD,
            "hills": ResourceType.FOOD,
            "forest": ResourceType.WOOD,
            "mountains": ResourceType.STONE,
        }
        counts: Dict[ResourceType, int] = {
            ResourceType.FOOD: 0,
            ResourceType.WOOD: 0,
            ResourceType.STONE: 0,
        }

        for tile in tiles:
            res = terrain_map.get(tile.terrain)
            if res:
                counts[res] += 1

        workers = min(faction.workers.assigned, faction.citizens.count)
        for res, count in counts.items():
            gather_amount = min(count, workers)
            resources[res] += gather_amount

    def tick(self, factions: List['Faction']) -> None:
        for faction in factions:
            self.gather_for_faction(faction)
