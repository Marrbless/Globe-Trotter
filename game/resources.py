# Resource management utilities
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from world.world import World, Hex, ResourceType
from . import settings

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
            self.data[faction.name] = faction.resources.copy()

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


        # Worker assignment is now handled entirely by ``FactionManager``.
        # ``ResourceManager`` assumes the ``assigned`` count is already
        # up to date when gathering occurs.

        # Sum resource values for adjacent tiles
        counts: Dict[ResourceType, int] = {}
        for tile in tiles:
            for res_type, amount in tile.resources.items():
                counts[res_type] = counts.get(res_type, 0) + amount

        # Determine how many workers can gather (limited by assigned workers and available citizens)
        workers_available = min(faction.workers.assigned, faction.citizens.count)
        workers_remaining = workers_available

        # Gather from each resource type while tracking remaining workers
        for res_type, amount in counts.items():
            bonus = sum(
                b.resource_bonus
                for b in getattr(faction, "buildings", [])
                if getattr(b, "resource_type", None) == res_type
            )
            # Limit gathered amount by remaining workers plus any building bonus
            gathered = int(min(amount, workers_remaining + bonus) * settings.SCALE_FACTOR)
            gathered = min(amount, workers_remaining + bonus)
            efficiency = getattr(faction, "worker_efficiency", 1.0)
            gathered = int(round(gathered * efficiency))
            if res_type not in resources:
                resources[res_type] = 0
            resources[res_type] += gathered
            # Keep faction's personal resource store in sync
            if res_type not in faction.resources:
                faction.resources[res_type] = 0
            faction.resources[res_type] += gathered

            # Reduce workers remaining based on how many were used for this resource
            workers_used = max(0, gathered - bonus)
            workers_remaining = max(workers_remaining - workers_used, 0)

    def tick(self, factions: List["Faction"]) -> None:
        for faction in factions:
            self.gather_for_faction(faction)
