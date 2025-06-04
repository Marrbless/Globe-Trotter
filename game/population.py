from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Faction


@dataclass
class Citizen:
    """Represents total citizens for a faction."""

    count: int = 0


@dataclass
class Worker:
    """Represents workers assigned to tasks for a faction."""

    assigned: int = 0

    def available(self, total_citizens: int) -> int:
        """Return how many citizens are free to be assigned as workers."""
        return max(total_citizens - self.assigned, 0)


@dataclass
class FactionManager:
    """Manage population counts for all factions."""

    factions: List[Faction] = field(default_factory=list)

    def add_faction(self, faction: Faction) -> None:
        self.factions.append(faction)

    def assign_workers(self, faction: Faction, number: int) -> int:
        """Assign available citizens as workers."""
        available = faction.workers.available(faction.citizens.count)
        to_assign = min(number, available)
        faction.workers.assigned += to_assign
        return to_assign

    def unassign_workers(self, faction: Faction, number: int) -> int:
        """Remove workers from their tasks."""
        to_unassign = min(number, faction.workers.assigned)
        faction.workers.assigned -= to_unassign
        return to_unassign

    def tick(self) -> None:
        """Update population for each faction."""
        for faction in self.factions:
            self._update_population(faction)

    def _update_population(self, faction: Faction) -> None:
        """Apply births, deaths and migration to a faction."""
        citizens = faction.citizens.count
        births = random.randint(0, max(1, citizens // 10))
        deaths = random.randint(0, max(1, citizens // 20))
        migration = random.randint(-2, 2)
        faction.citizens.count = max(0, citizens + births - deaths + migration)
        if faction.workers.assigned > faction.citizens.count:
            faction.workers.assigned = faction.citizens.count
