from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from .game import Faction


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
    # When True, automatically assign idle citizens during each tick for
    # factions that have ``manual_assignment`` disabled.
    auto_assign: bool = True
    assign_strategy: Callable[[Faction], None] | None = None
    strategies: Dict[str, Callable[[Faction], None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.assign_strategy is None:
            self.assign_strategy = self._default_assign_strategy
        if not self.strategies:
            self.strategies = {
                "basic": self._basic_assign_strategy,
                "mid": self._default_assign_strategy,
                "advanced": self._advanced_assign_strategy,
            }

    def add_faction(self, faction: Faction) -> None:
        if faction not in self.factions:
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

    def _default_assign_strategy(self, faction: Faction) -> None:
        """Assign all idle citizens as workers."""
        idle = faction.workers.available(faction.citizens.count)
        faction.workers.assigned += idle

    def _basic_assign_strategy(self, faction: Faction) -> None:
        """Assign half of idle citizens as workers."""
        idle = faction.workers.available(faction.citizens.count)
        faction.workers.assigned += idle // 2

    def _advanced_assign_strategy(self, faction: Faction) -> None:
        """Assign most citizens while keeping some idle for flexibility."""
        idle = faction.workers.available(faction.citizens.count)
        reserve = max(1, idle // 5) if idle > 0 else 0
        faction.workers.assigned += max(idle - reserve, 0)

    def tick(self) -> None:
        """Update population for each faction."""
        for faction in self.factions:
            self._update_population(faction)
            if self.auto_assign and not getattr(faction, "manual_assignment", False):
                level = getattr(faction, "automation_level", "mid")
                strategy = self.strategies.get(level, self.assign_strategy)
                if strategy:
                    strategy(faction)

    def _update_population(self, faction: Faction) -> None:
        """Apply births, deaths and migration to a faction."""
        citizens = faction.citizens.count
        births = random.randint(0, max(1, citizens // 10))
        deaths = random.randint(0, max(1, citizens // 20))
        migration = random.randint(-2, 2)
        faction.citizens.count = max(0, citizens + births - deaths + migration)
        if faction.workers.assigned > faction.citizens.count:
            faction.workers.assigned = faction.citizens.count
