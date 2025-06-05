from __future__ import annotations

"""High-level god powers that players can invoke at great expense."""

from dataclasses import dataclass
from typing import Callable, Dict, TYPE_CHECKING, List

from world.world import ResourceType

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .game import Game, Faction


@dataclass
class GodPower:
    """Representation of a powerful ability unlocked late game."""

    name: str
    cost: Dict[ResourceType, int]
    description: str
    apply_effect: Callable[[Game], None]
    unlock_condition: Callable[[Faction, set[str]], bool]

    def is_unlocked(self, faction: Faction, completed: set[str]) -> bool:
        return self.unlock_condition(faction, completed)

    def apply(self, game: Game) -> None:
        faction = game.player_faction
        if faction is None:
            raise RuntimeError("Player faction not initialized")

        completed = {p.name for p in faction.completed_projects()}
        if not self.is_unlocked(faction, completed):
            raise ValueError(f"{self.name} not unlocked")

        for res, amt in self.cost.items():
            if faction.resources.get(res, 0) < amt:
                raise ValueError(f"Not enough {res.value} for {self.name}")

        for res, amt in self.cost.items():
            faction.resources[res] -= amt
        self.apply_effect(game)


# ---------------------------------------------------------------------------
# Power implementations
# ---------------------------------------------------------------------------

def _summon_harvest_effect(game: Game) -> None:
    faction = game.player_faction
    if faction:
        faction.resources[ResourceType.FOOD] = faction.resources.get(ResourceType.FOOD, 0) + 500


def _summon_harvest_unlock(faction: Faction, completed: set[str]) -> bool:
    wood = faction.resources.get(ResourceType.WOOD, 0)
    stone = faction.resources.get(ResourceType.STONE, 0)
    return wood + stone >= 1000


SUMMON_HARVEST = GodPower(
    name="Summon Harvest",
    cost={ResourceType.WOOD: 300, ResourceType.STONE: 300},
    description="Conjure abundant crops, granting 500 food.",
    apply_effect=_summon_harvest_effect,
    unlock_condition=_summon_harvest_unlock,
)


def _quell_disaster_effect(game: Game) -> None:
    faction = game.player_faction
    if faction:
        faction.citizens.count += 20
        for res in (ResourceType.WOOD, ResourceType.STONE, ResourceType.FOOD):
            faction.resources[res] = faction.resources.get(res, 0) + 100


def _quell_disaster_unlock(faction: Faction, completed: set[str]) -> bool:
    return "Sky Fortress" in completed or "Grand Cathedral" in completed


QUELL_DISASTER = GodPower(
    name="Quell Disaster",
    cost={
        ResourceType.WOOD: 200,
        ResourceType.STONE: 200,
        ResourceType.FOOD: 200,
    },
    description="Stabilize the realm, restoring people and supplies.",
    apply_effect=_quell_disaster_effect,
    unlock_condition=_quell_disaster_unlock,
)


ALL_POWERS: List[GodPower] = [SUMMON_HARVEST, QUELL_DISASTER]

__all__ = ["GodPower", "ALL_POWERS", "SUMMON_HARVEST", "QUELL_DISASTER"]
