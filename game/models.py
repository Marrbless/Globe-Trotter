from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from world.world import ResourceType
from .buildings import Building
from .population import Citizen, Worker
from .technology import TechLevel
from . import settings

if TYPE_CHECKING:
    from .diplomacy import TradeDeal
    from .game import Game


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Settlement:
    name: str
    position: Position


@dataclass
class GreatProject:
    """High-cost project that requires multiple turns to complete."""

    name: str
    build_time: int
    victory_points: int = 0
    bonus: str = ""
    progress: int = 0
    bonus_applied: bool = False

    def is_complete(self) -> bool:
        return self.progress >= self.build_time

    def advance(self, amount: int = 1) -> None:
        self.progress = min(self.build_time, self.progress + amount)


@dataclass
class Faction:
    """Core data model representing a faction in the game."""

    name: str
    settlement: Settlement
    citizens: Citizen = field(default_factory=lambda: Citizen(count=10))
    resources: Dict[ResourceType, int] = field(
        default_factory=lambda: {
            ResourceType.FOOD: 100,
            ResourceType.WOOD: 50,
            ResourceType.STONE: 30,
            ResourceType.ORE: 0,
            ResourceType.METAL: 0,
            ResourceType.CLOTH: 0,
            ResourceType.WHEAT: 0,
            ResourceType.FLOUR: 0,
            ResourceType.BREAD: 0,
            ResourceType.WOOL: 0,
            ResourceType.CLOTHES: 0,
            ResourceType.PLANK: 0,
            ResourceType.STONE_BLOCK: 0,
            ResourceType.VEGETABLE: 0,
            ResourceType.SOUP: 0,
            ResourceType.WEAPON: 0,
        }
    )
    workers: Worker = field(default_factory=lambda: Worker(assigned=10))
    buildings: List[Building] = field(default_factory=list)
    projects: List[GreatProject] = field(default_factory=list)
    total_roads: int = 0
    city_count: int = 0
    army_size: int = 0
    unlocked_actions: List[str] = field(default_factory=list)
    manual_assignment: bool = False
    automation_level: str = "mid"
    tech_level: TechLevel = TechLevel.PRIMITIVE
    research_points: int = 0

    def toggle_manual_assignment(self, manual: bool, level: str | None = None) -> None:
        self.manual_assignment = manual
        if not manual and level is not None:
            self.automation_level = level

    @property
    def population(self) -> int:
        return self.citizens.count

    @population.setter
    def population(self, value: int) -> None:
        self.citizens.count = value

    def start_project(self, project: GreatProject, claimed_projects: set[str]) -> None:
        if project.name in claimed_projects:
            raise ValueError(f"{project.name} already claimed")
        claimed_projects.add(project.name)
        self.projects.append(project)

    def progress_projects(self) -> None:
        from .game import apply_project_bonus

        for proj in self.projects:
            if not proj.is_complete():
                proj.advance()
            if proj.is_complete() and not proj.bonus_applied:
                apply_project_bonus(self, proj)
                proj.bonus_applied = True

    def completed_projects(self) -> List[GreatProject]:
        return [p for p in self.projects if p.is_complete()]

    def get_victory_points(self, game: "Game" | None = None) -> int:
        """Calculate total victory points for this faction."""
        # Base score from number of factions in the game
        total = (settings.AI_FACTION_COUNT + 1) * 10

        # Points from settlements and cities
        total += 1  # home settlement
        total += self.city_count * 2

        # Buildings and completed projects
        total += sum(b.victory_points for b in self.buildings)
        total += sum(p.victory_points for p in self.completed_projects())

        # Bonus for longest road and largest army
        if game is not None and game.map.factions:
            longest = max(f.total_roads for f in game.map.factions)
            if self.total_roads == longest and longest > 0:
                total += 5

            largest_army = max(f.army_size for f in game.map.factions)
            if self.army_size == largest_army and largest_army > 0:
                total += 5

        return total

    def progress_research(self, amount: int = 1) -> None:
        """Advance research and update tech level when thresholds are met."""
        if self.tech_level is TechLevel.INDUSTRIAL:
            return
        self.research_points += amount
        thresholds = {
            TechLevel.PRIMITIVE: 10,
            TechLevel.MEDIEVAL: 20,
        }
        needed = thresholds.get(self.tech_level)
        if needed is not None and self.research_points >= needed:
            self.research_points -= needed
            if self.tech_level is TechLevel.PRIMITIVE:
                self.tech_level = TechLevel.MEDIEVAL
            elif self.tech_level is TechLevel.MEDIEVAL:
                self.tech_level = TechLevel.INDUSTRIAL

    def build_structure(self, building: Building) -> None:
        if building.tech_level.value > self.tech_level.value:
            raise ValueError(
                f"{building.name} requires {building.tech_level.name} technology"
            )
        cost: Dict[ResourceType, int] = building.construction_cost
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to build {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        self.buildings.append(building)

    def upgrade_structure(self, building: Building) -> None:
        if building.tech_level.value > self.tech_level.value:
            raise ValueError(
                f"{building.name} requires {building.tech_level.name} technology"
            )
        cost: Dict[ResourceType, int] = building.upgrade_cost()
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to upgrade {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        building.upgrade()

    def transfer_resources(self, other: "Faction", resources: Dict[ResourceType, int]) -> None:
        for res, amt in resources.items():
            if self.resources.get(res, 0) >= amt:
                self.resources[res] -= amt
                other.resources[res] = other.resources.get(res, 0) + amt

    def form_trade_deal(
        self,
        other: "Faction",
        game: "Game",
        resources_to_other: Dict[ResourceType, int] | None = None,
        resources_from_other: Dict[ResourceType, int] | None = None,
        duration: int = 0,
    ) -> TradeDeal:
        return game.form_trade_deal(
            self,
            other,
            resources_to_other or {},
            resources_from_other or {},
            duration,
        )

    def declare_war(self, other: "Faction", game: "Game") -> None:
        game.declare_war(self, other)

    def agree_truce(self, other: "Faction", game: "Game", duration: int) -> None:
        game.form_truce(self, other, duration)
