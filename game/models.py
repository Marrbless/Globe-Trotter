from dataclasses import dataclass, field
from typing import Dict, List

from .buildings import Building


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
    name: str
    settlement: Settlement
    population: int = 10
    resources: Dict[str, int] = field(
        default_factory=lambda: {"food": 100, "wood": 50, "stone": 30}
    )
    workers: Dict[str, int] = field(default_factory=lambda: {"food": 10, "wood": 0, "stone": 0})
    buildings: List[Building] = field(default_factory=list)
    projects: List[GreatProject] = field(default_factory=list)

    def start_project(self, project: GreatProject) -> None:
        """Begin constructing a great project."""
        self.projects.append(project)

    def progress_projects(self) -> None:
        for proj in self.projects:
            if not proj.is_complete():
                proj.advance()

    def completed_projects(self) -> List[GreatProject]:
        return [p for p in self.projects if p.is_complete()]

    def get_victory_points(self) -> int:
        total = sum(b.victory_points for b in self.buildings)
        total += sum(p.victory_points for p in self.completed_projects())
        return total

    def build_structure(self, building: Building) -> None:
        """Pay required resources and add the Building instance."""
        cost: Dict[str, int] = building.construction_cost
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to build {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        self.buildings.append(building)

    def upgrade_structure(self, building: Building) -> None:
        """Pay upgrade cost then call the building's upgrade method."""
        cost: Dict[str, int] = building.upgrade_cost()
        for res_type, amt in cost.items():
            if self.resources.get(res_type, 0) < amt:
                raise ValueError(f"Not enough {res_type} to upgrade {building.name}")
        for res_type, amt in cost.items():
            self.resources[res_type] -= amt
        building.upgrade()
