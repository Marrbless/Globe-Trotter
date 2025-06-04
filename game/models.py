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


