from __future__ import annotations

"""Data model for a single world hex tile."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .resource_types import ResourceType

Coordinate = Tuple[int, int]


@dataclass
class Hex:
    coord: Coordinate
    terrain: str = "plains"
    elevation: float = 0.0
    moisture: float = 0.0
    temperature: float = 0.0
    resources: Dict[ResourceType, int] = field(default_factory=dict)
    flooded: bool = False
    ruined: bool = False
    river: bool = False
    lake: bool = False
    water_flow: float = 0.0

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)


__all__ = ["Hex", "Coordinate"]
