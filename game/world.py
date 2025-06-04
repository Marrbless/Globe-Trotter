import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Road:
    start: Tuple[int, int]
    end: Tuple[int, int]


class World:
    """Simple hex-based world."""

    def __init__(self, width=30, height=30):
        self.width = width
        self.height = height
        self.hexes = [
            [self._generate_hex(q, r) for q in range(width)]
            for r in range(height)
        ]
        self.roads: List[Road] = []

    def _generate_hex(self, q, r):
        terrains = ["plains", "forest", "mountains", "hills", "water"]
        return {
            "q": q,
            "r": r,
            "terrain": random.choice(terrains),
        }

    def get(self, q, r):
        if 0 <= q < self.width and 0 <= r < self.height:
            return self.hexes[r][q]
        return None

    def has_road(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        for road in self.roads:
            if (
                (road.start == start and road.end == end)
                or (road.start == end and road.end == start)
            ):
                return True
        return False

    def add_road(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        if not self.get(*start) or not self.get(*end):
            raise ValueError("Invalid road endpoints")
        if start == end:
            raise ValueError("Road must connect two different hexes")
        if not self.has_road(start, end):
            self.roads.append(Road(start, end))

    def trade_efficiency(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Return trade efficiency modifier between two points."""
        base = 1.0
        if self.has_road(start, end):
            return base * 1.5
        return base
