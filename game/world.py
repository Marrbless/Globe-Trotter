import random

class World:
    """Simple hex-based world."""

    def __init__(self, width=30, height=30):
        self.width = width
        self.height = height
        self.hexes = [
            [self._generate_hex(q, r) for q in range(width)]
            for r in range(height)
        ]

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
