from dataclasses import dataclass

@dataclass
class Building:
    """Base class for all buildings."""

    name: str
    construction_cost: int
    upkeep: int
    resource_bonus: int = 0
    population_bonus: int = 0
    level: int = 1

    def upgrade_cost(self) -> int:
        """Cost required to upgrade this building."""
        return int(self.construction_cost * 1.5 * self.level)

    def upgrade(self) -> None:
        """Upgrade this building, improving bonuses and upkeep."""
        self.level += 1
        self.resource_bonus = int(self.resource_bonus * 1.5)
        self.population_bonus = int(self.population_bonus * 1.5)
        self.upkeep = int(self.upkeep * 1.2)


@dataclass
class Farm(Building):
    name: str = "Farm"
    construction_cost: int = 100
    upkeep: int = 10
    resource_bonus: int = 5


@dataclass
class Mine(Building):
    name: str = "Mine"
    construction_cost: int = 150
    upkeep: int = 15
    resource_bonus: int = 10


@dataclass
class House(Building):
    name: str = "House"
    construction_cost: int = 50
    upkeep: int = 5
    population_bonus: int = 2
