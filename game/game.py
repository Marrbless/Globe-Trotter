import random
from dataclasses import dataclass, field

from . import settings

@dataclass
class Position:
    x: int
    y: int

@dataclass
class Settlement:
    name: str
    position: Position


@dataclass
class Building:
    """Simple constructed building worth victory points."""

    name: str
    victory_points: int = 0


@dataclass
class GreatProject:
    """High-cost project that requires multiple turns to complete."""

    name: str
    build_time: int
    victory_points: int = 0
    bonus: str = ""
    progress: int = 0

    def is_complete(self) -> bool:
        return self.progress >= self.build_time

    def advance(self, amount: int = 1) -> None:
        self.progress = min(self.build_time, self.progress + amount)


# Predefined templates for special high-cost projects
GREAT_PROJECT_TEMPLATES = {
    "Grand Cathedral": GreatProject(
        name="Grand Cathedral",
        build_time=5,
        victory_points=10,
        bonus="Increases faith across the realm",
    ),
    "Sky Fortress": GreatProject(
        name="Sky Fortress",
        build_time=8,
        victory_points=15,
        bonus="Provides unmatched military power",
    ),
}

@dataclass
class Faction:
    name: str
    settlement: Settlement
    buildings: list[Building] = field(default_factory=list)
    projects: list[GreatProject] = field(default_factory=list)

    def start_project(self, project: GreatProject) -> None:
        """Begin constructing a great project."""
        self.projects.append(project)

    def progress_projects(self) -> None:
        for proj in self.projects:
            if not proj.is_complete():
                proj.advance()

    def completed_projects(self) -> list[GreatProject]:
        return [p for p in self.projects if p.is_complete()]

    def get_victory_points(self) -> int:
        total = sum(b.victory_points for b in self.buildings)
        total += sum(p.victory_points for p in self.completed_projects())
        return total

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.factions = []

    def is_occupied(self, position: Position) -> bool:
        for faction in self.factions:
            if faction.settlement.position == position:
                return True
        return False

    def distance(self, pos1: Position, pos2: Position) -> float:
        return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5

    def add_faction(self, faction: Faction):
        if not self.is_occupied(faction.settlement.position):
            self.factions.append(faction)
        else:
            raise ValueError("Position is already occupied")

    def spawn_ai_factions(self, player_settlement: Settlement):
        count = settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
        while spawned < count and attempts < count * 10:
            attempts += 1
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if (
                not self.is_occupied(pos)
                and self.distance(pos, player_settlement.position)
                >= settings.MIN_DISTANCE_FROM_PLAYER
            ):
                ai = Faction(name=f"AI #{spawned + 1}", settlement=Settlement(name=f"AI Town {spawned + 1}", position=pos))
                self.add_faction(ai)
                spawned += 1

class Game:
    def __init__(self):
        self.map = Map(*settings.MAP_SIZE)
        self.player_faction: Faction | None = None
        self.turn = 0

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        self.map.spawn_ai_factions(self.player_faction.settlement)
        print("Game started with factions:")
        for faction in self.map.factions:
            print(f"- {faction.name} at {faction.settlement.position}")

    def advance_turn(self) -> None:
        """Progress construction on all ongoing projects."""
        self.turn += 1
        for faction in self.map.factions:
            faction.progress_projects()

    def calculate_scores(self) -> dict[str, int]:
        """Return victory points for all factions."""
        return {f.name: f.get_victory_points() for f in self.map.factions}

def main():
    game = Game()
    # Example: player places settlement at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()

if __name__ == "__main__":
    main()
