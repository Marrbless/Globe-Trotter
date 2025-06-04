import random
from typing import List, Dict

from .persistence import GameState, load_state, save_state
from .buildings import Building, mitigate_building_damage, mitigate_population_loss
from . import settings
from .world import World
from .resources import ResourceManager
from .models import Position, Settlement, Faction, GreatProject




# Predefined templates for special high-cost projects
GREAT_PROJECT_TEMPLATES: Dict[str, GreatProject] = {
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




class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.factions: List[Faction] = []

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

    def spawn_ai_factions(self, player_settlement: Settlement) -> List[Faction]:
        count = settings.AI_FACTION_COUNT
        spawned = 0
        attempts = 0
        new_factions: List[Faction] = []
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
                ai = Faction(
                    name=f"AI #{spawned + 1}",
                    settlement=Settlement(name=f"AI Town {spawned + 1}", position=pos),
                    population=random.randint(8, 15),
                )
                self.add_faction(ai)
                new_factions.append(ai)
                spawned += 1
        return new_factions


class Game:
    def __init__(self, state: GameState | None = None, world: World | None = None):
        # Initialize map and world
        self.map = Map(*settings.MAP_SIZE)
        self.world = world or World(*settings.MAP_SIZE)

        # Load or create state
        self.state = state or load_state()

        # If state has stored resources, use them; otherwise create ResourceManager
        self.resources: ResourceManager | Dict[str, int]
        if isinstance(self.state.resources, ResourceManager):
            self.resources = self.state.resources
        else:
            self.resources = ResourceManager(self.world)
            # Overwrite with persisted dictionary if necessary
            for faction in self.map.factions:
                self.resources.register(faction)
            # In a freshly loaded state, the resources dict should map faction names to resource states
            # If needed, user should handle registration elsewhere

        self.population = self.state.population
        self.player_faction: Faction | None = None
        self.player_buildings: List[Building] = []
        self.turn = 0

    def place_initial_settlement(self, x: int, y: int, name: str = "Player"):
        pos = Position(x, y)
        if self.map.is_occupied(pos):
            raise ValueError("Cannot place settlement on occupied location")
        settlement = Settlement(name="Home", position=pos)
        self.player_faction = Faction(name=name, settlement=settlement)
        self.map.add_faction(self.player_faction)
        # Register resources for the new faction
        if isinstance(self.resources, ResourceManager):
            self.resources.register(self.player_faction)

    def add_building(self, building: Building):
        """Add a defensive building to the player's settlement."""
        self.player_buildings.append(building)

    def begin(self):
        if not self.player_faction:
            raise RuntimeError("Player settlement not placed")
        ai_factions = self.map.spawn_ai_factions(self.player_faction.settlement)
        if isinstance(self.resources, ResourceManager):
            for faction in self.map.factions:
                self.resources.register(faction)

        print("Game started with factions:")
        for faction in self.map.factions:
            pos = faction.settlement.position
            print(f"- {faction.name} at ({pos.x}, {pos.y})")

        # Show initial resource and population state
        print(f"Resources: {self.state.resources}")
        print(f"Population: {self.state.population}")

        # Simulate a sample event demonstrating defensive buildings
        self.simulate_events()

    def simulate_events(self):
        """Run sample attack and disaster to show defensive buildings."""
        if not self.player_faction:
            return
        base_pop_loss = 100
        base_damage = 50
        pop_loss = mitigate_population_loss(self.player_buildings, base_pop_loss)
        damage = mitigate_building_damage(self.player_buildings, base_damage)
        print(f"Population loss mitigated from {base_pop_loss} to {pop_loss}")
        print(f"Building damage mitigated from {base_damage} to {damage}")

    def build_for_player(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.build_structure(building)

    def upgrade_player_building(self, building: Building) -> None:
        if not self.player_faction:
            raise RuntimeError("Player faction not initialized")
        self.player_faction.upgrade_structure(building)

    def tick(self) -> None:
        """
        Advance the game state by one tick. This includes:
          1. Population growth
          2. Basic resource generation (food from population)
          3. Building-based resource bonuses
        """
        for faction in self.map.factions:
            # 1. Population growth
            faction.population += 1

            # 2. Generate base food from population
            food_gain = faction.population // 2
            faction.resources["food"] = faction.resources.get("food", 0) + food_gain

            # 3. Building effects
            for building in faction.buildings:
                b_type = getattr(building, "name", None)
                if b_type == "Farm":
                    faction.resources["food"] = faction.resources.get("food", 0) + 5
                elif b_type == "LumberMill":
                    faction.resources["wood"] = faction.resources.get("wood", 0) + 3
                elif b_type == "Quarry":
                    faction.resources["stone"] = faction.resources.get("stone", 0) + 2
                elif b_type == "Mine":
                    faction.resources["stone"] = faction.resources.get("stone", 0) + 4

        # Debug output for the player faction
        if self.player_faction:
            res = self.player_faction.resources
            pop = self.player_faction.population
            print(f"Resources: {res} | Population: {pop}")

    def save(self) -> None:
        self.state.resources = self.resources
        self.state.population = self.population
        save_state(self.state)

    def advance_turn(self) -> None:
        """Progress construction on all ongoing projects."""
        self.turn += 1
        for faction in self.map.factions:
            faction.progress_projects()

    def calculate_scores(self) -> Dict[str, int]:
        """Return victory points for all factions."""
        return {f.name: f.get_victory_points() for f in self.map.factions}


def main():
    game = Game()
    # Example: player places settlement at (0, 0)
    game.place_initial_settlement(0, 0)
    game.begin()
    game.save()


if __name__ == "__main__":
    main()
