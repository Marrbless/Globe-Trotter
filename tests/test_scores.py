import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position
from game.buildings import Building
from game.models import GreatProject
from game import settings


def test_victory_point_totals(monkeypatch):
    """Verify victory points include cities, roads and armies."""

    # Simplify base score
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 1)

    game = Game()
    game.place_initial_settlement(0, 0)
    player = game.player_faction

    rival = Faction(name="Rival", settlement=Settlement(name="Riv", position=Position(1, 1)))
    game.map.add_faction(rival)

    # Configure attributes
    player.total_roads = 3
    player.city_count = 1
    player.army_size = 5

    rival.total_roads = 2
    rival.army_size = 4

    # Add building and project for player
    building = Building(name="Monument", construction_cost={}, upkeep=0, victory_points=2)
    player.buildings.append(building)

    project = GreatProject(name="Wonder", build_time=1, victory_points=3)
    project.progress = 1
    player.projects.append(project)

    scores = game.calculate_scores()

    base = (settings.AI_FACTION_COUNT + 1) * 10

    expected_player = base + 1 + player.city_count * 2
    expected_player += building.victory_points + project.victory_points
    expected_player += 5 + 5  # longest road and largest army

    expected_rival = base + 1 + rival.city_count * 2

    assert scores[player.name] == expected_player
    assert scores[rival.name] == expected_rival

