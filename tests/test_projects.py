import copy
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position, GREAT_PROJECT_TEMPLATES


def test_project_completion_unlocks_actions_and_scores():
    faction = Faction(name="Test", settlement=Settlement(name="Home", position=Position(0, 0)))
    template = GREAT_PROJECT_TEMPLATES["Grand Cathedral"]
    project = copy.deepcopy(template)
    faction.start_project(project)
    for _ in range(project.build_time):
        faction.progress_projects()
    assert project.is_complete()
    assert "celebrate_festival" in faction.unlocked_actions
    assert faction.get_victory_points() == project.victory_points


def test_projects_can_only_be_claimed_once():
    game = Game()
    game.place_initial_settlement(0, 0)
    rival = Faction(
        name="Rival",
        settlement=Settlement(name="Riv", position=Position(1, 1)),
    )
    game.map.add_faction(rival)

    first = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    game.player_faction.start_project(first, claimed_projects=game.claimed_projects)

    second = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    with pytest.raises(ValueError):
        rival.start_project(second, claimed_projects=game.claimed_projects)

