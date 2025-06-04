import copy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Faction, Settlement, Position, GREAT_PROJECT_TEMPLATES


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

