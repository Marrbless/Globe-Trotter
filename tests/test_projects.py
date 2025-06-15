import copy
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import (
    Game,
    Faction,
    Settlement,
    Position,
    GREAT_PROJECT_TEMPLATES,
)
from world.world import WorldSettings, World, RiverSegment
import game.persistence as persistence
from game import settings


def test_project_completion_unlocks_actions_and_scores():
    faction = Faction(name="Test", settlement=Settlement(name="Home", position=Position(0, 0)))
    template = GREAT_PROJECT_TEMPLATES["Grand Cathedral"]
    project = copy.deepcopy(template)
    faction.start_project(project, claimed_projects=set())
    for _ in range(project.build_time):
        faction.progress_projects()
    assert project.is_complete()
    assert "celebrate_festival" in faction.unlocked_actions
    expected = (settings.AI_FACTION_COUNT + 1) * 10
    expected += 1  # settlement
    expected += faction.city_count * 2
    expected += project.victory_points
    assert faction.get_victory_points() == expected


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


def test_claimed_projects_persist_after_loading(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    game = Game(save_file=tmp_file)
    game.place_initial_settlement(0, 0)

    first = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    game.player_faction.start_project(first, claimed_projects=game.claimed_projects)

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    new_game = Game(save_file=tmp_file)
    new_game.place_initial_settlement(0, 0)
    new_game.begin()

    rival = Faction(
        name="Rival",
        settlement=Settlement(name="Riv", position=Position(1, 1)),
    )
    new_game.map.add_faction(rival)

    second = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    with pytest.raises(ValueError):
        rival.start_project(second, claimed_projects=new_game.claimed_projects)


def test_claimed_projects_block_reclaim_after_resave(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    game = Game(save_file=tmp_file)
    game.place_initial_settlement(0, 0)

    first = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    game.player_faction.start_project(first, claimed_projects=game.claimed_projects)

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    mid_game = Game(save_file=tmp_file)
    mid_game.place_initial_settlement(0, 0)
    mid_game.begin()
    mid_game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    new_game = Game(save_file=tmp_file)
    new_game.place_initial_settlement(0, 0)
    new_game.begin()

    rival = Faction(
        name="Rival",
        settlement=Settlement(name="Riv", position=Position(1, 1)),
    )
    new_game.map.add_faction(rival)

    second = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Grand Cathedral"])
    with pytest.raises(ValueError):
        rival.start_project(second, claimed_projects=new_game.claimed_projects)


def test_great_dam_converts_river_to_lake():
    settings_obj = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings_obj.width, height=settings_obj.height, settings=settings_obj)
    center = (1, 1)
    neighbor = (1, 2)
    for nq, nr in world._neighbors(*center):
        h = world.get(nq, nr)
        if h:
            h.river = False
    world.rivers.clear()
    hex_n = world.get(*neighbor)
    hex_n.river = True
    world.rivers.append(RiverSegment(neighbor, center))

    faction = Faction(
        name="Test",
        settlement=Settlement(name="Home", position=Position(*center)),
        world=world,
    )
    template = copy.deepcopy(GREAT_PROJECT_TEMPLATES["Great Dam"])
    faction.start_project(template, claimed_projects=set())
    for _ in range(template.build_time):
        faction.progress_projects()

    assert template.is_complete()
    assert hex_n.lake
    assert not hex_n.river
    assert neighbor in world.lakes

