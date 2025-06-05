import os
import sys
import json
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from game.game import GREAT_PROJECT_TEMPLATES
from world.world import World, ResourceType
import game.persistence as persistence
from game import settings


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
    return w


def test_save_and_load(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name
    game.resources.data[player][ResourceType.FOOD] = 7
    game.save()

    loaded, _ = persistence.load_state()
    assert loaded.resources[player][ResourceType.FOOD] == 7


def test_offline_gains(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)

    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.ORE: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name
    initial_pop = game.player_faction.citizens.count

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1005.0)
    ticks = int((1005.0 - 1000.0) // persistence.TICK_DURATION)
    values = [1, 0, 0] * ticks
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))
    loaded, _ = persistence.load_state(world=world, factions=[game.player_faction])
    assert loaded.population == initial_pop + ticks
    assert loaded.resources[player][ResourceType.ORE] == 25


def test_offline_processing_buildings(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)

    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    player = faction.name

    faction.resources = {
        ResourceType.ORE: 4,
        ResourceType.METAL: 0,
        ResourceType.FOOD: 0,
    }
    game.resources.data[player] = faction.resources.copy()
    from game.buildings import Smeltery

    faction.buildings.append(Smeltery())

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1003.0)
    ticks = int((1003.0 - 1000.0) // persistence.TICK_DURATION)
    values = [0, 0, 0] * ticks
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))
    loaded, _ = persistence.load_state(world=world, factions=[faction])

    assert loaded.resources[player][ResourceType.METAL] == 4
    assert loaded.resources[player][ResourceType.ORE] == 0


def test_begin_applies_saved_state_to_faction(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)

    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name

    # Prepare some saved data
    game.player_faction.resources[ResourceType.FOOD] = 10
    game.player_faction.citizens.count = 12
    game.resources.data[player][ResourceType.FOOD] = 10
    game.state.resources = game.resources.data
    game.state.population = game.player_faction.citizens.count
    game.state.factions = {
        player: {
            "citizens": 12,
            "workers": game.player_faction.workers.assigned,
            "buildings": [],
            "projects": [],
        }
    }

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    persistence.save_state(game.state)

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    new_game = Game(world=world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    assert new_game.player_faction.resources[ResourceType.FOOD] == 10
    assert new_game.player_faction.citizens.count == 12


def test_offline_project_completion(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)

    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    assert faction is not None

    from copy import deepcopy

    template = GREAT_PROJECT_TEMPLATES["Grand Cathedral"]
    project = deepcopy(template)
    faction.start_project(project, claimed_projects=game.claimed_projects)

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    offline_time = 1000.0 + project.build_time * persistence.TICK_DURATION
    monkeypatch.setattr(persistence.time, "time", lambda: offline_time)
    monkeypatch.setattr("random.randint", lambda a, b: 0)

    loaded, _ = persistence.load_state(world=world, factions=[faction])

    assert faction.projects[0].is_complete()
    player = faction.name
    assert loaded.factions[player]["projects"][0]["progress"] == project.build_time


def test_population_persists_between_sessions(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    initial_pop = game.player_faction.citizens.count

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1005.0)
    ticks = int((1005.0 - 1000.0) // persistence.TICK_DURATION)
    values = [1, 0, 0] * ticks
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    new_game = Game(world=world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    assert new_game.player_faction.citizens.count == initial_pop + ticks


def test_extended_state_roundtrip(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    world = make_world()
    world.add_road((0, 0), (1, 1))
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    assert faction is not None
    faction.tech_level = 3
    faction.god_powers = {"smite": 1}
    game.event_turn_counters = {"raid": 5}

    game.save()

    new_game = Game(world=world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    assert any(r.start == (0, 0) and r.end == (1, 1) for r in new_game.world.roads)
    assert new_game.event_turn_counters == {"raid": 5}
    assert new_game.player_faction.tech_level == 3
    assert new_game.player_faction.god_powers == {"smite": 1}


def test_buildings_persist_across_save(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    assert faction is not None

    from game.buildings import Farm, House

    faction.resources[ResourceType.WOOD] = 300
    faction.build_structure(Farm())
    faction.build_structure(House())
    game.resources.data[faction.name] = faction.resources.copy()

    game.save()

    new_game = Game(world=world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    new_faction = new_game.player_faction
    assert new_faction is not None
    assert any(b.name == "Farm" for b in new_faction.buildings)
    assert any(b.name == "House" for b in new_faction.buildings)
    assert new_faction.resources[ResourceType.WOOD] == faction.resources[ResourceType.WOOD]
