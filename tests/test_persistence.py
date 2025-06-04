import os
import sys
import json
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World, ResourceType
import game.persistence as persistence


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

    loaded = persistence.load_state()
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

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    monkeypatch.setattr(persistence.time, "time", lambda: 1005.0)
    loaded = persistence.load_state(world=world, factions=[game.player_faction])

    assert loaded.population == 15
    assert loaded.resources[player][ResourceType.ORE] == 30


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
