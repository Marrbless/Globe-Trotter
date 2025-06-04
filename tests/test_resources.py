import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from game.world import World
from game.buildings import Farm, LumberMill, Quarry, Mine


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
    return w


def test_resources_increase_without_buildings():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    initial = game.player_faction.resources["food"]
    for _ in range(5):
        game.tick()
    after = game.player_faction.resources["food"]
    assert after > initial


def test_farm_increases_food():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(Farm())
    initial = game.player_faction.resources["food"]
    game.tick()
    after = game.player_faction.resources["food"]
    assert after - initial >= 10  # base + farm bonus


def test_lumbermill_increases_wood():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(LumberMill())
    initial = game.player_faction.resources["wood"]
    game.tick()
    after = game.player_faction.resources["wood"]
    assert after - initial >= 3


def test_quarry_increases_stone():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(Quarry())
    initial = game.player_faction.resources["stone"]
    game.tick()
    after = game.player_faction.resources["stone"]
    assert after - initial >= 2


def test_resource_manager_updates_once():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name
    before = game.resources.data[player]["food"]
    game.tick()
    after = game.resources.data[player]["food"]
    assert after - before == 6
