import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World, ResourceType
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
    initial = game.player_faction.resources[ResourceType.FOOD]
    for _ in range(5):
        game.tick()
    after = game.player_faction.resources[ResourceType.FOOD]
    assert after > initial


def test_farm_increases_food():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(Farm())
    initial = game.player_faction.resources[ResourceType.FOOD]
    game.tick()
    after = game.player_faction.resources[ResourceType.FOOD]
    assert after - initial >= 10  # base + farm bonus


def test_lumbermill_increases_wood():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(LumberMill())
    initial = game.player_faction.resources[ResourceType.WOOD]
    game.tick()
    after = game.player_faction.resources[ResourceType.WOOD]
    assert after - initial >= 3


def test_quarry_increases_stone():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.buildings.append(Quarry())
    initial = game.player_faction.resources[ResourceType.STONE]
    game.tick()
    after = game.player_faction.resources[ResourceType.STONE]
    assert after - initial >= 2


def test_resource_manager_updates_once():
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.ORE: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name
    before = game.resources.data[player][ResourceType.ORE]
    game.tick()
    after = game.resources.data[player][ResourceType.ORE]
    assert after - before == 6


def test_auto_assignment_gathers_resources():
    """Idle citizens should automatically become workers and gather."""
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.WOOD: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    # Remove all assigned workers to simulate idle population
    game.player_faction.workers.assigned = 0
    player = game.player_faction.name
    before = game.resources.data[player][ResourceType.WOOD]
    game.tick()
    after = game.resources.data[player][ResourceType.WOOD]
    assert after - before == 6
