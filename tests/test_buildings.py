import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World, ResourceType
from game.buildings import (
    Farm,
    Smeltery,
    Mill,
    Bakery,
    Forge,
)


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
    return w


def test_building_upgrade():
    farm = Farm()
    farm.upgrade()
    assert farm.level == 2
    assert farm.resource_bonus == int(5 * 1.5)
    assert farm.upkeep == int(10 * 1.2)


def test_tick_applies_building_bonus():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.resources = {
        ResourceType.FOOD: 0,
        ResourceType.WOOD: 0,
        ResourceType.STONE: 0,
    }
    farm = Farm()
    farm.upgrade()
    faction.buildings.append(farm)
    initial_population = faction.citizens.count
    game.tick()
    expected_food = (initial_population + 1) // 2 + farm.resource_bonus
    assert faction.resources[ResourceType.FOOD] == expected_food


def test_processing_building_converts_resources():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.resources = {
        ResourceType.ORE: 5,
        ResourceType.METAL: 0,
        ResourceType.FOOD: 0,
    }
    smeltery = Smeltery()
    faction.buildings.append(smeltery)
    game.tick()
    assert faction.resources[ResourceType.ORE] == 3
    assert faction.resources[ResourceType.METAL] == 2


def test_full_processing_chain_over_ticks():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.resources = {
        ResourceType.WHEAT: 4,
        ResourceType.FLOUR: 0,
        ResourceType.BREAD: 0,
    }
    mill = Mill()
    bakery = Bakery()
    faction.buildings.extend([mill, bakery])

    game.tick()
    assert faction.resources[ResourceType.WHEAT] == 2
    assert faction.resources[ResourceType.FLOUR] == 0
    assert faction.resources[ResourceType.BREAD] == 2

    game.tick()
    assert faction.resources[ResourceType.WHEAT] == 0
    assert faction.resources[ResourceType.FLOUR] == 0
    assert faction.resources[ResourceType.BREAD] == 4
