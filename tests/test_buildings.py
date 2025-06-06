import os
import sys
import pytest

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
from game.technology import TechLevel


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
            tile.resources.clear()
    return w


def test_building_upgrade():
    farm = Farm()
    farm.upgrade()
    assert farm.level == 2
    assert farm.resource_bonus == int(5 * 1.5)
    assert farm.upkeep == int(10 * 1.2)


def test_tick_applies_building_bonus(monkeypatch):
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

    # Deterministic population change: +1 citizen
    values = [1, 0, 0]
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    initial_population = faction.citizens.count
    game.tick()

    expected_food = (initial_population + 1) // 2 + farm.resource_bonus
    assert faction.citizens.count == initial_population + 1
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


def test_build_structure_deducts_resources():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    farm = Farm()
    faction.resources[ResourceType.WOOD] = 200
    starting = faction.resources[ResourceType.WOOD]
    cost = farm.construction_cost[ResourceType.WOOD]
    faction.build_structure(farm)
    assert faction.resources[ResourceType.WOOD] == starting - cost
    assert farm in faction.buildings


def test_building_requires_tech_level():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.tech_level = TechLevel.PRIMITIVE
    faction.resources[ResourceType.WOOD] = 500
    forge = Forge()
    with pytest.raises(ValueError):
        faction.build_structure(forge)
