import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World
from game.buildings import Farm


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
    faction.resources = {"food": 0, "wood": 0, "stone": 0}
    farm = Farm()
    farm.upgrade()
    faction.buildings.append(farm)
    initial_population = faction.population
    game.tick()
    expected_food = (initial_population + 1) // 2 + farm.resource_bonus
    assert faction.resources["food"] == expected_food
