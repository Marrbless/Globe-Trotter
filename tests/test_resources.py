import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from game.world import World


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile["terrain"] = "plains"
    return w


def test_resources_increase_without_buildings():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    manager = game.resources
    initial = manager.data[game.player_faction.name]["food"]
    for _ in range(5):
        game.tick()
    after = manager.data[game.player_faction.name]["food"]
    assert after > initial
