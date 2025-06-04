import random
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Map, Position
from world.world import World
from game import settings


def test_hex_distance_calculation():
    m = Map(10, 10)
    assert m.distance(Position(0, 0), Position(2, 1)) == 3
    assert m.distance(Position(1, 1), Position(1, 4)) == 3
    assert m.distance(Position(0, 0), Position(0, 0)) == 0


def test_ai_spawn_distance():
    random.seed(0)
    world = World(width=5, height=5)
    game = Game(world=world)
    game.place_initial_settlement(0, 0)
    ai_factions = game.map.spawn_ai_factions(game.player_faction.settlement)
    assert len(ai_factions) > 0
    for ai in ai_factions:
        dist = game.map.distance(ai.settlement.position, game.player_faction.settlement.position)
        assert dist >= settings.MIN_DISTANCE_FROM_PLAYER
