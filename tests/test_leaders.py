import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position
from world.world import World, Road


def test_army_leader_vp_updates():
    world = World(width=3, height=3)
    game = Game(world=world)
    game.place_initial_settlement(0, 0)
    player = game.player_faction

    rival = Faction(name="Rival", settlement=Settlement(name="Riv", position=Position(1, 0)))
    game.map.add_faction(rival)
    game.resources.register(rival)
    game.faction_manager.add_faction(rival)

    player.units = 5
    rival.units = 3
    game.update_leaders()
    scores = game.calculate_scores()
    assert scores[player.name] == player.get_victory_points() + 2
    assert scores[rival.name] == rival.get_victory_points()

    rival.units = 6
    game.update_leaders()
    scores = game.calculate_scores()
    assert scores[rival.name] == rival.get_victory_points() + 2
    assert scores[player.name] == player.get_victory_points()


def test_longest_road_leader_vp_updates():
    world = World(width=5, height=5)
    game = Game(world=world)
    game.place_initial_settlement(0, 0)
    player = game.player_faction

    rival = Faction(name="Rival", settlement=Settlement(name="Riv", position=Position(4, 4)))
    game.map.add_faction(rival)
    game.resources.register(rival)
    game.faction_manager.add_faction(rival)

    world.roads = [Road((0, 0), (1, 0)), Road((1, 0), (2, 0)), Road((4, 4), (4, 3))]

    game.update_leaders()
    scores = game.calculate_scores()
    assert scores[player.name] == player.get_victory_points() + 2
    assert scores[rival.name] == rival.get_victory_points()

    world.roads.append(Road((4, 3), (4, 2)))
    world.roads.append(Road((4, 2), (4, 1)))
    game.update_leaders()
    scores = game.calculate_scores()
    assert scores[rival.name] == rival.get_victory_points() + 2
    assert scores[player.name] == player.get_victory_points()

