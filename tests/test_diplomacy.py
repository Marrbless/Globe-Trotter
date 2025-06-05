import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position
from world.world import World, ResourceType


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
            tile.resources = {}
    return w


def test_trade_deal_transfers_resources(monkeypatch):
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction

    rival = Faction(name="Rival", settlement=Settlement(name="Riv", position=Position(0,0)))
    rival.workers.assigned = 0
    rival.manual_assignment = True
    player.workers.assigned = 0
    player.manual_assignment = True
    game.map.add_faction(rival)
    game.resources.register(rival)
    game.faction_manager.add_faction(rival)

    player.resources = {ResourceType.FOOD: 5, ResourceType.WOOD: 0}
    rival.resources = {ResourceType.FOOD: 0, ResourceType.WOOD: 5}

    game.form_trade_deal(player, rival, {ResourceType.FOOD: 1}, {ResourceType.WOOD: 2})

    monkeypatch.setattr("random.randint", lambda a, b: 0)
    game.tick()

    assert player.resources[ResourceType.FOOD] == 9
    assert player.resources[ResourceType.WOOD] == 2
    assert rival.resources[ResourceType.FOOD] == 6
    assert rival.resources[ResourceType.WOOD] == 3
