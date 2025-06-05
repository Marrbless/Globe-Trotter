import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position
from world.world import World, ResourceType
from game import ai
from game.population import Citizen


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
            tile.resources = {}
    return w


def _setup_two_ai(game):
    a = Faction(
        name="A",
        settlement=Settlement(name="A", position=Position(0, 0)),
        citizens=Citizen(count=15),
    )
    b = Faction(
        name="B",
        settlement=Settlement(name="B", position=Position(2, 2)),
        citizens=Citizen(count=5),
    )
    for f in (a, b):
        f.workers.assigned = 0
        f.manual_assignment = True
        game.map.add_faction(f)
        game.resources.register(f)
        game.faction_manager.add_faction(f)
    return a, b


def test_alliance_then_betrayal(monkeypatch):
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    a, b = _setup_two_ai(game)

    a.resources[ResourceType.FOOD] = 1
    a.resources[ResourceType.WOOD] = 100
    b.resources[ResourceType.FOOD] = 100
    b.resources[ResourceType.WOOD] = 1

    monkeypatch.setattr("random.random", lambda: 0.0)
    ai._consider_alliance(game, a, b)
    assert game.is_allied(a, b)

    ai._consider_betrayal(game, a, b)
    assert game.is_at_war(a, b)
    assert not game.is_allied(a, b)


def test_break_truce(monkeypatch):
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    a, b = _setup_two_ai(game)
    game.form_truce(a, b, duration=5)

    monkeypatch.setattr("random.random", lambda: 0.0)
    ai._consider_break_truce(game, a, b)
    assert not game.truces
    assert game.is_at_war(a, b)


def test_trade_and_alliance_over_ticks(monkeypatch):
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    a, b = _setup_two_ai(game)

    a.resources[ResourceType.FOOD] = 1
    a.resources[ResourceType.WOOD] = 100
    b.resources[ResourceType.FOOD] = 100
    b.resources[ResourceType.WOOD] = 1

    vals = iter([0.0, 0.0, 1.0, 0.0])

    monkeypatch.setattr("random.random", lambda: next(vals))

    ai.evaluate_relations(game)
    assert game.trade_deals
    assert game.is_allied(a, b)

    ai.evaluate_relations(game)
    assert game.is_at_war(a, b)
    assert not game.is_allied(a, b)

