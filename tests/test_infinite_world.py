import random
from world.world import World, WorldSettings
from game.game import Game


def test_get_beyond_bounds_infinite():
    settings = WorldSettings(seed=1, width=3, height=3, infinite=True)
    world = World(width=settings.width, height=settings.height, settings=settings)
    hex_ = world.get(10, 10)
    assert hex_ is not None
    assert hex_.coord == (10, 10)


def test_deserialize_faction_outside_bounds_infinite():
    settings = WorldSettings(seed=1, width=3, height=3, infinite=True)
    world = World(width=settings.width, height=settings.height, settings=settings)
    game = Game(world=world)
    game.place_initial_settlement(0, 0)
    player = game.player_faction.name

    game.state.factions = {
        player: {
            "citizens": 10,
            "workers": 0,
            "units": 0,
            "buildings": [],
            "projects": [],
            "settlement": {"name": "Home", "position": {"x": 5, "y": 5}},
        }
    }
    game.state.resources = {player: {}}
    game.state.world = {
        "settings": {"seed": 1, "width": 3, "height": 3, "infinite": True},
        "hexes": {},
    }

    game.map.factions = []
    game._restore_factions_from_state()
    restored = next(f for f in game.map.factions if f.name == player)
    assert restored.settlement.position.x == 5
    assert restored.settlement.position.y == 5
