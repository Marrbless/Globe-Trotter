import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World, WorldSettings, ResourceType
from game.resources import ResourceManager
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


def test_resource_manager_updates_once(monkeypatch):
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.ORE: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name

    values = [0, 0, 0]
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    before = game.resources.data[player][ResourceType.ORE]
    game.tick()
    after = game.resources.data[player][ResourceType.ORE]
    assert after - before == 6


def test_auto_assignment_gathers_resources(monkeypatch):
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

    values = [0, 0, 0]
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    before = game.resources.data[player][ResourceType.WOOD]
    game.tick()
    after = game.resources.data[player][ResourceType.WOOD]
    assert after - before == 6


def test_richer_tiles_yield_more_resources(monkeypatch):
    world = make_world()
    center = (1, 1)
    amounts = [5, 1, 1, 1, 1, 1]
    for (dq, dr), amt in zip(
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)], amounts
    ):
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.ORE: amt}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name

    values = [0, 0, 0]
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    before = game.resources.data[player][ResourceType.ORE]
    game.tick()
    after = game.resources.data[player][ResourceType.ORE]
    assert after - before == sum(amounts)


def test_resource_manager_tick_updates_faction_store():
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.WOOD: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    assert faction is not None
    player = faction.name

    before_store = game.resources.data[player][ResourceType.WOOD]
    before_faction = faction.resources[ResourceType.WOOD]

    game.resources.tick([faction])

    after_store = game.resources.data[player][ResourceType.WOOD]
    after_faction = faction.resources[ResourceType.WOOD]

    assert after_store - before_store == 6
    assert after_faction - before_faction == 6


def test_register_uses_faction_resources():
    """ResourceManager should start with faction's initial amounts."""
    world = World(width=3, height=3)
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    assert faction is not None

    # Food has a non-zero default value on a new faction
    expected_food = faction.resources[ResourceType.FOOD]
    manager = ResourceManager(world)
    manager.register(faction)

    assert manager.data[faction.name][ResourceType.FOOD] == expected_food


def test_advanced_resources_generated():
    """World generation should include some rare resources."""
    settings = WorldSettings(seed=42, width=5, height=5)
    world = World(width=settings.width, height=settings.height, settings=settings)

    advanced = {ResourceType.IRON, ResourceType.GOLD, ResourceType.WHEAT, ResourceType.WOOL}
    found: set[ResourceType] = set()
    for row in world.hexes:
        for hex_ in row:
            for res in hex_.resources:
                if res in advanced:
                    found.add(res)

    assert any(res in found for res in advanced)
