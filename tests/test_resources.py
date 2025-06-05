import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import (
    World,
    WorldSettings,
    ResourceType,
    STRATEGIC_RESOURCES,
    LUXURY_RESOURCES,
)
from world.resources import RESOURCE_RULES
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
    assert after - initial >= 10


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
    assert after - before == 5


def test_auto_assignment_gathers_resources(monkeypatch):
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.WOOD: 1}

    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    game.player_faction.workers.assigned = 0
    assert game.player_faction.workers.assigned == 0
    player = game.player_faction.name

    values = [0, 0, 0]
    monkeypatch.setattr("random.randint", lambda a, b: values.pop(0))

    before = game.resources.data[player][ResourceType.WOOD]
    game.tick()
    after = game.resources.data[player][ResourceType.WOOD]
    assert game.player_faction.workers.assigned == game.player_faction.citizens.count
    assert after - before == 5


def test_manual_assignment_yields_more_over_time(monkeypatch):
    world = make_world()
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = world.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.resources = {ResourceType.WOOD: 1}

    monkeypatch.setattr("random.randint", lambda a, b: 0)

    manual_game = Game(world=world)
    manual_game.place_initial_settlement(1, 1)
    manual_game.faction_manager.toggle_assignment(
        manual_game.player_faction, True
    )
    manual_game.player_faction.workers.assigned = manual_game.player_faction.citizens.count

    auto_game = Game(world=world)
    auto_game.place_initial_settlement(1, 1)
    auto_game.faction_manager.toggle_assignment(
        auto_game.player_faction, False, "mid"
    )

    manual_before = manual_game.player_faction.resources[ResourceType.WOOD]
    auto_before = auto_game.player_faction.resources[ResourceType.WOOD]

    for _ in range(5):
        manual_game.tick()
        auto_game.tick()

    manual_after = manual_game.player_faction.resources[ResourceType.WOOD]
    auto_after = auto_game.player_faction.resources[ResourceType.WOOD]

    assert (manual_after - manual_before) > (auto_after - auto_before)


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
    expected = int(round(sum(amounts) * 0.8))
    assert after - before == expected


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
    player = faction.name

    before_store = game.resources.data[player][ResourceType.WOOD]
    before_faction = faction.resources[ResourceType.WOOD]

    game.resources.tick([faction])

    after_store = game.resources.data[player][ResourceType.WOOD]
    after_faction = faction.resources[ResourceType.WOOD]

    assert after_store - before_store == 6
    assert after_faction - before_faction == 6


def test_register_uses_faction_resources():
    world = World(width=3, height=3)
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction

    expected_food = faction.resources[ResourceType.FOOD]
    manager = ResourceManager(world)
    manager.register(faction)

    assert manager.data[faction.name][ResourceType.FOOD] == expected_food


def test_advanced_resources_generated():
    settings = WorldSettings(seed=42, width=5, height=5)
    world = World(width=settings.width, height=settings.height, settings=settings)

    advanced = {ResourceType.IRON, ResourceType.GOLD, ResourceType.WHEAT, ResourceType.WOOL}
    found: set[ResourceType] = set()
    for hex_ in world.all_hexes():
        for res in hex_.resources:
            if res in advanced:
                found.add(res)

    assert any(res in found for res in advanced)


def test_strategic_and_luxury_resources_generated():
    settings = WorldSettings(seed=99, width=10, height=10)
    world = World(width=settings.width, height=settings.height, settings=settings)

    strategic_found: set[ResourceType] = set()
    luxury_found: set[ResourceType] = set()

    for hex_ in world.all_hexes():
        for res in hex_.resources:
            if res in STRATEGIC_RESOURCES:
                strategic_found.add(res)
            if res in LUXURY_RESOURCES:
                luxury_found.add(res)

    assert strategic_found
    assert luxury_found


def test_all_resource_types_can_generate():
    expected = {rule[0] for rules in RESOURCE_RULES.values() for rule in rules}
    found: set[ResourceType] = set()
    base = WorldSettings(
        seed=0,
        width=20,
        height=20,
        moisture=1.0,
        temperature=1.0,
        rainfall_intensity=5,
        sea_level=0.3,
    )

    for seed in range(30):
        s = WorldSettings(
            seed=seed,
            width=base.width,
            height=base.height,
            moisture=base.moisture,
            temperature=base.temperature,
            rainfall_intensity=base.rainfall_intensity,
            sea_level=base.sea_level,
        )
        world = World(width=s.width, height=s.height, settings=s)
        for hex_ in world.all_hexes():
            found.update(hex_.resources.keys())
        if expected.issubset(found):
            break

    assert expected.issubset(found)
