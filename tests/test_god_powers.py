import copy
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position, GREAT_PROJECT_TEMPLATES
from world.world import World, ResourceType


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile.terrain = "plains"
    return w


def complete_project(faction: Faction, name: str) -> None:
    template = copy.deepcopy(GREAT_PROJECT_TEMPLATES[name])
    faction.start_project(template, claimed_projects=set())
    for _ in range(template.build_time):
        faction.progress_projects()


def test_summon_harvest_cost_and_effect():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.resources[ResourceType.WOOD] = 700
    faction.resources[ResourceType.STONE] = 700
    faction.resources[ResourceType.FOOD] = 0

    assert any(p.name == "Summon Harvest" for p in game.available_powers())
    game.use_power("Summon Harvest")
    assert faction.resources[ResourceType.FOOD] == 500
    assert faction.resources[ResourceType.WOOD] == 400
    assert faction.resources[ResourceType.STONE] == 400


def test_quell_disaster_requires_project_and_deducts_resources():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction = game.player_faction
    faction.resources[ResourceType.WOOD] = 500
    faction.resources[ResourceType.STONE] = 500
    faction.resources[ResourceType.FOOD] = 500

    complete_project(faction, "Sky Fortress")

    assert any(p.name == "Quell Disaster" for p in game.available_powers())
    pop_before = faction.citizens.count
    game.use_power("Quell Disaster")
    assert faction.citizens.count == pop_before + 20
    assert faction.resources[ResourceType.WOOD] == 400
    assert faction.resources[ResourceType.STONE] == 400
    assert faction.resources[ResourceType.FOOD] == 400


def test_power_fails_without_requirements():
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)

    with pytest.raises(ValueError):
        game.use_power("Summon Harvest")
