import copy
import os
import sys
import time
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game, Faction, Settlement, Position, GREAT_PROJECT_TEMPLATES
import game.persistence as persistence
from game.persistence import GameState, load_state
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


def test_power_cooldown_and_persistence(tmp_path):
    world = make_world()
    tmp_file = tmp_path / "save.json"
    game = Game(world=world, state=GameState(timestamp=time.time(), resources={}, population=0), save_file=tmp_file)
    game.place_initial_settlement(1, 1)
    fac = game.player_faction
    fac.resources[ResourceType.WOOD] = 700
    fac.resources[ResourceType.STONE] = 700

    game.use_power("Summon Harvest")
    assert game.power_cooldowns["Summon Harvest"] == 3
    game.tick()
    assert game.power_cooldowns["Summon Harvest"] == 2
    game.save()

    # Load new game from saved state
    new_state, _ = load_state(file_path=tmp_file)
    new_game = Game(state=new_state, world=world, save_file=tmp_file)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()
    assert new_game.power_cooldowns["Summon Harvest"] <= 2
