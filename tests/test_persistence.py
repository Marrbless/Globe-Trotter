import os
import sys
import json
import tempfile
import random
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from game.game import GREAT_PROJECT_TEMPLATES
from world.world import World, ResourceType
import game.persistence as persistence
from game import settings


# --- Constants & Helpers ------------------------------------------------------

# Axial hex directions (dq, dr)
NEIGHBOR_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
]


def make_world(resource_per_tile: dict[ResourceType, int] | None = None) -> World:
    """
    Create a 3×3 World centered at (1, 1).  By default, sets all six direct neighbors
    to terrain="plains".  If `resource_per_tile` is provided, each neighbor tile also
    gets those resources assigned.
    """
    w = World(width=3, height=3)
    center_x, center_y = 1, 1
    for dq, dr in NEIGHBOR_DIRECTIONS:
        tile = w.get(center_x + dq, center_y + dr)
        if tile:
            tile.terrain = "plains"
            if resource_per_tile is not None:
                tile.resources = resource_per_tile.copy()
    return w


@pytest.fixture
def initialized_game(tmp_path, monkeypatch):
    """
    Pytest fixture that does the repeated setup:
    - Creates a temporary save file
    - Patches persistence.SAVE_FILE → tmp_path/"save.json"
    - Creates a 3×3 world, places an initial settlement, and returns (game, faction_name).
    """
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)

    # Suppress AI factions unless explicitly needed
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    faction_name = game.player_faction.name
    return game, faction_name


# --- Tests --------------------------------------------------------------------


def test_save_and_load_returns_same_food_count(initialized_game):
    """Saving a game and reloading it should restore FOOD exactly as it was."""
    game, player = initialized_game

    # Add 7 units of FOOD to the player faction, then save
    game.resources.data[player][ResourceType.FOOD] = 7
    game.save()

    # Load from disk
    loaded_state, _ = persistence.load_state()
    assert loaded_state.resources[player][ResourceType.FOOD] == 7


def test_offline_gains_increase_population_and_ore(initialized_game, monkeypatch):
    """
    If we go offline for a few ticks and each tick yields 1 ORE 1/3 of the time,
    population should increase by ticks, and total ORE should be 25 after 5 seconds.
    """
    game, player = initialized_game

    # Populate neighbor tiles with exactly 1 ORE each
    resource_map = {ResourceType.ORE: 1}
    world = make_world(resource_map)
    # Replace the game's world with this new world so that persistence.load_state can find it
    game.world = world

    # Place settlement on (1,1), get initial population
    game.place_initial_settlement(1, 1)
    initial_pop = game.player_faction.citizens.count

    # Simulate save at t=1000.0
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Simulate load at t=1005.0 → ticks = (1005 - 1000) // TICK_DURATION = 5
    monkeypatch.setattr(persistence.time, "time", lambda: 1005.0)
    ticks = int((1005.0 - 1000.0) // persistence.TICK_DURATION)

    # Values: one ORE per 3 calls of random.randint
    values = [1, 0, 0] * ticks
    monkeypatch.setattr(random, "randint", lambda a, b: values.pop(0))

    loaded_state, _ = persistence.load_state(world=world, factions=[game.player_faction])

    # Population: initial_pop + ticks
    assert loaded_state.population == initial_pop + ticks

    # Total ORE: each neighbor had 1 → 6 neighbors × 1 ORE = 6 per tick, but 
    # because random.randint returns 1 only once per 3 calls on average,
    # we expect exactly 25 ORE after 5 ticks (since [1,0,0] repeats).
    assert loaded_state.resources[player][ResourceType.ORE] == 25


def test_offline_processing_buildings_convert_all_ore_to_metal(initialized_game, monkeypatch):
    """
    If the player has a Smeltery and starts with 4 ORE, after 3 offline ticks with random.randint→0,
    all 4 ORE should convert to 4 METAL and ORE should go to zero.
    """
    game, player = initialized_game

    # Create a pristine world (no resources on neighbors)
    world = make_world(resource_per_tile={})
    game.world = world
    game.place_initial_settlement(1, 1)

    # Give the faction 4 ORE and 0 METAL
    faction = game.player_faction
    faction.resources = {
        ResourceType.ORE: 4,
        ResourceType.METAL: 0,
        ResourceType.FOOD: 0,
    }
    game.resources.data[player] = faction.resources.copy()

    # Build a Smeltery so that ORE → METAL conversion can happen
    from game.buildings import Smeltery
    faction.buildings.append(Smeltery())

    # Simulate save at t=1000.0
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Simulate 3 seconds offline: ticks = (1003 - 1000) // 1 = 3
    monkeypatch.setattr(persistence.time, "time", lambda: 1003.0)
    ticks = int((1003.0 - 1000.0) // persistence.TICK_DURATION)

    # Make random.randint always return 0 → no new ORE from neighbors
    monkeypatch.setattr(random, "randint", lambda a, b: 0)

    loaded_state, _ = persistence.load_state(world=world, factions=[faction])

    # After processing, all initial 4 ORE must have been smelted into 4 METAL
    assert faction.resources[ResourceType.METAL] == 4
    assert faction.resources[ResourceType.ORE] == 0


def test_begin_applies_saved_state_to_faction(tmp_path, monkeypatch):
    """
    Directly saving state via game.state and then calling new_game.begin()
    should restore resources and citizens count correctly on the new faction object.
    """
    # --- Setup and patching ---
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    # Build a world and initial game
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name

    # Manually set some faction‐state
    game.player_faction.resources[ResourceType.FOOD] = 10
    game.player_faction.citizens.count = 12
    game.resources.data[player][ResourceType.FOOD] = 10

    # Reflect that in game.state before saving
    game.state.resources = game.resources.data
    game.state.population = game.player_faction.citizens.count
    game.state.factions = {
        player: {
            "citizens": 12,
            "workers": game.player_faction.workers.assigned,
            "buildings": [],
            "projects": [],
        }
    }

    # Save via game.save() to ensure game.save() pathway is covered
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Create a brand‐new Game, place the settlement, then call begin() to load saved state
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    new_game = Game(world=world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    new_faction = new_game.player_faction
    assert new_faction.resources[ResourceType.FOOD] == 10
    assert new_faction.citizens.count == 12


def test_offline_project_completion(initialized_game, monkeypatch):
    """
    If a faction starts a Great Project (Grand Cathedral) and then goes offline for enough ticks,
    upon loading, that project's progress should be at full build_time and get marked complete.
    """
    game, player = initialized_game

    # Start a Great Project called "Grand Cathedral"
    from copy import deepcopy
    template = GREAT_PROJECT_TEMPLATES["Grand Cathedral"]
    project = deepcopy(template)
    faction = game.player_faction
    faction.start_project(project, claimed_projects=game.claimed_projects)

    # Save at t=1000.0
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Simulate offline exactly build_time ticks
    offline_time = 1000.0 + project.build_time * persistence.TICK_DURATION
    monkeypatch.setattr(persistence.time, "time", lambda: offline_time)

    # Make random.randint always 0 (no extra resources)
    monkeypatch.setattr(random, "randint", lambda a, b: 0)

    loaded_state, _ = persistence.load_state(world=game.world, factions=[faction])

    # Confirm that the in‐memory faction’s project is complete
    assert faction.projects[0].is_complete()

    # Also verify saved state's project progress
    assert loaded_state.factions[player]["projects"][0]["progress"] == project.build_time


def test_population_persists_between_sessions(initialized_game, monkeypatch):
    """
    If AI_FACTION_COUNT=0, then population increments should persist from one Game session
    to the next via Game.begin().
    """
    game, player = initialized_game
    initial_pop = game.player_faction.citizens.count

    # Save at t=1000.0
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Go offline until t=1005.0 → ticks = 5
    monkeypatch.setattr(persistence.time, "time", lambda: 1005.0)
    ticks = int((1005.0 - 1000.0) // persistence.TICK_DURATION)
    values = [1, 0, 0] * ticks
    monkeypatch.setattr(random, "randint", lambda a, b: values.pop(0))

    # Start a new Game instance
    new_game = Game(world=game.world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    # The new faction’s population should have increased by exactly `ticks`
    assert new_game.player_faction.citizens.count == initial_pop + ticks


def test_extended_state_roundtrip(initialized_game):
    """
    Confirm that roads, event_turn_counters, tech_level, god_powers all survive a save/load cycle.
    """
    game, player = initialized_game

    # Add a road: from (0,0) to (1,1)
    game.world.add_road((0, 0), (1, 1))

    # Manually tweak faction: tech_level=3, god_powers={"smite":1}
    faction = game.player_faction
    faction.tech_level = 3
    faction.god_powers = {"smite": 1}

    # Tweak an event counter
    game.event_turn_counters = {"raid": 5}

    # Save at current patched time (we don't need a monkeypatch here since nothing else is time-dependent)
    game.save()

    # Create new Game and call begin()
    new_game = Game(world=game.world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    # Check that road persisted
    assert any(
        r.start == (0, 0) and r.end == (1, 1)
        for r in new_game.world.roads
    )
    # Check that event_turn_counters survived
    assert new_game.event_turn_counters == {"raid": 5}

    # Check that faction tech_level and god_powers survived
    from game.technology import TechLevel
    assert new_game.player_faction.tech_level == TechLevel.INDUSTRIAL
    assert new_game.player_faction.god_powers == {"smite": 1}


def test_buildings_persist_across_save(initialized_game):
    """
    Given a faction that builds a Farm and a House, after save/load,
    those buildings should still exist, and their resource counts persist.
    """
    game, player = initialized_game

    from game.buildings import Farm, House

    faction = game.player_faction
    faction.resources[ResourceType.WOOD] = 300

    faction.build_structure(Farm())
    faction.build_structure(House())
    game.resources.data[player] = faction.resources.copy()

    game.save()

    new_game = Game(world=game.world)
    new_game.place_initial_settlement(1, 1)
    new_game.begin()

    new_faction = new_game.player_faction
    assert any(b.name == "Farm" for b in new_faction.buildings)
    assert any(b.name == "House" for b in new_faction.buildings)
    assert new_faction.resources[ResourceType.WOOD] == faction.resources[ResourceType.WOOD]


def test_offline_progress_after_reload(initialized_game, monkeypatch):
    """
    If a faction has 4 ORE, a Smeltery, and is building a Great Project, 
    after going offline for enough ticks, on reload:
      - The Smeltery should have smelted all ORE → METAL
      - The Great Project’s progress should be at build_time
      - The persisted state in loaded_state should match in-memory changes
    """
    game, player = initialized_game

    # Setup a world with no extra ORE on neighbor tiles
    world = make_world(resource_per_tile={})
    game.world = world
    game.place_initial_settlement(1, 1)

    faction = game.player_faction
    # Assign 4 ORE to the faction
    faction.resources = {
        ResourceType.ORE: 4,
        ResourceType.METAL: 0,
        ResourceType.FOOD: 0,
    }
    game.resources.data[player] = faction.resources.copy()

    # Build a Smeltery so ORE → METAL can happen offline
    from game.buildings import Smeltery
    faction.buildings.append(Smeltery())

    # Start a Great Project (Grand Cathedral)
    from copy import deepcopy
    template = GREAT_PROJECT_TEMPLATES["Grand Cathedral"]
    project = deepcopy(template)
    faction.start_project(project, claimed_projects=game.claimed_projects)

    # Save at t=1000.0
    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    # Go offline exactly build_time ticks
    offline_time = 1000.0 + project.build_time * persistence.TICK_DURATION
    monkeypatch.setattr(persistence.time, "time", lambda: offline_time)

    # Force random.randint→0 so no new ORE from neighbors
    monkeypatch.setattr(random, "randint", lambda a, b: 0)

    loaded_state, _ = persistence.load_state(world=world, factions=[faction])

    # In‐memory faction: ORE→0, METAL→4, project fully progressed
    assert faction.resources[ResourceType.METAL] == 4
    assert faction.resources[ResourceType.ORE] == 0
    assert faction.projects[0].progress == project.build_time

    # Persisted state: matches loaded_state as well
    assert loaded_state.resources[player][ResourceType.METAL] == 4
    assert loaded_state.factions[player]["projects"][0]["progress"] == project.build_time


def test_save_load_empty_world(tmp_path, monkeypatch):
    """
    If we save a Game whose World is 0×0 (no tiles), loading it back should not error.
    """

    # Patch SAVE_FILE
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)

    # Create an “empty” world
    empty_world = World(width=0, height=0)
    game = Game(world=empty_world)
    game.place_initial_settlement(0, 0)  # This may create a 1×1 “default” tile, depending on implementation

    # Save and reload
    monkeypatch.setattr(persistence.time, "time", lambda: 2000.0)
    game.save()

    # On load, ensure no exceptions, and world is still empty or minimal
    loaded_state, _ = persistence.load_state(world=empty_world, factions=[game.player_faction])

    # If the implementation creates at least one tile at (0, 0), verify that coordinates match
    if empty_world.get(0, 0):
        assert empty_world.get(0, 0).terrain in (None, "")  # depending on defaults
    else:
        # Truly empty world: assert loaded_state.world has no tiles
        assert all(loaded_state.world.width == 0 and loaded_state.world.height == 0 for _ in [empty_world])


def test_offline_gains_batched_large_elapsed(tmp_path, monkeypatch):
    """Offline gains should use batched calculation when elapsed ticks are large."""

    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)
    monkeypatch.setattr(persistence, "MAX_TICKS_BATCH", 3)

    world = make_world({ResourceType.WOOD: 1})
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save()

    offline_ticks = persistence.MAX_TICKS_BATCH + 5
    new_time = 1000.0 + offline_ticks * persistence.TICK_DURATION
    monkeypatch.setattr(persistence.time, "time", lambda: new_time)
    monkeypatch.setattr(random, "randint", lambda a, b: 0)

    from game.resources import ResourceManager
    original = ResourceManager.get_per_tick_output
    call_count = {"n": 0}

    def wrapped(self, fac):
        call_count["n"] += 1
        return original(self, fac)

    monkeypatch.setattr(ResourceManager, "get_per_tick_output", wrapped)

    loaded_state, _ = persistence.load_state(world=world, factions=[game.player_faction])

    assert call_count["n"] > 0
    expected = 50 + offline_ticks * 5
    assert loaded_state.resources[player][ResourceType.WOOD] == expected


# End of test_persistence.py
