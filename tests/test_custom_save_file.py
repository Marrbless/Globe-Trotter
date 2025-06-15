import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from world.world import World, ResourceType
import game.persistence as persistence
from game import settings

def test_save_to_custom_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AI_FACTION_COUNT", 0)
    world = World(width=3, height=3)
    game = Game(world=world)
    game.place_initial_settlement(1, 1)

    # Modify resources so we can verify persistence
    faction = game.player_faction
    faction.resources[ResourceType.FOOD] = 5
    game.resources.data[faction.name][ResourceType.FOOD] = 5

    custom_path = tmp_path / "mysave.json"

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    game.save(save_file=custom_path)

    assert custom_path.exists()

    monkeypatch.setattr(persistence.time, "time", lambda: 1000.0)
    loaded_state, _ = persistence.load_state(save_file=custom_path)

    assert loaded_state.resources[faction.name][ResourceType.FOOD] == 5


