import os
import sys
import json
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.game import Game
from game.world import World
import game.persistence as persistence


def make_world():
    w = World(width=3, height=3)
    center = (1, 1)
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        tile = w.get(center[0] + dq, center[1] + dr)
        if tile:
            tile["terrain"] = "plains"
    return w


def test_save_and_load(tmp_path, monkeypatch):
    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    world = make_world()
    game = Game(world=world)
    game.place_initial_settlement(1, 1)
    player = game.player_faction.name
    game.resources.data[player]["food"] = 7
    game.save()

    loaded = persistence.load_state()
    assert loaded.resources[player]["food"] == 7

