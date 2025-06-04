import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from game.population import FactionManager, Citizen, Worker
from game.game import Faction, Settlement, Position


def make_faction(pop=10):
    return Faction(name="Test", settlement=Settlement(name="Home", position=Position(0,0)), citizens=Citizen(count=pop), workers=Worker())

def test_tick_births(monkeypatch):
    faction = make_faction(10)
    mgr = FactionManager([faction])
    values = [2, 0, 0]
    def fake_randint(a, b):
        return values.pop(0)
    monkeypatch.setattr("random.randint", fake_randint)
    mgr.tick()
    assert faction.citizens.count == 12

def test_tick_deaths(monkeypatch):
    faction = make_faction(10)
    mgr = FactionManager([faction])
    values = [0, 3, 0]
    def fake_randint(a, b):
        return values.pop(0)
    monkeypatch.setattr("random.randint", fake_randint)
    mgr.tick()
    assert faction.citizens.count == 7

def test_tick_migration(monkeypatch):
    faction = make_faction(10)
    mgr = FactionManager([faction])
    values = [0, 0, -1]
    def fake_randint(a, b):
        return values.pop(0)
    monkeypatch.setattr("random.randint", fake_randint)
    mgr.tick()
    assert faction.citizens.count == 9
