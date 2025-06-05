import random
import game.persistence as persistence
from game.events import EventSystem, SettlementState, Earthquake, Hurricane
from world.world import WorldSettings, World


def test_event_frequency_scales_with_intensity():
    high = WorldSettings(seed=1, width=5, height=5, disaster_intensity=1.0)
    low = WorldSettings(seed=1, width=5, height=5, disaster_intensity=0.0)
    world_h = World(width=high.width, height=high.height, settings=high)
    world_l = World(width=low.width, height=low.height, settings=low)
    rng = random.Random(0)
    sys_h = EventSystem(high, rng=rng)
    rng2 = random.Random(0)
    sys_l = EventSystem(low, rng=rng2)
    state_h = SettlementState(location=(0, 0))
    state_l = SettlementState(location=(0, 0))

    events_h = sum(1 for _ in range(100) if sys_h.advance_turn(state_h, world_h))
    events_l = sum(1 for _ in range(100) if sys_l.advance_turn(state_l, world_l))

    assert events_l < events_h


def test_terrain_change_persists(tmp_path, monkeypatch):
    settings = WorldSettings(seed=2, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    coord = (1, 1)
    state = SettlementState(location=coord)
    eq = Earthquake()
    monkeypatch.setattr(eq, "severity", lambda *_: 1.4)
    eq.apply(state, world)
    assert world.get(*coord).terrain == "mountains"

    hur = Hurricane()
    monkeypatch.setattr(hur, "severity", lambda *_: 1.4)
    hur.apply(state, world)
    assert world.get(*coord).terrain == "water"

    tmp_file = tmp_path / "save.json"
    monkeypatch.setattr(persistence, "SAVE_FILE", tmp_file)
    gs = persistence.GameState(timestamp=0, resources={}, population=0)
    gs.world = persistence.serialize_world(world)
    persistence.save_state(gs)

    new_world = World(width=settings.width, height=settings.height, settings=settings)
    persistence.load_state(world=new_world)
    assert new_world.get(*coord).terrain == "water"
