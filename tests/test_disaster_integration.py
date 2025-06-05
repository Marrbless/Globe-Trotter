import random
import game.persistence as persistence
from game.events import EventSystem, SettlementState, Earthquake, Hurricane, Flood
from world.world import WorldSettings, World, RiverSegment


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


def test_flood_respects_world_changes(monkeypatch):
    coord = (1, 1)
    on = WorldSettings(seed=3, width=3, height=3, world_changes=True)
    off = WorldSettings(seed=3, width=3, height=3, world_changes=False)
    world_on = World(width=on.width, height=on.height, settings=on)
    world_off = World(width=off.width, height=off.height, settings=off)

    for world in (world_on, world_off):
        h = world.get(*coord)
        h.terrain = "hills"
        h.river = True
        world.rivers.append(RiverSegment(coord, (1, 2)))

    event_on = Flood()
    event_off = Flood()
    monkeypatch.setattr(event_on, "severity", lambda *_: 1.4)
    monkeypatch.setattr(event_off, "severity", lambda *_: 1.4)
    state_on = SettlementState(location=coord)
    state_off = SettlementState(location=coord)
    event_on.apply(state_on, world_on)
    event_off.apply(state_off, world_off)

    h_on = world_on.get(*coord)
    h_off = world_off.get(*coord)
    assert h_on.lake and not h_on.river
    assert not h_off.lake and h_off.river


def test_earthquake_respects_world_changes(monkeypatch):
    coord = (1, 1)
    on = WorldSettings(seed=4, width=3, height=3, world_changes=True)
    off = WorldSettings(seed=4, width=3, height=3, world_changes=False)
    world_on = World(width=on.width, height=on.height, settings=on)
    world_off = World(width=off.width, height=off.height, settings=off)

    world_on.get(*coord).terrain = "hills"
    world_off.get(*coord).terrain = "hills"

    event_on = Earthquake()
    event_off = Earthquake()
    monkeypatch.setattr(event_on, "severity", lambda *_: 1.4)
    monkeypatch.setattr(event_off, "severity", lambda *_: 1.4)
    state_on = SettlementState(location=coord)
    state_off = SettlementState(location=coord)
    event_on.apply(state_on, world_on)
    event_off.apply(state_off, world_off)

    assert world_on.get(*coord).terrain == "mountains"
    assert world_off.get(*coord).terrain == "hills"


def test_hurricane_respects_world_changes(monkeypatch):
    coord = (1, 1)
    on = WorldSettings(seed=5, width=3, height=3, world_changes=True)
    off = WorldSettings(seed=5, width=3, height=3, world_changes=False)
    world_on = World(width=on.width, height=on.height, settings=on)
    world_off = World(width=off.width, height=off.height, settings=off)

    for world in (world_on, world_off):
        h = world.get(*coord)
        h.river = True
        world.rivers.append(RiverSegment(coord, (1, 2)))

    event_on = Hurricane()
    event_off = Hurricane()
    monkeypatch.setattr(event_on, "severity", lambda *_: 1.4)
    monkeypatch.setattr(event_off, "severity", lambda *_: 1.4)
    state_on = SettlementState(location=coord)
    state_off = SettlementState(location=coord)
    event_on.apply(state_on, world_on)
    event_off.apply(state_off, world_off)

    h_on = world_on.get(*coord)
    h_off = world_off.get(*coord)
    assert h_on.lake and not h_on.river
    assert not h_off.lake and h_off.river
