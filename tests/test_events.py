import random
from game.events import EventSystem, SettlementState, Flood, Drought, Raid
from world.world import WorldSettings, World, RiverSegment
import pickle


def test_event_frequency_reduced():
    settings = WorldSettings(seed=42, width=5, height=5)
    world = World(width=settings.width, height=settings.height, settings=settings)
    rng = random.Random(0)
    system = EventSystem(settings, rng=rng)
    state = SettlementState(location=(0, 0))

    events = 0
    for _ in range(50):
        if system.advance_turn(state, world):
            events += 1
    assert events <= 6


def test_event_severity_varies_by_location():
    settings = WorldSettings(seed=42, width=5, height=5)
    world = World(width=settings.width, height=settings.height, settings=settings)

    state_a = SettlementState(location=(0, 0), resources=100, buildings=10)
    state_b = SettlementState(location=(3, 3), resources=100, buildings=10)

    event_a = Flood()
    event_b = Flood()

    event_a.apply(state_a, world)
    event_b.apply(state_b, world)

    assert (state_a.buildings != state_b.buildings) or (state_a.resources != state_b.resources)


def test_flood_converts_river_to_lake(monkeypatch):
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    coord = (1, 1)
    hex_ = world.get(*coord)
    hex_.river = True
    world.rivers.append(RiverSegment(coord, (1, 2)))
    state = SettlementState(location=coord)
    event = Flood()
    monkeypatch.setattr(event, "severity", lambda *_: 1.4)
    event.apply(state, world)
    assert hex_.lake
    assert not hex_.river
    assert coord in world.lakes
    assert all(seg.start != coord and seg.end != coord for seg in world.rivers)


def test_drought_drains_lake(monkeypatch):
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    coord = (1, 1)
    hex_ = world.get(*coord)
    hex_.lake = True
    hex_.terrain = "water"
    world.lakes.append(coord)
    state = SettlementState(location=coord)
    event = Drought()
    monkeypatch.setattr(event, "severity", lambda *_: 1.4)
    event.apply(state, world)
    assert not hex_.lake
    assert hex_.terrain == "plains"
    assert coord not in world.lakes


def test_raid_raises_hill(monkeypatch):
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    coord = (1, 1)
    hex_ = world.get(*coord)
    hex_.terrain = "hills"
    state = SettlementState(location=coord, buildings=10)
    event = Raid()
    monkeypatch.setattr(event, "severity", lambda *_: 1.4)
    event.apply(state, world)
    assert hex_.terrain == "mountains"


def test_disaster_intensity_affects_flood_damage():
    low = WorldSettings(seed=1, width=3, height=3, disaster_intensity=0.0)
    high = WorldSettings(seed=1, width=3, height=3, disaster_intensity=1.0)
    world_low = World(width=low.width, height=low.height, settings=low)
    world_high = World(width=high.width, height=high.height, settings=high)
    loc = (1, 1)
    state_low = SettlementState(location=loc, buildings=10)
    state_high = SettlementState(location=loc, buildings=10)
    Flood().apply(state_low, world_low)
    Flood().apply(state_high, world_high)
    assert state_high.buildings < state_low.buildings


def test_world_changes_disabled(monkeypatch):
    settings = WorldSettings(seed=1, width=3, height=3, world_changes=False)
    world = World(width=settings.width, height=settings.height, settings=settings)
    coord = (1, 1)
    state = SettlementState(location=coord)

    baseline = pickle.dumps(world)

    flood = Flood()
    monkeypatch.setattr(flood, "severity", lambda *_: 1.4)
    flood.apply(state, world)
    assert pickle.dumps(world) == baseline

    drought = Drought()
    monkeypatch.setattr(drought, "severity", lambda *_: 1.4)
    drought.apply(state, world)
    assert pickle.dumps(world) == baseline
