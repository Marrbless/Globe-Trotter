import random
from game.events import EventSystem, SettlementState, Flood
from world.world import WorldSettings, World


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

