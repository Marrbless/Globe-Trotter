import random
from game.events import EventSystem, SettlementState
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

