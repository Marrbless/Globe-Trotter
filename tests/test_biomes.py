from world.world import WorldSettings, World
from world.generation import determine_biome


def test_temperature_and_rainfall_attributes():
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    for r in range(settings.height):
        for q in range(settings.width):
            h = world.get(q, r)
            assert 0.0 <= h.temperature <= 1.0
            assert 0.0 <= h.moisture <= 1.0


def test_biome_map_used_for_terrain():
    settings = WorldSettings(seed=2, width=4, height=4)
    world = World(width=settings.width, height=settings.height, settings=settings)
    for r in range(settings.height):
        for q in range(settings.width):
            h = world.get(q, r)
            expected = determine_biome(h.elevation, h.temperature, h.moisture)
            assert h.terrain == expected
