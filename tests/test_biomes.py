from world.world import WorldSettings, World
from world.generation import generate_biome_map


def test_temperature_and_rainfall_maps():
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    assert len(world.temperature_map) == settings.height
    assert len(world.temperature_map[0]) == settings.width
    assert len(world.rainfall_map) == settings.height
    assert len(world.rainfall_map[0]) == settings.width


def test_biome_map_used_for_terrain():
    settings = WorldSettings(seed=2, width=4, height=4)
    world = World(width=settings.width, height=settings.height, settings=settings)
    expected = generate_biome_map(world.elevation_map, world.temperature_map, world.rainfall_map)
    terrains = [
        [world.get(q, r).terrain for q in range(settings.width)]
        for r in range(settings.height)
    ]
    assert terrains == expected
