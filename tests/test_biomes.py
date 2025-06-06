from world.world import WorldSettings, World, determine_biome


def test_temperature_and_rainfall_attributes():
    settings = WorldSettings(seed=1, width=3, height=3)
    world = World(width=settings.width, height=settings.height, settings=settings)
    for r in range(settings.height):
        for q in range(settings.width):
            h = world.get(q, r)
            assert 0.0 <= h.temperature <= 1.0, f"Temperature out of bounds at ({q},{r}): {h.temperature}"
            assert 0.0 <= h.moisture <= 1.0, f"Moisture out of bounds at ({q},{r}): {h.moisture}"


def test_biome_map_used_for_terrain():
    # Disable water features so terrain is based solely on biome determination
    settings = WorldSettings(seed=2, width=4, height=4, rainfall_intensity=0.0)
    world = World(width=settings.width, height=settings.height, settings=settings)
    for r in range(settings.height):
        for q in range(settings.width):
            h = world.get(q, r)
            expected = determine_biome(h.elevation, h.temperature, h.moisture)
            assert h.terrain == expected, f"Biome mismatch at ({q},{r}): got {h.terrain}, expected {expected}"
