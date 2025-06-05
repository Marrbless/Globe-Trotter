from world.world import WorldSettings, World

def test_rivers_generated():
    settings = WorldSettings(seed=1, width=5, height=5, rainfall_intensity=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)

    # Assert rivers were generated
    assert len(world.rivers) > 0, "Expected at least one river to be generated"

    # Assert at least one tile is a lake
    lakes_present = any(hex_.lake for hex_ in world.all_hexes())
    assert lakes_present, "Expected at least one lake to be generated"
