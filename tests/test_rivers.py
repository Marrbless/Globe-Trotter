from world.world import WorldSettings, World

def test_rivers_generated():
    settings = WorldSettings(seed=1, width=5, height=5, rainfall_intensity=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)
    assert len(world.rivers) > 0
    assert any(hex_.lake for hex_ in world.all_hexes())
