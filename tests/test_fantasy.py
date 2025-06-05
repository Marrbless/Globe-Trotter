from world.world import WorldSettings, World


def test_fantasy_features_applied():
    settings = WorldSettings(seed=5, width=5, height=5, fantasy_level=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)
    terrains = {world.get(q, r).terrain for r in range(settings.height) for q in range(settings.width)}
    assert "floating_island" in terrains or "crystal_forest" in terrains
