from world.world import WorldSettings, World


def test_fantasy_features_applied():
    settings = WorldSettings(seed=5, width=5, height=5, fantasy_level=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)
    terrains = {world.get(q, r).terrain for r in range(settings.height) for q in range(settings.width)}
    ley_lines = any(
        world.get(q, r).ley_line
        for r in range(settings.height)
        for q in range(settings.width)
    )
    assert (
        "floating_island" in terrains
        or "crystal_forest" in terrains
        or "faerie_forest" in terrains
        or ley_lines
    )


def test_fantasy_resources_regenerated():
    settings = WorldSettings(seed=6, width=6, height=6, fantasy_level=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)
    fantasy_tiles = [
        world.get(q, r)
        for r in range(settings.height)
        for q in range(settings.width)
        if world.get(q, r).terrain in {"floating_island", "crystal_forest", "faerie_forest"}
    ]
    assert fantasy_tiles
    assert any(tile.resources for tile in fantasy_tiles)
