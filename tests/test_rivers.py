from world.world import WorldSettings, World

def test_rivers_generated():
    settings = WorldSettings(seed=1, width=5, height=5, rainfall_intensity=1.0)
    world = World(width=settings.width, height=settings.height, settings=settings)

    # Assert rivers were generated
    assert len(world.rivers) > 0, "Expected at least one river to be generated"

    # Assert at least one tile is a lake
    lakes_present = any(hex_.lake for hex_ in world.all_hexes())
    assert lakes_present, "Expected at least one lake to be generated"


def test_persistent_lakes_and_merging():
    settings = WorldSettings(seed=5, width=8, height=8, rainfall_intensity=5.0)
    world = World(width=settings.width, height=settings.height, settings=settings)

    persistent = [world.get(*c) for c in world.lakes if world.get(*c).persistent_lake]
    assert len(persistent) > 0, "Expected at least one persistent lake"

    merge_counts = {}
    for seg in world.rivers:
        merge_counts[seg.end] = merge_counts.get(seg.end, 0) + 1
    assert any(count > 1 for count in merge_counts.values()), "Expected at least one river merge"
