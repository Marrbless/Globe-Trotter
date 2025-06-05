from world.world import WorldSettings, World


def test_elevation_setting_affects_elevation_map():
    low = WorldSettings(seed=1, width=5, height=5, elevation=0.1)
    high = WorldSettings(seed=1, width=5, height=5, elevation=0.9)
    low_world = World(width=low.width, height=low.height, settings=low)
    high_world = World(width=high.width, height=high.height, settings=high)
    avg_low = sum(
        low_world.get(q, r).elevation
        for r in range(low.height)
        for q in range(low.width)
    ) / (low.width * low.height)
    avg_high = sum(
        high_world.get(q, r).elevation
        for r in range(high.height)
        for q in range(high.width)
    ) / (high.width * high.height)
    assert avg_high > avg_low
