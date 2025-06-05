from world.world import WorldSettings, World


def test_elevation_setting_affects_elevation_map():
    low = WorldSettings(seed=1, width=5, height=5, elevation=0.1)
    high = WorldSettings(seed=1, width=5, height=5, elevation=0.9)
    low_world = World(width=low.width, height=low.height, settings=low)
    high_world = World(width=high.width, height=high.height, settings=high)
    avg_low = sum(sum(row) for row in low_world.elevation_map) / (low.width * low.height)
    avg_high = sum(sum(row) for row in high_world.elevation_map) / (high.width * high.height)
    assert avg_high > avg_low
