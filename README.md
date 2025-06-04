# Projects

This repository contains modules for procedural world generation.

## World Generation Module

The `world/generation.py` module provides utilities to create a hex based
world with configurable settings.

```
from world.generation import WorldSettings, adjust_settings, generate_hexes

settings = WorldSettings(seed=42, width=10, height=10)
adjust_settings(settings, moisture=0.7, elevation=0.3)
world = generate_hexes(settings)
hex_tile = world.get_hex(0, 0)
```

`WorldSettings` exposes sliders such as moisture, elevation and temperature to
influence the generated terrain.

