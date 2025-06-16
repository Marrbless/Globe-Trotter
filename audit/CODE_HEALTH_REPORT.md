# Code Health Report

## Architecture / Layering
- Modules split into `game`, `ui`, and `world` packages.
- Strong coupling between world tiles and events; `Hex` dataclass recently refactored but tests expect older API.
- Persistence code uses pickle for world chunks and JSON for game state.
- UI modules depend on dearpygui, not available in CI causing type errors.

## Complexity Hot Spots
- `world/world.py` has 704 statements with average complexity `A` but many functions >100 lines.
- `game/game.py` is 478 statements, 69% coverage, with several large methods causing mypy errors.

## Test Coverage
- Overall coverage 68%.
- Low coverage modules: `main.py`, UI modules, and parts of `world/world.py` and `game/persistence.py`.

## Security Review
- `pickle` used for chunk caching (B301) – risky if files untrusted.
- Temporary files in `/tmp` (B108).
- Many uses of `random.Random` flagged (B311) but acceptable for non‑crypto use.
- No secrets detected.

## Performance Notes
- Perlin noise functions cached with LRU; water feature generation may be heavy but lazily evaluated.
- Disk serialization to `/tmp` per chunk may become bottleneck.

## Lint & Static Analysis
- `ruff` passes.
- `mypy` reports 78 errors, mostly missing attributes and stub packages.
- `bandit` reports 194 low‑severity, 3 medium issues.

## Prioritized Findings
- **P1** failing tests due to outdated API in `Hex` and missing `Game` methods.
- **P1** `pickle` usage for world chunks without validation (security risk).
- **P2** `mypy` errors across modules due to missing type definitions and libs.
- **P3** low test coverage on UI and main entry point.
