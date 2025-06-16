# Code Health Report - 2024-06-16

## Overview
- Project is a turn-based strategy game with world generation, resources, diplomacy, UI using DearPyGui.
- Contains ~8.4k lines of Python code. `world/world.py` and `game/game.py` are the largest modules.

## Test Results
- `pytest` => **29 failed**, 51 passed.
- Failures mostly due to missing methods (`available_powers`, `use_power`) and missing Hex attributes (`river`, `lake`, `flooded`).

## Lint Results (ruff)
- 777 violations reported (E402 import order, unused imports, etc.).

## Security Scan (bandit)
- 194 Low, 3 Medium issues. Example: use of insecure `random` for security-sensitive logic in `game/ai.py:78`.

## Complexity (radon)
- Average complexity grade A (3.88) across 398 blocks.

## Observations
- Code uses dataclasses heavily but some attributes are added dynamically (e.g., `Hex.lake`). Tests assume these exist by default, causing `AttributeError`.
- Game logic lacks implementations for god power mechanics (`available_powers`, `use_power`), causing tests to fail.
- Many modules miss docstrings and type hints are partial.
- Persistence module >600 LOC may benefit from splitting.

