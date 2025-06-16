Initial audit on 2024-06-16.
- Ran `pytest` -> 29 failures out of 80 tests.
- Ran `ruff` -> 777 lint errors (stored in /tmp/ruff.log).
- Installed bandit & radon via pip.
- `bandit -r .` -> 194 low, 3 medium issues, mostly insecure random usage in game/ai.py.
- `radon cc` -> average complexity A (3.88).
- Observed many failing tests due to missing attributes (`river`, `lake`, etc.) and missing methods (`available_powers`, `use_power`).
- World generator uses new WaterState enum but tests expect old attributes.
