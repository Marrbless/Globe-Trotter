# Python Strategy Game Prototypes

This repository contains modular prototypes for a turn-based strategy game. It includes:

- A **Faction Creation UI** to let players customize their faction
- A **Procedural World Generator** using hex tiles
- A **Game Runtime Module** that spawns AI factions after the player starts the game

---

## Requirements

* Python 3.10+
* [DearPyGui](https://github.com/hoffstadt/dearpygui) for all GUI windows
* [pytest](https://docs.pytest.org/) for running the test suite

Install the Python packages with:

```bash
pip install dearpygui pytest
```

## 1. Faction Creation UI

A simple GUI for experimenting with player faction creation.

**Location:** `ui/faction_creation.py`

### How to Run

```bash
python ui/faction_creation.py
```

## 2. Random Event System

The `game.events` module provides a lightweight framework for triggering
random events such as floods, droughts or raids during gameplay. Each
event affects a settlement's resources, population and buildings.
Probabilities are influenced by configurable event weights and the
world's weather settings, allowing different worlds to feel unique.

Example usage:

```python
from world.world import WorldSettings
from game.events import EventSystem, SettlementState

settings = WorldSettings()
events = EventSystem(settings, event_weights={"flood": 1.0, "drought": 1.0, "raid": 0.5})
state = SettlementState()

# Advance turns and print triggered events
for _ in range(10):
    event = events.advance_turn(state)
    if event:
        print("Triggered", event.name)
```

## Running the Game

Launch the main prototype loop with:

```bash
python main.py
```

This loads `save.json` if it exists, applying any resource gains that would have
accumulated while the game was not running.

## Saving Progress

Call `game.save()` from your own scripts to persist the current state to
`save.json`. The next time you run the game, `load_state()` will apply offline
progress based on the timestamp stored in this file.

## Running Tests

Execute the unit tests with:

```bash
pytest
```
