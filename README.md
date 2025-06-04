# Python Strategy Game Prototypes

This repository contains modular prototypes for a turn-based strategy game. It includes:

- A **Faction Creation UI** to let players customize their faction
- A **Procedural World Generator** using hex tiles
- A **Game Runtime Module** that spawns AI factions after the player starts the game

---

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
Probabilities are influenced by the biome and weather settings from the
world generator, allowing different worlds to feel unique.

Example usage:

```python
from world.world import WorldSettings
from game.events import EventSystem, SettlementState

settings = WorldSettings()
events = EventSystem(settings)
state = SettlementState()

# Advance turns and print triggered events
for _ in range(10):
    event = events.advance_turn(state)
    if event:
        print("Triggered", event.name)
```
