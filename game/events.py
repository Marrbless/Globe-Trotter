from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from world.world import WorldSettings


@dataclass
class SettlementState:
    """Simple container tracking the state of a settlement."""

    resources: int = 100
    population: int = 100
    buildings: int = 10
    defenses: int = 0


class Event:
    """Base class for random world events."""

    name: str = "event"

    def apply(self, state: SettlementState) -> None:
        raise NotImplementedError


class Flood(Event):
    name = "flood"

    def apply(self, state: SettlementState) -> None:
        # Damage buildings. Defenses mitigate some damage.
        loss = max(0, int(0.3 * state.buildings) - state.defenses)
        state.buildings = max(0, state.buildings - loss)
        # Resources lost proportional to building loss
        res_loss = min(state.resources, loss * 2)
        state.resources -= res_loss


class Drought(Event):
    name = "drought"

    def apply(self, state: SettlementState) -> None:
        res_loss = int(0.2 * state.resources)
        state.resources = max(0, state.resources - res_loss)
        pop_loss = int(0.1 * state.population)
        state.population = max(0, state.population - pop_loss)


class Raid(Event):
    name = "raid"

    def apply(self, state: SettlementState) -> None:
        # Defenses reduce impact
        effective = max(1, 5 - state.defenses)
        res_loss = min(state.resources, effective * 3)
        bld_loss = max(0, effective - 1)
        state.resources -= res_loss
        state.buildings = max(0, state.buildings - bld_loss)


ALL_EVENTS: List[type[Event]] = [Flood, Drought, Raid]


class EventSystem:
    """Schedules and triggers events based on world settings."""

    def __init__(self, settings: WorldSettings, rng: Optional[random.Random] = None):
        self.settings = settings
        self.rng = rng or random.Random()
        self.turn_counter = 0
        self.next_event_turn = self._schedule_next()

    def _schedule_next(self) -> int:
        # Time until next event influenced by weather randomness
        base = self.rng.randint(2, 5)
        weather_factor = 1.0 + self.settings.moisture - 0.5
        delay = max(1, int(base * (1.0 + weather_factor)))
        return self.turn_counter + delay

    def _choose_event(self) -> Event:
        flood_w = self.settings.weather_patterns.get("rain", 0.1) * self.settings.moisture
        drought_w = self.settings.weather_patterns.get("dry", 0.1) * (1 - self.settings.moisture)
        raid_w = 0.2 + self.settings.biome_distribution.get("plains", 0)
        weights = [flood_w, drought_w, raid_w]
        event_cls = self.rng.choices(ALL_EVENTS, weights=weights, k=1)[0]
        return event_cls()

    def advance_turn(self, state: SettlementState) -> Optional[Event]:
        """Advance the internal clock and trigger events when scheduled."""

        self.turn_counter += 1
        if self.turn_counter >= self.next_event_turn:
            event = self._choose_event()
            event.apply(state)
            self.next_event_turn = self._schedule_next()
            return event
        return None

