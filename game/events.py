from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from world.world import WorldSettings, World
from world.generation import perlin_noise


@dataclass
class SettlementState:
    """Simple container tracking the state of a settlement."""

    resources: int = 100
    population: int = 100
    buildings: int = 10
    defenses: int = 0
    location: Tuple[int, int] | None = None


class Event:
    """Base class for random world events."""

    name: str = "event"

    def severity(self, state: SettlementState, world: World) -> float:
        """Return a location-based severity influenced by disaster intensity."""
        if not state.location:
            base = 1.0
        else:
            x, y = state.location
            n = perlin_noise(x, y, world.settings.seed, scale=0.1)
            base = max(0.5, min(1.5, 1 + (n - 0.5) * 2))
        return base * (1 + world.settings.disaster_intensity)

    def apply(self, state: SettlementState, world: World) -> None:
        raise NotImplementedError


class Flood(Event):
    name = "flood"

    def apply(self, state: SettlementState, world: World) -> None:
        sev = self.severity(state, world)
        # Damage buildings. Defenses mitigate some damage.
        loss = max(0, int(0.3 * state.buildings * sev) - state.defenses)
        state.buildings = max(0, state.buildings - loss)
        # Resources lost proportional to building loss
        res_loss = min(state.resources, int(loss * 2 * sev))
        state.resources -= res_loss
        if state.location:
            hex_ = world.get(*state.location)
            if hex_:
                hex_.flooded = True
                if sev > 1.3:
                    if hex_.river:
                        # convert rivers into lakes when flooding is severe
                        hex_.river = False
                        world.rivers = [
                            seg
                            for seg in world.rivers
                            if seg.start != hex_.coord and seg.end != hex_.coord
                        ]
                        if hex_.coord not in world.lakes:
                            world.lakes.append(hex_.coord)
                        hex_.lake = True
                        hex_.terrain = "water"
                    if hex_.terrain == "hills":
                        # severe flooding can reshape hills into mountains
                        hex_.terrain = "mountains"
                    else:
                        hex_.terrain = "water"
                        hex_.lake = True


class Drought(Event):
    name = "drought"

    def apply(self, state: SettlementState, world: World) -> None:
        sev = self.severity(state, world)
        res_loss = int(0.2 * state.resources * sev)
        state.resources = max(0, state.resources - res_loss)
        pop_loss = int(0.1 * state.population * sev)
        state.population = max(0, state.population - pop_loss)
        if state.location:
            hex_ = world.get(*state.location)
            if hex_:
                hex_.moisture = max(0.0, hex_.moisture - 0.1 * sev)
                if sev > 1.3:
                    if hex_.lake:
                        # severe drought dries up lakes completely
                        hex_.lake = False
                        hex_.terrain = "plains"
                        world.lakes = [c for c in world.lakes if c != hex_.coord]
                    else:
                        hex_.terrain = "desert"


class Raid(Event):
    name = "raid"

    def apply(self, state: SettlementState, world: World) -> None:
        sev = self.severity(state, world)
        # Defenses reduce impact
        effective = max(1, int((5 - state.defenses) * sev))
        res_loss = min(state.resources, effective * 3)
        bld_loss = max(0, effective - 1)
        state.resources -= res_loss
        state.buildings = max(0, state.buildings - bld_loss)
        if state.location:
            hex_ = world.get(*state.location)
            if hex_:
                hex_.ruined = True
                if sev > 1.3:
                    state.buildings = 0
                    if hex_.terrain == "hills":
                        # extreme raids can reshape the land into mountains
                        hex_.terrain = "mountains"


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
        base = self.rng.randint(5, 10)
        weather_factor = 1.0 + self.settings.moisture * 0.5
        intensity_factor = 1.0 - 0.5 * self.settings.disaster_intensity
        delay = max(2, int(base * weather_factor * intensity_factor))
        return self.turn_counter + delay

    def _choose_event(self) -> Event:
        flood_w = self.settings.weather_patterns.get("rain", 0.1) * self.settings.moisture
        drought_w = self.settings.weather_patterns.get("dry", 0.1) * (1 - self.settings.moisture)
        raid_w = 0.2 + self.settings.biome_distribution.get("plains", 0)
        weights = [flood_w, drought_w, raid_w]
        event_cls = self.rng.choices(ALL_EVENTS, weights=weights, k=1)[0]
        return event_cls()

    def advance_turn(self, state: SettlementState, world: World) -> Optional[Event]:
        """Advance the internal clock and trigger events when scheduled."""

        self.turn_counter += 1
        if self.turn_counter >= self.next_event_turn:
            event = self._choose_event()
            event.apply(state, world)
            self.next_event_turn = self._schedule_next()
            return event
        return None

