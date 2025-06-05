from __future__ import annotations

from __future__ import annotations

"""Basic AI heuristics for diplomacy between factions."""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .game import Game
    from .models import Faction

from world.world import ResourceType


def evaluate_relations(game: "Game") -> None:
    """Evaluate diplomacy actions for all AI factions."""
    factions = [f for f in game.map.factions if f is not game.player_faction]
    for i, faction in enumerate(factions):
        for other in factions[i + 1 :]:
            _consider_trade(game, faction, other)
            _consider_alliance(game, faction, other)

    for truce in list(game.truces):
        f1, f2 = truce.factions
        _consider_break_truce(game, f1, f2)

    for alliance in list(game.alliances):
        f1, f2 = alliance.factions
        _consider_betrayal(game, f1, f2)


def _consider_trade(game: "Game", a: Faction, b: Faction) -> None:
    if game.is_at_war(a, b) or game.is_allied(a, b):
        return
    for deal in game.trade_deals:
        if {deal.faction_a, deal.faction_b} == {a, b}:
            return
    if random.random() < 0.2:
        game.form_trade_deal(a, b, {ResourceType.FOOD: 1}, {ResourceType.WOOD: 1})


def _consider_alliance(game: "Game", a: Faction, b: Faction) -> None:
    if game.is_at_war(a, b) or game.is_allied(a, b) or game.is_under_truce(a, b):
        return
    if random.random() < 0.05:
        game.form_alliance(a, b)


def _consider_break_truce(game: "Game", a: Faction, b: Faction) -> None:
    if not game.is_under_truce(a, b):
        return
    if random.random() < 0.1:
        game.break_truce(a, b)
        if random.random() < 0.5:
            game.declare_war(a, b)


def _consider_betrayal(game: "Game", a: Faction, b: Faction) -> None:
    if not game.is_allied(a, b):
        return
    stronger = a.citizens.count > b.citizens.count + 5
    if stronger and random.random() < 0.05:
        game.break_alliance(a, b)
        game.declare_war(a, b)
