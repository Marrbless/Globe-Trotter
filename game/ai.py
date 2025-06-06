from __future__ import annotations

"""Basic AI heuristics for diplomacy between factions."""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .game import Game
    from .models import Faction

from world.world import ResourceType


def _resource_shortages(faction: "Faction") -> set[ResourceType]:
    """Return resources that the faction has very little of (<5)."""
    return {
        r for r, amt in faction.resources.items() if amt < 5
    }


def _resource_surpluses(faction: "Faction") -> set[ResourceType]:
    """Return resources that the faction has plenty of (>50)."""
    return {
        r for r, amt in faction.resources.items() if amt > 50
    }


def evaluate_relations(game: "Game", consider_player: bool = False) -> None:
    """Evaluate diplomacy actions for all AI factions.

    Parameters
    ----------
    consider_player: bool
        Include the player faction when evaluating diplomacy if True.
    """
    factions = [
        f for f in game.map.factions if consider_player or f is not game.player_faction
    ]
    for i, faction in enumerate(factions):
        for other in factions[i + 1 :]:
            _consider_trade(game, faction, other)
            _consider_alliance(game, faction, other)

    for truce in list(game.truces):
        f1, f2 = truce.factions
        if not consider_player and (
            f1 is game.player_faction or f2 is game.player_faction
        ):
            continue
        _consider_break_truce(game, f1, f2)

    for alliance in list(game.alliances):
        f1, f2 = alliance.factions
        if not consider_player and (
            f1 is game.player_faction or f2 is game.player_faction
        ):
            continue
        _consider_betrayal(game, f1, f2)


def _consider_trade(game: "Game", a: Faction, b: Faction) -> None:
    if game.is_at_war(a, b) or game.is_allied(a, b):
        return
    for deal in game.trade_deals:
        if (deal.faction_a is a and deal.faction_b is b) or (
            deal.faction_a is b and deal.faction_b is a
        ):
            return

    shortages_a = _resource_shortages(a)
    surpluses_a = _resource_surpluses(a)
    shortages_b = _resource_shortages(b)
    surpluses_b = _resource_surpluses(b)

    # If A needs something B has plenty of
    common_ab = shortages_a & surpluses_b
    if common_ab and random.random() < 0.4:
        res = next(iter(common_ab))
        game.form_trade_deal(a, b, {}, {res: 1})
        return

    # If B needs something A has plenty of
    common_ba = shortages_b & surpluses_a
    if common_ba and random.random() < 0.4:
        res = next(iter(common_ba))
        game.form_trade_deal(a, b, {res: 1}, {})


def _consider_alliance(game: "Game", a: Faction, b: Faction) -> None:
    if game.is_at_war(a, b) or game.is_allied(a, b) or game.is_under_truce(a, b):
        return

    shortages_a = _resource_shortages(a)
    shortages_b = _resource_shortages(b)
    surpluses_a = _resource_surpluses(a)
    surpluses_b = _resource_surpluses(b)

    complementary = (shortages_a & surpluses_b) and (shortages_b & surpluses_a)

    if complementary and random.random() < 0.1:
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
    strength_a = a.citizens.count + getattr(a, "army_size", 0)
    strength_b = b.citizens.count + getattr(b, "army_size", 0)
    if strength_b == 0:
        strength_b = 1
    ratio = strength_a / strength_b
    if ratio > 1.5 and random.random() < 0.2:
        game.break_alliance(a, b)
        # cancel any trade deals between the factions
        game.trade_deals = [
            d
            for d in game.trade_deals
            if not ((d.faction_a is a and d.faction_b is b) or (d.faction_a is b and d.faction_b is a))
        ]
        game.declare_war(a, b)
