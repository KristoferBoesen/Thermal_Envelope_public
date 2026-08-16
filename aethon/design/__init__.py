"""
Design-space exploration: what cooling infrastructure does this waste need?

Given a waste stream (decay curve + campaign mass) and a matrix material, this
package sweeps canister designs and maps the result over the two variables the
user controls — radius and waste loading.  Nothing is ranked or filtered: the
trade-offs between fleet size, active-cooling duration, and time to repository
are the user's to weigh, and weighing them needs the whole space in view.
"""

from aethon.design.archetypes import BUILTIN_ARCHETYPES, resolve_archetypes
from aethon.design.canister import canister_count, canister_volume
from aethon.design.candidates import evaluate_candidates, parse_candidates
from aethon.design.search import run_exploration

__all__ = [
    "BUILTIN_ARCHETYPES",
    "resolve_archetypes",
    "canister_count",
    "canister_volume",
    "evaluate_candidates",
    "parse_candidates",
    "run_exploration",
]
