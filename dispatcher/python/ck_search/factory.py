# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build a vendored search strategy by name.

Same results as upstream ``make_strategy`` / ``_init_genomes`` (``api.py`` at
the pinned commit, see VENDOR.md) for the strategies vendored here; upstream
``api.py`` itself is not vendored because it imports the ROCKE build, measure
and database layers.

Warm starts are kernel records (the dicts the space was built from) or ready
``WarmStart`` objects, best first. A portfolio evaluates them before any
member asks; ``de``, ``hill``, ``tpe`` and ``gp`` take them (encoded and
repaired to the nearest listed config) as their initial genomes and measure
every one before their own picks (``de`` grows its population size, the size
of every generation, to the number of distinct warm starts); ``random`` and
``exhaustive`` ignore them. Only ``tpe`` and ``gp``
use ``prior``; a portfolio hands it to its members.
"""

import random
from typing import Any, List, Mapping, Optional, Sequence, Union

from .de import DifferentialEvolution
from .local import HillClimb
from .params import Genome, SearchSpace
from .portfolio import Portfolio, WarmStart
from .strategy import Exhaustive, RandomSearch, Strategy
from .surrogate import GPEI, TPE, Prior
from .trial import Trial

__all__ = ["DEFAULT_PORTFOLIO", "STRATEGY_NAMES", "make_strategy", "warm_starts"]

STRATEGY_NAMES = ("random", "exhaustive", "de", "hill", "tpe", "gp", "portfolio")
DEFAULT_PORTFOLIO = ("hill", "gp", "de")


def warm_starts(
    warm_records: Sequence[Union[Mapping[str, object], WarmStart]],
) -> List[WarmStart]:
    """``warm_records`` as ``WarmStart``s in order; a plain record at position
    ``i`` gets the source ``record:<i>``."""
    return [
        w if isinstance(w, WarmStart) else WarmStart(dict(w), f"record:{i}")
        for i, w in enumerate(warm_records)
    ]


class _ListedExhaustive(Exhaustive):
    """``Exhaustive`` over a config list: the same trials in the same order,
    without walking the product of the field levels. Upstream walks every
    genome of that product and skips the unlisted ones; for kernel records
    that vary in many fields the product runs to millions while the list
    holds a few, so a single ``ask`` could take minutes. Here ``_pos`` counts
    listed genomes instead of product positions."""

    def __init__(self, space: SearchSpace, seed: int = 0):
        super().__init__(space, seed)
        # enumerate() yields canonical genomes in lexicographic order.
        self._order = sorted({space.canonical_genome(g) for g in space.listed()})

    def ask(self, n: int = 1) -> List[Trial]:
        out: List[Trial] = []
        taken = self.pending_keys()
        while len(out) < int(n):
            if self._pos >= len(self._order):
                self._done = True
                break
            g = self._order[self._pos]
            self._pos += 1
            k = self.space.key(g)
            if k is None or k in self.history or k in taken:
                continue
            taken.add(k)
            out.append(self.trial(g))
        return out


def _init_genomes(
    space: SearchSpace, warm: Sequence[WarmStart], seed: int
) -> List[Genome]:
    """Upstream ``_init_genomes``: every warm start encoded and repaired,
    unrepairable ones dropped."""
    rng = random.Random(f"warm:{seed}")
    out = []
    for w in warm:
        r = space.repair(space.encode(w.config), rng)
        if r is not None:
            out.append(r)
    return out


def make_strategy(
    name: str,
    space: SearchSpace,
    seed: int = 0,
    *,
    warm_records: Sequence[Union[Mapping[str, object], WarmStart]] = (),
    prior: Optional[Prior] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Strategy:
    """A strategy by name (:data:`STRATEGY_NAMES`) over ``space``.

    ``options`` go to the strategy's constructor; a portfolio also reads
    ``members`` (default :data:`DEFAULT_PORTFOLIO`) from them. Raises
    ``ValueError`` for an unknown name or member list.
    """
    opts = dict(options or {})
    warm = warm_starts(warm_records)
    if name == "portfolio":
        members = tuple(opts.pop("members", DEFAULT_PORTFOLIO))
        bad = [m for m in members if m == "portfolio" or m not in STRATEGY_NAMES]
        if not members or bad:
            raise ValueError(
                f"make_strategy - members={members!r}, expected one or more of "
                f"{STRATEGY_NAMES[:-1]} (no nested portfolio)"
            )
        # A climber inside a portfolio starts from the others' finds, so it
        # skips its own stratified probe (upstream rule).
        built = [
            make_strategy(
                m,
                space,
                seed * 101 + i + 1,
                prior=prior,
                options={"n_init": 0} if m == "hill" else None,
            )
            for i, m in enumerate(members)
        ]
        return Portfolio(space, seed, members=built, warm=warm, **opts)
    if name == "random":
        return RandomSearch(space, seed, **opts)
    if name == "exhaustive":
        listed = space.listed() is not None
        return (_ListedExhaustive if listed else Exhaustive)(space, seed)
    if name not in STRATEGY_NAMES:
        raise ValueError(f"strategy={name!r}, expected one of {STRATEGY_NAMES}")
    init = _init_genomes(space, warm, seed)
    if name == "de":
        de_opts = {"variant": "current_to_pbest1bin", "adapt": "shade"}
        de_opts.update(opts)
        de = DifferentialEvolution(space, seed, init=init, **de_opts)
        # DE seeds its first population with the warm starts and drops those
        # past pop_size: grow pop_size (every generation's size) so every
        # distinct one is measured.
        n_warm = len({space.key(g) for g in init})
        if n_warm > de.pop_size:
            de_opts["pop_size"] = n_warm
            de = DifferentialEvolution(space, seed, init=init, **de_opts)
        return de
    if name == "tpe":
        return TPE(space, seed, prior=prior, init=init, **opts)
    if name == "gp":
        return GPEI(space, seed, prior=prior, init=init, **opts)
    # hill: the first warm start is the start point, the rest follow it.
    if "start" not in opts:
        opts["start"] = init[0] if init else None
        opts.setdefault("init", init[1:])
    return HillClimb(space, seed, **opts)
