# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Main sampling orchestrator: Sobol -> LHS pad -> maximin refine.

Implements the Daily Tier sampling pipeline from the RFC:
  1. Sobol base draw (90% of budget)
  2. LHS padding for marginal axis coverage (10% reserve)
  3. Maximin simulated annealing post-pass (200 iterations)
"""

import random

from sampling.sobol import SobolSequence
from sampling.lhs import lhs_pad
from sampling.maximin import maximin_anneal
from sampling.feasible_set import GEMM_AXES, normalize_axis_values, normalize_point


def sample_feasible_set(feasible_set, budget, seed, axes=None, maximin_iterations=200):
    """Select `budget` items from `feasible_set` using Sobol + LHS + maximin.

    Args:
        feasible_set: List of parameter dicts (each with tile/trait keys).
        budget: Maximum number of items to select.
        seed: Integer seed for deterministic selection.
        axes: List of axis names (defaults to GEMM_AXES).
        maximin_iterations: SA iterations for maximin pass (default 200).

    Returns:
        Tuple of (selected_items, sampler_method_string).
    """
    n = len(feasible_set)
    if axes is None:
        axes = GEMM_AXES

    if budget >= n:
        return list(feasible_set), "full", list(range(n))

    if n == 0:
        return [], "empty", []

    rng = random.Random(seed)

    # Phase 1: Sobol base selection (fill as much as possible from Sobol)
    sobol = SobolSequence(d=1, scramble=True, seed=seed)
    raw_points = sobol.generate(min(budget * 4, n * 2))

    selected_indices = []
    seen = set()
    for pt in raw_points:
        idx = min(int(pt[0] * n), n - 1)
        if idx not in seen:
            seen.add(idx)
            selected_indices.append(idx)
            if len(selected_indices) >= budget:
                break

    # If Sobol didn't produce enough unique points, fill with RNG
    if len(selected_indices) < budget:
        remaining = list(set(range(n)) - seen)
        rng.shuffle(remaining)
        for idx in remaining:
            if len(selected_indices) >= budget:
                break
            selected_indices.append(idx)
            seen.add(idx)

    # Phase 2: LHS padding — swap in points that cover uncovered axis values
    available_axes = [ax for ax in axes if any(ax in item for item in feasible_set)]
    lhs_additions = lhs_pad(
        selected_indices,
        feasible_set,
        available_axes,
        max(0, budget - len(selected_indices)),
        rng,
    )
    for idx in lhs_additions:
        if idx not in seen:
            seen.add(idx)
            selected_indices.append(idx)

    # Trim to budget
    if len(selected_indices) > budget:
        selected_indices = selected_indices[:budget]

    # Phase 3: Maximin simulated annealing
    meta = normalize_axis_values(feasible_set, available_axes)
    all_coords = [normalize_point(item, available_axes, meta) for item in feasible_set]

    selected_indices = maximin_anneal(
        selected_indices,
        feasible_set,
        all_coords,
        iterations=maximin_iterations,
        rng=rng,
    )

    # Sort by original index for deterministic output order
    selected_indices.sort()

    selected_items = [feasible_set[i] for i in selected_indices]
    return selected_items, "sobol+lhs+maximin", selected_indices
