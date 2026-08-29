# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Latin Hypercube Sampling padding for marginal coverage.

Ensures every distinct value on every parameter axis appears at least once
in the selected sample, using a greedy set-cover heuristic.
"""

from collections import defaultdict


def lhs_pad(selected_indices, feasible_set, axes, budget_remaining, rng):
    """Add indices to guarantee marginal coverage on all axes.

    Args:
        selected_indices: Already-selected feasible-set indices.
        feasible_set: Full list of parameter dicts.
        axes: List of axis names to ensure coverage for.
        budget_remaining: Max additional indices to add.
        rng: random.Random instance for tie-breaking.

    Returns:
        List of additional indices to include.
    """
    if budget_remaining <= 0:
        return []

    selected_set = set(selected_indices)

    # Build per-axis coverage maps: axis -> value -> set of feasible indices with that value
    axis_value_indices = {}
    for ax in axes:
        value_map = defaultdict(set)
        for i, item in enumerate(feasible_set):
            if ax in item:
                value_map[str(item[ax])].add(i)
        axis_value_indices[ax] = value_map

    # Find which axis values are already covered
    covered = {}
    for ax in axes:
        covered[ax] = set()
        for idx in selected_set:
            if ax in feasible_set[idx]:
                covered[ax].add(str(feasible_set[idx][ax]))

    # Find uncovered axis values
    uncovered_pairs = []  # (axis, value) pairs not yet covered
    for ax in axes:
        for val in axis_value_indices[ax]:
            if val not in covered[ax]:
                uncovered_pairs.append((ax, val))

    if not uncovered_pairs:
        return []

    # Greedy set-cover: pick indices that cover the most uncovered (axis, value) pairs
    additional = []
    uncovered_set = set(range(len(uncovered_pairs)))

    while uncovered_set and len(additional) < budget_remaining:
        # For each candidate index, count how many uncovered pairs it covers
        best_idx = -1
        best_count = 0
        best_covers = set()

        # Build candidate pool: indices that appear in at least one uncovered pair's index set
        candidates = set()
        for ui in uncovered_set:
            ax, val = uncovered_pairs[ui]
            candidates.update(axis_value_indices[ax][val])
        candidates -= selected_set
        candidates -= set(additional)

        if not candidates:
            break

        # Sample a subset to avoid O(N*U) when both are large
        candidate_list = list(candidates)
        if len(candidate_list) > 500:
            rng.shuffle(candidate_list)
            candidate_list = candidate_list[:500]

        for ci in candidate_list:
            item = feasible_set[ci]
            covers = set()
            for ui in uncovered_set:
                ax, val = uncovered_pairs[ui]
                if ax in item and str(item[ax]) == val:
                    covers.add(ui)
            if len(covers) > best_count:
                best_count = len(covers)
                best_idx = ci
                best_covers = covers

        if best_idx < 0:
            break

        additional.append(best_idx)
        uncovered_set -= best_covers
        # Update covered sets
        item = feasible_set[best_idx]
        for ax in axes:
            if ax in item:
                covered[ax].add(str(item[ax]))

    return additional
