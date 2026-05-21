# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Maximin simulated annealing post-pass.

Improves minimum pairwise distance in the selected subset by swapping
points with unselected candidates. RFC specifies 200 iterations.

Uses a cached pairwise distance matrix to avoid O(n^2) recomputation
per iteration. Updates are O(n) per swap.
"""

import math


def _manhattan_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def maximin_anneal(
    selected_indices, feasible_set, normalized_coords, iterations=200, rng=None
):
    """Improve minimum pairwise distance via simulated annealing.

    Args:
        selected_indices: List of indices into feasible_set (will be modified in-place).
        feasible_set: Full list of parameter dicts (not modified).
        normalized_coords: List of normalized coordinate vectors, one per feasible-set item.
        iterations: Number of SA iterations (default 200 per RFC).
        rng: random.Random instance.

    Returns:
        Modified selected_indices list.
    """
    import random as random_mod

    if rng is None:
        rng = random_mod.Random(42)

    n = len(selected_indices)
    if n < 3:
        return selected_indices

    all_indices = set(range(len(feasible_set)))
    selected_set = set(selected_indices)
    unselected = list(all_indices - selected_set)

    if not unselected:
        return selected_indices

    sel_coords = [normalized_coords[i] for i in selected_indices]

    # Build per-point minimum distance cache: for each point, store its
    # minimum distance to any other selected point and the index of that neighbor
    min_dists = [float("inf")] * n
    min_neighbors = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            d = _manhattan_distance(sel_coords[i], sel_coords[j])
            if d < min_dists[i]:
                min_dists[i] = d
                min_neighbors[i] = j
            if d < min_dists[j]:
                min_dists[j] = d
                min_neighbors[j] = i

    for iteration in range(iterations):
        t = 1.0 - (iteration / iterations) * 0.99

        # Find the point with the globally smallest min_dist (half of closest pair)
        victim_pos = min(range(n), key=lambda i: min_dists[i])
        old_min_dist = min_dists[victim_pos]
        victim_idx = selected_indices[victim_pos]

        # Pick a random unselected candidate
        candidate_pos = rng.randint(0, len(unselected) - 1)
        candidate_idx = unselected[candidate_pos]
        candidate_coord = normalized_coords[candidate_idx]

        # Compute candidate's min distance to all other selected points
        new_cand_min = float("inf")
        for k in range(n):
            if k == victim_pos:
                continue
            d = _manhattan_distance(candidate_coord, sel_coords[k])
            if d < new_cand_min:
                new_cand_min = d

        delta = new_cand_min - old_min_dist
        accept = delta > 0
        if not accept and t > 0.001:
            try:
                prob = math.exp(delta / t)
                accept = rng.random() < prob
            except (OverflowError, ValueError):
                accept = False

        if accept:
            unselected[candidate_pos] = victim_idx
            selected_indices[victim_pos] = candidate_idx
            sel_coords[victim_pos] = candidate_coord

            # Recompute min_dists for the swapped position and any point
            # whose nearest neighbor was the victim
            for k in range(n):
                if k == victim_pos:
                    # Recompute for the new point
                    min_dists[k] = float("inf")
                    min_neighbors[k] = 0
                    for j in range(n):
                        if j == k:
                            continue
                        d = _manhattan_distance(sel_coords[k], sel_coords[j])
                        if d < min_dists[k]:
                            min_dists[k] = d
                            min_neighbors[k] = j
                elif min_neighbors[k] == victim_pos:
                    # Nearest neighbor was replaced — full recompute for this point
                    min_dists[k] = float("inf")
                    for j in range(n):
                        if j == k:
                            continue
                        d = _manhattan_distance(sel_coords[k], sel_coords[j])
                        if d < min_dists[k]:
                            min_dists[k] = d
                            min_neighbors[k] = j
                else:
                    # Check if the new point is closer than current minimum
                    d = _manhattan_distance(sel_coords[k], sel_coords[victim_pos])
                    if d < min_dists[k]:
                        min_dists[k] = d
                        min_neighbors[k] = victim_pos

    return selected_indices
