# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Op-weighted budget allocation.

Distributes a total instance budget across active ops proportional to
their registered weights. Implements RFC section 3.4.
"""

import json
from pathlib import Path


_DEFAULT_WEIGHTS_FILE = Path(__file__).parent / "op_weights.json"


def load_op_weights(weights_file=None):
    """Load op weights from JSON file.

    Returns:
        Dict mapping op name to weight (float).
    """
    path = Path(weights_file) if weights_file else _DEFAULT_WEIGHTS_FILE
    with open(path) as f:
        data = json.load(f)
    return data["weights"]


def allocate_budget(total_budget, active_ops, weights, strict=True):
    """Distribute total_budget across active_ops proportional to weights.

    Args:
        total_budget: Total instance budget (e.g. 8000).
        active_ops: List of active op names.
        weights: Dict mapping op name to weight.
        strict: If True, raise ValueError for unweighted active ops.

    Returns:
        Dict mapping op name to allocated budget (int).
        Sum of allocations exactly equals total_budget.
    """
    if not active_ops:
        return {}

    # Check all active ops have weights
    missing = [op for op in active_ops if op not in weights]
    if missing and strict:
        raise ValueError(
            f"Active ops without registered weights: {missing}. "
            f"Add them to op_weights.json before running with sampling enabled."
        )

    # Compute weight sum for active ops only
    active_weights = {op: weights.get(op, 0.0) for op in active_ops}
    total_weight = sum(active_weights.values())

    if total_weight <= 0:
        # Equal distribution fallback
        per_op = total_budget // len(active_ops)
        alloc = {op: per_op for op in active_ops}
        remainder = total_budget - sum(alloc.values())
        for i, op in enumerate(active_ops):
            if i < remainder:
                alloc[op] += 1
        return alloc

    # Proportional allocation with floor
    alloc = {}
    for op in active_ops:
        alloc[op] = int(total_budget * active_weights[op] / total_weight)

    # Distribute remainder to highest-weight ops
    remainder = total_budget - sum(alloc.values())
    sorted_ops = sorted(active_ops, key=lambda op: active_weights[op], reverse=True)
    for i in range(remainder):
        alloc[sorted_ops[i % len(sorted_ops)]] += 1

    return alloc
