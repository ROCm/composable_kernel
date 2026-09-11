# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for budget allocation, called by CMake at configure time.

Usage:
    python -m sampling.allocate_budget \
        --total-budget 8000 \
        --active-ops "gemm_universal,gemm_multi_d,gemm_preshuffle,grouped_gemm" \
        --output-dir /build/sampling_alloc \
        [--weights-file /path/to/op_weights.json]

Writes per-op budget files (e.g. gemm_universal_budget.txt) containing a single integer.
"""

import argparse
import json
import sys
from pathlib import Path


def _setup_path():
    _this_dir = Path(__file__).resolve().parent
    _tile_engine_dir = _this_dir.parent
    if str(_tile_engine_dir) not in sys.path:
        sys.path.insert(0, str(_tile_engine_dir))


_setup_path()

from sampling.budget import allocate_budget  # noqa: E402
from sampling.budget import load_op_weights  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Allocate instance budget across ops")
    parser.add_argument(
        "--total-budget",
        type=int,
        required=True,
        help="Total instance budget (e.g. 8000 for daily tier)",
    )
    parser.add_argument(
        "--active-ops",
        type=str,
        required=True,
        help="Comma or semicolon-separated list of active op names",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write per-op budget files",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="Path to op_weights.json (default: built-in)",
    )
    args = parser.parse_args()

    # Parse active ops (support both comma and semicolon separators)
    active_ops = [
        op.strip() for op in args.active_ops.replace(";", ",").split(",") if op.strip()
    ]

    if not active_ops:
        print("ERROR: No active ops specified", file=sys.stderr)
        sys.exit(1)

    weights = load_op_weights(args.weights_file)
    alloc = allocate_budget(args.total_budget, active_ops, weights, strict=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-op budget files
    for op, budget in alloc.items():
        budget_file = output_dir / f"{op}_budget.txt"
        budget_file.write_text(str(budget))

    # Write combined allocation metadata
    meta = {
        "total_budget": args.total_budget,
        "active_ops": active_ops,
        "allocations": alloc,
        "weights_used": {op: weights.get(op, 0.0) for op in active_ops},
    }
    meta_file = output_dir / "sampling_allocations.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print(f"Budget allocation (total={args.total_budget}):")
    for op, budget in sorted(alloc.items()):
        print(f"  {op}: {budget}")
    print(f"  Sum: {sum(alloc.values())}")


if __name__ == "__main__":
    main()
