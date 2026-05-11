#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

"""
Analyze bank conflict profiling results from rocprofv3

This script processes the SQLite databases produced by rocprofv3 profiling
and provides a comprehensive comparison between plain and XOR transpose
implementations.

Usage:
    python3 analyze_bank_conflicts.py <plain_profile_dir> <xor_profile_dir>

Example:
    python3 analyze_bank_conflicts.py /tmp/plain /tmp/xor
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


def find_results_db(profile_dir: Path) -> Optional[Path]:
    """
    Find the results.db file in the profiling output directory.

    rocprofv3 creates a timestamped subdirectory containing results.db
    """
    db_files = list(profile_dir.glob("*/results.db"))
    if not db_files:
        return None
    return db_files[0]


def query_pmc_results(db_path: Path) -> Tuple[int, int]:
    """
    Query SQ_LDS_BANK_CONFLICT and SQ_INSTS_LDS from results.db

    Returns:
        Tuple of (conflicts, lds_instructions)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT counter_name, SUM(counter_value) as total
    FROM pmc_events
    WHERE counter_name IN ('SQ_LDS_BANK_CONFLICT', 'SQ_INSTS_LDS')
    GROUP BY counter_name
    """

    results = cursor.execute(query).fetchall()
    conn.close()

    data = dict(results)
    conflicts = data.get('SQ_LDS_BANK_CONFLICT', 0)
    lds_insts = data.get('SQ_INSTS_LDS', 0)

    return int(conflicts), int(lds_insts)


def print_separator(char='─', width=78):
    """Print a horizontal separator line"""
    print(char * width)


def print_header(title: str):
    """Print a formatted header box"""
    width = 78
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {title.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")
    print()


def print_analysis(plain_dir: Path, xor_dir: Path):
    """Print comprehensive analysis of both versions"""

    # Find results.db files
    plain_db = find_results_db(plain_dir)
    xor_db = find_results_db(xor_dir)

    if plain_db is None:
        print(f"Error: Could not find results.db in {plain_dir}", file=sys.stderr)
        sys.exit(1)

    if xor_db is None:
        print(f"Error: Could not find results.db in {xor_dir}", file=sys.stderr)
        sys.exit(1)

    # Query both databases
    plain_conflicts, plain_insts = query_pmc_results(plain_db)
    xor_conflicts, xor_insts = query_pmc_results(xor_db)

    # Handle edge cases
    if plain_insts == 0 or xor_insts == 0:
        print("Error: No LDS instructions found in profiling data!", file=sys.stderr)
        print("Make sure the kernels executed successfully.", file=sys.stderr)
        sys.exit(1)

    # Calculate rates
    plain_rate = (plain_conflicts / plain_insts * 100)
    xor_rate = (xor_conflicts / xor_insts * 100)
    plain_per_inst = plain_conflicts / plain_insts
    xor_per_inst = xor_conflicts / xor_insts

    # Calculate improvements
    conflict_reduction = plain_conflicts - xor_conflicts
    reduction_pct = (conflict_reduction / plain_conflicts * 100) if plain_conflicts > 0 else 0
    rate_improvement = plain_rate - xor_rate
    speedup = plain_per_inst / xor_per_inst if xor_per_inst > 0 else 0

    # Theoretical minimum (2-way conflicts for 64 threads / 32 banks)
    theoretical_min_conflicts_per_inst = 2.0
    theoretical_min_rate = 100.0  # 100% = 1 conflict per instruction = 2-way

    gap_to_optimal = xor_per_inst / theoretical_min_conflicts_per_inst

    # Print results
    print_header("Bank Conflict Analysis Results")

    # Main comparison table
    print(f"{'Metric':<35} {'Plain LDS':>20} {'XOR LDS':>20}")
    print_separator()
    print(f"{'SQ_LDS_BANK_CONFLICT':<35} {plain_conflicts:>20,} {xor_conflicts:>20,}")
    print(f"{'SQ_INSTS_LDS':<35} {plain_insts:>20,} {xor_insts:>20,}")
    print(f"{'Conflict Rate (%)':<35} {plain_rate:>20.1f} {xor_rate:>20.1f}")
    print(f"{'Conflicts per Instruction':<35} {plain_per_inst:>20.2f} {xor_per_inst:>20.2f}")
    print_separator()

    # Improvement metrics
    print()
    print(f"{'Improvement Metrics':<35} {'Value':>20}")
    print_separator()
    print(f"{'Conflict Reduction (absolute)':<35} {conflict_reduction:>20,}")
    print(f"{'Conflict Reduction (%)':<35} {reduction_pct:>20.1f}%")
    print(f"{'Rate Improvement (percentage points)':<35} {rate_improvement:>20.1f}")
    print(f"{'Speedup (conflicts/inst)':<35} {speedup:>20.2f}x")
    print_separator()

    # Theoretical analysis
    print()
    print(f"{'Theoretical Analysis':<35} {'Value':>20}")
    print_separator()
    print(f"{'Theoretical minimum (64t/32b)':<35} {theoretical_min_conflicts_per_inst:>20.1f}")
    print(f"{'Theoretical min rate (%)':<35} {theoretical_min_rate:>20.1f}%")
    print(f"{'Current XOR (conflicts/inst)':<35} {xor_per_inst:>20.2f}")
    print(f"{'Gap to theoretical optimal':<35} {gap_to_optimal:>20.2f}x")
    print_separator()

    # Interpretation
    print_header("Interpretation")

    print(f"✓ XOR swizzling reduces bank conflicts by {reduction_pct:.1f}%")
    print()
    print(f"  Plain LDS: ~{plain_per_inst:.1f}-way conflicts per LDS instruction")
    print(f"             ({plain_conflicts:,} total conflicts / {plain_insts:,} instructions)")
    print()
    print(f"  XOR LDS:   ~{xor_per_inst:.1f}-way conflicts per LDS instruction")
    print(f"             ({xor_conflicts:,} total conflicts / {xor_insts:,} instructions)")
    print()
    print(f"✓ Serialization improvement: {speedup:.1f}× fewer conflicts per instruction")
    print()
    print(f"✓ Theoretical minimum: ~{theoretical_min_conflicts_per_inst:.0f}-way conflicts")
    print(f"  (Pigeonhole principle: 64 threads / 32 banks = 2 threads per bank minimum)")
    print()
    print(f"✓ Gap to theoretical optimal: {gap_to_optimal:.1f}× away from best possible")
    print()

    # Performance impact estimation
    if plain_per_inst > 0:
        # Estimate transpose speedup (rough approximation)
        transpose_speedup = plain_per_inst / xor_per_inst if xor_per_inst > 0 else 1.0

        print(f"Estimated Performance Impact:")
        print(f"  If transpose is 10% of kernel time:")
        print(f"    Overall speedup: ~{(1.0 / (0.9 + 0.1/transpose_speedup) - 1.0) * 100:.1f}% faster")
        print(f"    (Amdahl's law: 90% unchanged + 10% × {transpose_speedup:.1f}× faster)")
        print()

    # Recommendations
    print_header("Recommendations")

    if xor_per_inst > theoretical_min_conflicts_per_inst * 2:
        print("⚠ Current XOR implementation is {:.1f}× from theoretical optimal.".format(gap_to_optimal))
        print()
        print("Potential further optimizations:")
        print("  1. Try 32×32 tiles instead of 64×32 (may achieve near-zero conflicts)")
        print("  2. Profile with omniperf for detailed bank conflict analysis")
        print("  3. Consider double buffering if LDS usage is not a constraint")
        print()
    else:
        print("✓ XOR implementation is close to theoretical optimal!")
        print()
        print(f"  Gap: {gap_to_optimal:.1f}× from theoretical minimum")
        print("  Further optimization may not be worth the complexity.")
        print()

    print("For detailed tutorial on bank conflicts and XOR swizzling, see:")
    print("  example/ck_tile/99_toy_tutorial/BANK_CONFLICT_TUTORIAL.md")
    print()


def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <plain_profile_dir> <xor_profile_dir>", file=sys.stderr)
        print(file=sys.stderr)
        print("Example:", file=sys.stderr)
        print(f"  {sys.argv[0]} /tmp/plain /tmp/xor", file=sys.stderr)
        sys.exit(1)

    plain_dir = Path(sys.argv[1])
    xor_dir = Path(sys.argv[2])

    # Validate directories exist
    if not plain_dir.exists():
        print(f"Error: Plain profile directory not found: {plain_dir}", file=sys.stderr)
        sys.exit(1)

    if not xor_dir.exists():
        print(f"Error: XOR profile directory not found: {xor_dir}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    print_analysis(plain_dir, xor_dir)


if __name__ == "__main__":
    main()
