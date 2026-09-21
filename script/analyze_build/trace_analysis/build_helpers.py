# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Helper functions for full build analysis.
"""

from .parse_file import get_metadata
from .phase_breakdown import get_phase_breakdown
from .template_analysis import get_template_instantiation_events


def extract_all_data(df):
    """
    Extract metadata, phase breakdown, and template events from a parsed DataFrame.

    Args:
        df: Parsed DataFrame from parse_file()

    Returns:
        Dictionary with keys:
        - build_unit: Source file path starting from composable_kernel/
        - trace_file_path: Path to the original trace JSON file
        - metadata: Metadata dictionary
        - phase: Phase breakdown DataFrame
        - template: Template events DataFrame
    """
    return {
        "build_unit": df.attrs["sourceFile"],
        "trace_file_path": df.attrs.get("traceFilePath"),
        "metadata": get_metadata(df).__dict__,
        "phase": get_phase_breakdown(df).df,
        "template": get_template_instantiation_events(df),
    }


def get_trace_file(metadata_df, build_unit):
    """
    Get the trace file path for a given build unit.

    Args:
        metadata_df: Metadata DataFrame with trace_file_mapping in .attrs
        build_unit: Source file path (build unit name)

    Returns:
        Path to the trace JSON file, or None if not found

    Examples:
        >>> # Get trace file for a specific build unit
        >>> trace_path = get_trace_file(metadata_df, "library/src/tensor/gemm.cpp")
        >>> print(f"Trace file: {trace_path}")
        >>>
        >>> # Get trace files for slowest compilation units
        >>> slowest = metadata_df.nlargest(5, "total_wall_time_s")
        >>> for _, row in slowest.iterrows():
        ...     trace_path = get_trace_file(metadata_df, row['build_unit'])
        ...     print(f"{row['build_unit']}: {trace_path}")
    """
    mapping = metadata_df.attrs.get("trace_file_mapping", {})
    return mapping.get(build_unit)


def print_phase_hierarchy(phase_df):
    """
    Print cumulative phase times in a hierarchical tree structure.

    Args:
        phase_df: DataFrame with columns: name, parent, depth, duration, build_unit
                  (as created by concatenating phase breakdown results)
    """
    # Aggregate by phase name, parent, and depth
    phase_summary = (
        phase_df.groupby(["name", "parent", "depth"])
        .agg({"duration": "sum"})
        .reset_index()
    )

    # Convert to seconds
    phase_summary["duration_s"] = phase_summary["duration"] / 1_000_000

    # Calculate total time from root node only (depth == 0)
    # With branchvalues="total", parent nodes include their children's time,
    # so summing all phases would double/triple count nested values
    root_phases = phase_summary[
        (phase_summary["parent"] == "")
        | (phase_summary["parent"].isna())
        | (phase_summary["depth"] == 0)
    ].sort_values("duration_s", ascending=False)

    if len(root_phases) == 0:
        raise ValueError("No root phase found (depth == 0)")
    if len(root_phases) > 1:
        raise ValueError(f"Multiple root phases found: {root_phases['name'].tolist()}")

    total_time_s = root_phases.iloc[0]["duration_s"]

    print("=== Cumulative Phase Time Across Build ===")
    print(f"\nTotal compilation time: {total_time_s:,.1f} s")
    print("\nBreakdown by phase:")

    # Track which phases we've printed to handle hierarchy
    printed_phases = set()

    def print_phase_tree(df, parent_name, depth=0):
        """Recursively print phases in hierarchical order"""
        # Get children of this parent at the next depth level
        children = df[(df["parent"] == parent_name) & (df["depth"] == depth)]
        # Sort by duration descending within each level
        children = children.sort_values("duration_s", ascending=False)

        for _, row in children.iterrows():
            phase_name = row["name"]
            if phase_name in printed_phases:
                continue

            time_s = row["duration_s"]
            pct = 100 * time_s / total_time_s
            indent = "  " * depth
            # Create indented name and pad the whole thing to align colons
            indented_name = f"{indent}{phase_name}"
            print(f"{indented_name:32s}: {time_s:12,.1f} s ({pct:5.1f}%)")
            printed_phases.add(phase_name)

            # Recursively print children
            print_phase_tree(df, phase_name, depth + 1)

    for _, row in root_phases.iterrows():
        phase_name = row["name"]
        if phase_name in printed_phases:
            continue

        time_s = row["duration_s"]
        pct = 100 * time_s / total_time_s
        # Pad root phase name to align with children
        print(f"{phase_name:32s}: {time_s:12,.1f} s ({pct:5.1f}%)")
        printed_phases.add(phase_name)

        # Print children recursively
        print_phase_tree(phase_summary, phase_name, 1)
