# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Phase breakdown analysis for Clang -ftime-trace data.

This module provides hierarchical breakdown of compilation phases using
the pre-aggregated "Total" events from Clang's -ftime-trace output.

The data is returned as a PhaseBreakdown object with rich display and
analysis capabilities optimized for Jupyter notebooks.
"""

import pandas as pd
from collections import namedtuple
from typing import Optional


# Lightweight namedtuple for iteration
Phase = namedtuple("Phase", ["name", "depth", "duration", "duration_ms", "percentage"])


class PhaseBreakdown:
    """
    Wrapper for compilation phase breakdown with notebook-friendly API.

    Provides hierarchical view of compilation phases from Clang -ftime-trace,
    with rich display, filtering, and visualization capabilities.

    Examples:
        >>> breakdown = get_phase_breakdown(df)
        >>>
        >>> # Display in Jupyter
        >>> breakdown
        >>>
        >>> # Access specific phases
        >>> breakdown['InstantiateFunction']
        >>> breakdown.frontend
        >>> breakdown.backend
        >>>
        >>> # Get metrics
        >>> print(f"Total: {breakdown.total_ms:.1f}ms")
        >>>
        >>> # Top N analysis
        >>> breakdown.top(10)
        >>> breakdown.frontend.top(5)
        >>>
        >>> # Visualization
        >>> import plotly.express as px
        >>> data = breakdown.to_plotly()
        >>> fig = px.sunburst(**data)
        >>> fig.show()
        >>>
        >>> # Iteration
        >>> for phase in breakdown:
        >>>     print(f"{phase.name}: {phase.duration_ms:.1f}ms")
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize from phase breakdown DataFrame.

        Args:
            df: DataFrame with columns name, parent, depth, duration
        """
        if df.empty:
            self._df = pd.DataFrame(columns=["name", "parent", "depth", "duration"])
            self._total_time = 0
        else:
            self._df = df
            self._total_time = self._get_total_time()

    def __repr__(self) -> str:
        """Simple text representation for console."""
        if self._df.empty:
            return "PhaseBreakdown(empty)"
        n_phases = len(self._df)
        return f"PhaseBreakdown({n_phases} phases, {self._total_time:.1f}ms total)"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        if self._df.empty:
            return "<div><i>PhaseBreakdown(empty)</i></div>"
        return self.to_dataframe()._repr_html_()

    @property
    def df(self) -> pd.DataFrame:
        """
        Access underlying DataFrame.

        Returns:
            DataFrame with columns name, parent, depth, duration
        """
        return self._df

    def to_dataframe(self, show_percentages: bool = True) -> pd.DataFrame:
        """
        Format as DataFrame for display.

        Creates a nicely formatted DataFrame suitable for Jupyter notebook display.

        Args:
            show_percentages: Include percentage of total time

        Returns:
            DataFrame with formatted columns
        """
        return self._format_dataframe(show_percentages)

    def to_plotly(self) -> dict:
        """
        Convert to plotly hierarchical visualization format.

        Returns a dictionary with data_frame, values, and path that can be directly
        used with plotly.express sunburst, treemap, or icicle charts.

        Returns:
            Dictionary with keys: data_frame, values, path, branchvalues

        Example:
            >>> data = breakdown.to_plotly()
            >>> import plotly.express as px
            >>>
            >>> # Create sunburst chart
            >>> fig = px.sunburst(**data)
            >>> fig.show()
            >>>
            >>> # Create treemap chart
            >>> fig = px.treemap(**data)
            >>> fig.show()
            >>>
            >>> # Create icicle chart
            >>> fig = px.icicle(**data)
            >>> fig.show()
        """
        return self._build_plotly_data()

    # Internal helper methods

    def _get_total_time(self) -> int:
        """Get total time from root ExecuteCompiler event."""
        root = self._df[self._df["depth"] == 0]
        if root.empty:
            return 0
        return int(root.iloc[0]["duration"])

    def _format_dataframe(self, show_percentages: bool) -> pd.DataFrame:
        """Format phase breakdown as DataFrame."""
        if self._df.empty:
            return pd.DataFrame()

        display_rows = []
        for _, row in self._df.iterrows():
            duration_ms = row["duration"] / 1000.0
            display_row = {
                "Name": row["name"],
                "Parent": row["parent"] if row["parent"] else "(root)",
                "Depth": row["depth"],
                "Duration (ms)": duration_ms,
            }
            if show_percentages and self._total_time > 0:
                pct = row["duration"] / self._total_time * 100
                display_row["% of Total"] = pct
            display_rows.append(display_row)

        display_df = pd.DataFrame(display_rows)

        if show_percentages:
            display_df["% of Total"] = display_df["% of Total"].round(1)

        return display_df

    def _build_plotly_data(self) -> dict:
        """Convert to plotly hierarchical visualization format."""
        return {
            "data_frame": self._df,
            "names": "name",
            "parents": "parent",
            "values": "duration",
            "branchvalues": "total",
        }


# Hierarchical phase specification
# There are over 100 totals in the JSON file, but a lot of them overlap.
# If the children total more than their parent, we will throw a ValueError.
#
# The hierarchy is specified as a nested dictionary where:
# - Keys are phase names (matching "Total <name>" events in the trace)
# - Values are dictionaries of child phases (or empty dict {} for leaf nodes)
# - Empty string "" as a key means "calculate Other as residual"
#
# This structure supports arbitrary nesting depth.
PHASE_HIERARCHY = {
    "ExecuteCompiler": {
        "Frontend": {
            "InstantiateFunction": {},
        },
        "Backend": {
            "Optimizer": {},
            "CodeGenPasses": {},
        },
    }
}


def get_phase_breakdown(df: pd.DataFrame) -> PhaseBreakdown:
    """
    Get hierarchical breakdown of compilation phases.

    Returns a PhaseBreakdown object with rich display and analysis methods,
    using the pre-aggregated "Total" events from Clang's -ftime-trace output
    for accurate statistics.

    All durations are in microseconds.

    The hierarchy is defined by the PHASE_HIERARCHY constant and supports
    arbitrary nesting depth. The tree is traversed recursively to build
    the phase breakdown.

    Args:
        df: DataFrame from parse_file()

    Returns:
        PhaseBreakdown object with rich display and analysis methods

    Raises:
        ValueError: If required Total events are missing or if calculated
                   "Other" values are negative (indicating data inconsistency)

    Examples:
        >>> df = parse_file('trace.json')
        >>> breakdown = get_phase_breakdown(df)
        >>>
        >>> # Display in Jupyter (automatic)
        >>> breakdown
        >>>
        >>> # Get total compilation time
        >>> print(f"Total: {breakdown.total_ms:.1f}ms")
        >>>
        >>> # Access specific phases
        >>> breakdown['InstantiateFunction']
        >>> breakdown.frontend
        >>> breakdown.backend.top(5)
        >>>
        >>> # Visualize
        >>> import plotly.express as px
        >>> data = breakdown.to_plotly()
        >>> fig = px.sunburst(**data)
        >>> fig.show()
    """
    if "name" not in df.columns or "dur" not in df.columns:
        raise ValueError("DataFrame missing required 'name' or 'dur' columns")

    # Pre-filter to Total events for efficient lookup
    total_events = df[df["name"].str.startswith("Total ", na=False)].copy()
    total_events["phase"] = total_events["name"].str.removeprefix("Total ")

    def get_duration(phase_name: str) -> Optional[int]:
        """Get duration in microseconds from a Total event."""
        matches = total_events[total_events["phase"] == phase_name]
        if matches.empty:
            return None
        return int(matches.iloc[0]["dur"])

    def process_node(
        node_name: str,
        parent_name: str,
        depth: int,
        children_spec: dict,
    ) -> list[dict]:
        """
        Recursively process a node and its children in the phase hierarchy.

        Args:
            node_name: Name of the current phase node
            parent_name: Name of the parent phase (empty string for root)
            depth: Current depth in the tree (0 for root)
            children_spec: Dictionary of child phases to process

        Returns:
            List of row dictionaries for this node and all descendants

        Raises:
            ValueError: If phase not found or children exceed parent duration
        """
        # Get duration for this node
        node_duration = get_duration(node_name)
        if node_duration is None:
            raise ValueError(f"No Total {node_name} event found in trace")

        # Add current node
        rows = [
            {
                "name": node_name,
                "parent": parent_name,
                "depth": depth,
                "duration": node_duration,
            }
        ]

        if not children_spec:
            return rows

        # Process all children recursively
        children_total = 0
        for child_name, grandchildren_spec in children_spec.items():
            if child_name == "":
                # Empty string means "Other" - skip for now, calculate as residual
                continue

            # Recursively process this child and its descendants
            child_rows = process_node(
                child_name, node_name, depth + 1, grandchildren_spec
            )
            rows.extend(child_rows)

            # Track total duration of direct children only (not grandchildren)
            children_total += child_rows[0]["duration"]

        # Calculate and add "Other" if there's unaccounted time
        other_duration = node_duration - children_total
        if other_duration < 0:
            raise ValueError(
                f"{node_name} children total ({children_total}) "
                f"exceeds parent total ({node_duration})"
            )

        if other_duration > 0:
            rows.append(
                {
                    "name": f"{node_name}_Other",
                    "parent": node_name,
                    "depth": depth + 1,
                    "duration": other_duration,
                }
            )

        return rows

    # Start recursive traversal from root
    root_name = "ExecuteCompiler"
    if root_name not in PHASE_HIERARCHY:
        raise ValueError(f"Root phase '{root_name}' not found in PHASE_HIERARCHY")

    all_rows = process_node(
        root_name,
        "",  # Root has no parent
        0,  # Root is at depth 0
        PHASE_HIERARCHY[root_name],
    )

    breakdown_df = pd.DataFrame(all_rows)
    return PhaseBreakdown(breakdown_df)
