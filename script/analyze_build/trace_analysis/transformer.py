# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data transformer for converting parsed events to DataFrames.

Transforms raw event dictionaries into structured pandas DataFrames
optimized for analysis.
"""

from typing import List, Dict, Any
import pandas as pd

from .parser import TraceParser


class TraceTransformer:
    """
    Transformer for converting trace events to pandas DataFrames.

    Provides efficient conversion from raw event dictionaries to
    structured DataFrames optimized for analytical queries.
    """

    @staticmethod
    def to_events_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert raw events to a DataFrame.

        Args:
            events: List of event dictionaries

        Returns:
            DataFrame with columns: name, dur, ts, pid, tid, ph, args
        """
        if not events:
            return pd.DataFrame(columns=["name", "dur", "ts", "pid", "tid", "ph"])

        # Extract key fields for efficient storage
        df = pd.DataFrame(
            {
                "name": [e.get("name", "Unknown") for e in events],
                "dur": [e.get("dur", 0) for e in events],
                "ts": [e.get("ts", 0) for e in events],
                "pid": [e.get("pid", 0) for e in events],
                "tid": [e.get("tid", 0) for e in events],
                "ph": [e.get("ph", "") for e in events],
            }
        )

        # Optimize dtypes for storage
        df["dur"] = df["dur"].astype("int64")
        df["ts"] = df["ts"].astype("int64")
        df["pid"] = df["pid"].astype("int32")
        df["tid"] = df["tid"].astype("int32")
        df["ph"] = df["ph"].astype("category")
        df["name"] = df["name"].astype("category")

        return df

    @staticmethod
    def to_templates_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert template events to a DataFrame.

        Args:
            events: List of event dictionaries

        Returns:
            DataFrame with template-specific information
        """
        # Filter for template events
        template_events = [e for e in events if TraceParser.is_template_event(e)]

        if not template_events:
            return pd.DataFrame(columns=["name", "dur", "template_detail"])

        df = pd.DataFrame(
            {
                "name": [e.get("name", "Unknown") for e in template_events],
                "dur": [e.get("dur", 0) for e in template_events],
                "template_detail": [
                    TraceParser.extract_template_detail(e) for e in template_events
                ],
            }
        )

        # Optimize dtypes
        df["dur"] = df["dur"].astype("int64")
        df["name"] = df["name"].astype("category")

        return df

    @staticmethod
    def compute_file_stats(
        events_df: pd.DataFrame, templates_df: pd.DataFrame, file_name: str
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for a file.

        Args:
            events_df: DataFrame of all events
            templates_df: DataFrame of template events
            file_name: Name of the file

        Returns:
            Dictionary of file statistics
        """
        return {
            "file_name": file_name,
            "total_events": len(events_df),
            "total_duration_us": int(events_df["dur"].sum())
            if len(events_df) > 0
            else 0,
            "template_event_count": len(templates_df),
            "template_duration_us": int(templates_df["dur"].sum())
            if len(templates_df) > 0
            else 0,
            "max_event_duration_us": int(events_df["dur"].max())
            if len(events_df) > 0
            else 0,
            "unique_event_types": events_df["name"].nunique()
            if len(events_df) > 0
            else 0,
        }

    @staticmethod
    def aggregate_event_types(events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate events by type.

        Args:
            events_df: DataFrame of events

        Returns:
            DataFrame with aggregated statistics per event type
        """
        if len(events_df) == 0:
            return pd.DataFrame(
                columns=[
                    "event_type",
                    "count",
                    "total_duration",
                    "avg_duration",
                    "max_duration",
                ]
            )

        agg_df = (
            events_df.groupby("name", observed=True)
            .agg({"dur": ["count", "sum", "mean", "max"]})
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = [
            "event_type",
            "count",
            "total_duration",
            "avg_duration",
            "max_duration",
        ]

        # Sort by total duration
        agg_df = agg_df.sort_values("total_duration", ascending=False)

        return agg_df

    @staticmethod
    def aggregate_templates(templates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate template instantiations.

        Args:
            templates_df: DataFrame of template events

        Returns:
            DataFrame with aggregated template statistics
        """
        if len(templates_df) == 0:
            return pd.DataFrame(
                columns=["template_detail", "count", "total_duration", "avg_duration"]
            )

        agg_df = (
            templates_df.groupby("template_detail")
            .agg({"dur": ["count", "sum", "mean"]})
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = ["template_detail", "count", "total_duration", "avg_duration"]

        # Sort by count
        agg_df = agg_df.sort_values("count", ascending=False)

        return agg_df
