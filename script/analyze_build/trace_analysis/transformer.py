# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data transformer for converting parsed events to DataFrames.

Transforms raw event dictionaries into structured pandas DataFrames
optimized for analysis, including multi-table schemas for template
relationships and timeline visualization.
"""

from typing import List, Dict, Any
import pandas as pd

from .parser import TraceParser
from .template_parser import TemplateParser


class TraceTransformer:
    """
    Transformer for converting trace events to pandas DataFrames.

    Provides efficient conversion from raw event dictionaries to
    structured DataFrames optimized for analytical queries.

    Supports both simple flat schemas and advanced multi-table schemas
    with template relationships and timeline data.
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
    def to_enhanced_schema(
        events: List[Dict[str, Any]], file_id: int = 0, beginning_of_time_us: int = 0
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert events to enhanced multi-table schema.

        Creates separate tables for templates, instantiations, and arguments
        with proper relationships for advanced analysis.

        Args:
            events: List of event dictionaries
            file_id: Unique file identifier
            beginning_of_time_us: Wall-clock start time (from trace JSON)

        Returns:
            Dictionary with keys:
            - 'templates': Unique templates with structure
            - 'instantiations': Template instantiation events
            - 'template_args': Template argument relationships
            - 'events': All compiler events
        """
        # Get template events
        template_events = [e for e in events if TraceParser.is_template_event(e)]

        # Build unique templates table
        templates_data = []
        template_map = {}  # signature -> template_id
        next_template_id = 0

        for event in template_events:
            signature = TraceParser.extract_template_detail(event)
            if not signature or signature in template_map:
                continue

            # Parse template structure
            base_name, args, depth = TemplateParser.parse_signature(signature)

            template_id = next_template_id
            template_map[signature] = template_id
            next_template_id += 1

            templates_data.append(
                {
                    "template_id": template_id,
                    "template_name": base_name,
                    "full_signature": signature,
                    "depth": depth,
                    "arg_count": len(args),
                }
            )

        templates_df = (
            pd.DataFrame(templates_data)
            if templates_data
            else pd.DataFrame(
                columns=[
                    "template_id",
                    "template_name",
                    "full_signature",
                    "depth",
                    "arg_count",
                ]
            )
        )

        # Build instantiations table
        instantiations_data = []
        instantiation_id = 0

        for event in template_events:
            signature = TraceParser.extract_template_detail(event)
            if signature in template_map:
                instantiations_data.append(
                    {
                        "instantiation_id": instantiation_id,
                        "template_id": template_map[signature],
                        "file_id": file_id,
                        "dur_us": event.get("dur", 0),
                        "ts_us": event.get("ts", 0),
                        "event_type": event.get("name", "Unknown"),
                    }
                )
                instantiation_id += 1

        instantiations_df = (
            pd.DataFrame(instantiations_data)
            if instantiations_data
            else pd.DataFrame(
                columns=[
                    "instantiation_id",
                    "template_id",
                    "file_id",
                    "dur_us",
                    "ts_us",
                    "event_type",
                ]
            )
        )

        # Build template arguments table
        args_data = []

        for template_id, signature in [(v, k) for k, v in template_map.items()]:
            _, args, _ = TemplateParser.parse_signature(signature)

            for pos, arg in enumerate(args):
                arg_type = TemplateParser.classify_argument(arg)
                arg_template_id = (
                    template_map.get(arg) if arg_type == "template" else None
                )

                args_data.append(
                    {
                        "parent_template_id": template_id,
                        "arg_position": pos,
                        "arg_template_id": arg_template_id,
                        "arg_type": arg_type,
                        "arg_text": arg,
                    }
                )

        template_args_df = (
            pd.DataFrame(args_data)
            if args_data
            else pd.DataFrame(
                columns=[
                    "parent_template_id",
                    "arg_position",
                    "arg_template_id",
                    "arg_type",
                    "arg_text",
                ]
            )
        )

        # Build events table with absolute timestamps
        events_df = TraceTransformer.to_events_dataframe(events)
        if len(events_df) > 0 and beginning_of_time_us > 0:
            events_df["ts_absolute_us"] = beginning_of_time_us + events_df["ts"]

        # Optimize dtypes
        if len(templates_df) > 0:
            templates_df["template_id"] = templates_df["template_id"].astype("int32")
            templates_df["template_name"] = templates_df["template_name"].astype(
                "category"
            )
            templates_df["depth"] = templates_df["depth"].astype("int8")
            templates_df["arg_count"] = templates_df["arg_count"].astype("int8")

        if len(instantiations_df) > 0:
            instantiations_df["instantiation_id"] = instantiations_df[
                "instantiation_id"
            ].astype("int64")
            instantiations_df["template_id"] = instantiations_df["template_id"].astype(
                "int32"
            )
            instantiations_df["file_id"] = instantiations_df["file_id"].astype("int32")
            instantiations_df["dur_us"] = instantiations_df["dur_us"].astype("int64")
            instantiations_df["ts_us"] = instantiations_df["ts_us"].astype("int64")
            instantiations_df["event_type"] = instantiations_df["event_type"].astype(
                "category"
            )

        if len(template_args_df) > 0:
            template_args_df["parent_template_id"] = template_args_df[
                "parent_template_id"
            ].astype("int32")
            template_args_df["arg_position"] = template_args_df["arg_position"].astype(
                "int8"
            )
            template_args_df["arg_template_id"] = template_args_df[
                "arg_template_id"
            ].astype("Int32")  # Nullable
            template_args_df["arg_type"] = template_args_df["arg_type"].astype(
                "category"
            )

        return {
            "templates": templates_df,
            "instantiations": instantiations_df,
            "template_args": template_args_df,
            "events": events_df,
        }

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

    @staticmethod
    def extract_beginning_of_time(trace_data: Dict[str, Any]) -> int:
        """
        Extract beginningOfTime from trace JSON data.

        Args:
            trace_data: Parsed JSON data from trace file

        Returns:
            Beginning of time in microseconds (0 if not found)
        """
        if isinstance(trace_data, dict):
            return trace_data.get("beginningOfTime", 0)
        return 0
