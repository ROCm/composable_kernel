# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Ninja build log parser.

Parses .ninja_log files to extract build timeline and parallelism information.
"""

from pathlib import Path
from typing import List
import pandas as pd

from .models import NinjaBuild


class NinjaLogParser:
    """
    Parser for ninja .ninja_log files.

    Extracts build events with timing information for timeline visualization
    and parallelism analysis.
    """

    @staticmethod
    def parse(log_path: Path) -> List[NinjaBuild]:
        """
        Parse a .ninja_log file.

        Args:
            log_path: Path to .ninja_log file

        Returns:
            List of NinjaBuild objects

        Note:
            The .ninja_log format is:
            # ninja log v5
            <start_ms>  <end_ms>  <mtime>  <target>  <cmdhash>

            Times are in milliseconds since epoch.
        """
        builds = []

        with open(log_path, "r") as f:
            # Skip header line
            f.readline()

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 5:
                    try:
                        builds.append(
                            NinjaBuild(
                                target=parts[3],
                                start_ms=int(parts[0]),
                                end_ms=int(parts[1]),
                                cmd_hash=parts[4],
                            )
                        )
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue

        return builds

    @staticmethod
    def to_dataframe(builds: List[NinjaBuild]) -> pd.DataFrame:
        """
        Convert NinjaBuild objects to a pandas DataFrame.

        Args:
            builds: List of NinjaBuild objects

        Returns:
            DataFrame with optimized dtypes
        """
        if not builds:
            return pd.DataFrame(
                columns=[
                    "target",
                    "start_ms",
                    "end_ms",
                    "duration_ms",
                    "cmd_hash",
                    "worker_id",
                ]
            )

        df = pd.DataFrame(
            {
                "target": [b.target for b in builds],
                "start_ms": [b.start_ms for b in builds],
                "end_ms": [b.end_ms for b in builds],
                "cmd_hash": [b.cmd_hash for b in builds],
                "worker_id": [b.worker_id for b in builds],
            }
        )

        # Compute duration
        df["duration_ms"] = df["end_ms"] - df["start_ms"]

        # Optimize dtypes
        df["start_ms"] = df["start_ms"].astype("int64")
        df["end_ms"] = df["end_ms"].astype("int64")
        df["duration_ms"] = df["duration_ms"].astype("int32")
        df["target"] = df["target"].astype("category")
        df["worker_id"] = df["worker_id"].astype(
            "int16"
        )  # int16 supports up to 32K workers

        return df

    @staticmethod
    def assign_workers(builds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign worker IDs based on temporal overlap.

        Builds that overlap in time must be on different workers.
        Uses a greedy algorithm to assign the minimum number of workers.

        Args:
            builds_df: DataFrame of builds (must have start_ms, end_ms columns)

        Returns:
            DataFrame with worker_id column populated
        """
        if len(builds_df) == 0:
            return builds_df

        # Sort by start time
        builds_df = builds_df.sort_values("start_ms").reset_index(drop=True)

        # Track active builds per worker
        worker_end_times = []
        worker_ids = []

        for idx, row in builds_df.iterrows():
            start = row["start_ms"]

            # Find first available worker (one whose last build finished before this starts)
            worker_id = -1
            for i, end_time in enumerate(worker_end_times):
                if end_time <= start:
                    worker_id = i
                    worker_end_times[i] = row["end_ms"]
                    break

            # If no worker available, create a new one
            if worker_id == -1:
                worker_id = len(worker_end_times)
                worker_end_times.append(row["end_ms"])

            worker_ids.append(worker_id)

        # Assign all worker IDs at once
        builds_df["worker_id"] = worker_ids
        builds_df["worker_id"] = builds_df["worker_id"].astype("int16")

        return builds_df

    @staticmethod
    def compute_worker_stats(builds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics per worker.

        Args:
            builds_df: DataFrame with worker_id assigned

        Returns:
            DataFrame with worker statistics
        """
        if len(builds_df) == 0 or "worker_id" not in builds_df.columns:
            return pd.DataFrame(
                columns=[
                    "worker_id",
                    "build_count",
                    "total_time_s",
                    "idle_time_s",
                    "utilization",
                ]
            )

        worker_stats = (
            builds_df.groupby("worker_id")
            .agg(
                {
                    "start_ms": "min",
                    "end_ms": "max",
                    "duration_ms": ["sum", "count"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        worker_stats.columns = [
            "worker_id",
            "first_start_ms",
            "last_end_ms",
            "total_duration_ms",
            "build_count",
        ]

        # Compute wall-clock time and idle time
        worker_stats["wall_clock_ms"] = (
            worker_stats["last_end_ms"] - worker_stats["first_start_ms"]
        )
        worker_stats["idle_time_ms"] = (
            worker_stats["wall_clock_ms"] - worker_stats["total_duration_ms"]
        )

        # Convert to seconds
        worker_stats["total_time_s"] = worker_stats["total_duration_ms"] / 1000.0
        worker_stats["idle_time_s"] = worker_stats["idle_time_ms"] / 1000.0
        worker_stats["wall_clock_s"] = worker_stats["wall_clock_ms"] / 1000.0

        # Compute utilization
        worker_stats["utilization"] = (
            worker_stats["total_duration_ms"] / worker_stats["wall_clock_ms"]
        )

        return worker_stats[
            [
                "worker_id",
                "build_count",
                "total_time_s",
                "idle_time_s",
                "wall_clock_s",
                "utilization",
            ]
        ]

    @staticmethod
    def link_to_trace_files(
        builds_df: pd.DataFrame, trace_files: List[Path]
    ) -> pd.DataFrame:
        """
        Link ninja builds to trace files.

        Matches based on target name (e.g., obj/foo.o -> foo.json).

        Args:
            builds_df: DataFrame of ninja builds
            trace_files: List of trace file paths

        Returns:
            DataFrame with trace_file column added
        """
        # Create mapping from target to trace file
        trace_map = {}
        for trace_path in trace_files:
            # Extract base name without .json extension
            base_name = trace_path.stem

            # Try to match to target
            # Common patterns: obj/foo.o -> foo.json, lib/bar.a -> bar.json
            for target in builds_df["target"].unique():
                target_base = Path(target).stem
                if target_base == base_name:
                    trace_map[target] = str(trace_path)

        # Add trace_file column
        builds_df["trace_file"] = builds_df["target"].map(trace_map)

        return builds_df
