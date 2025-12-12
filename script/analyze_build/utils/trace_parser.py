# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Utilities for parsing and analyzing Clang -ftime-trace JSON files.

These utilities are designed to handle large trace files safely by using
streaming and incremental processing to avoid memory issues.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import ijson


def iter_trace_files(trace_dir: Path, pattern: str = "*.json") -> Iterator[Path]:
    """
    Iterate over trace JSON files in a directory.

    Args:
        trace_dir: Directory containing trace files
        pattern: Glob pattern for matching files (default: "*.json")

    Yields:
        Path objects for each matching trace file
    """
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    for file_path in trace_path.rglob(pattern):
        if file_path.is_file():
            yield file_path


def stream_events(trace_file: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream events from a trace file without loading the entire file into memory.

    Args:
        trace_file: Path to the trace JSON file

    Yields:
        Individual event dictionaries
    """
    with open(trace_file, "rb") as f:
        # Stream the traceEvents array
        events = ijson.items(f, "traceEvents.item")
        for event in events:
            yield event


def load_trace_metadata(trace_file: Path) -> Dict[str, Any]:
    """
    Load only the metadata from a trace file (not the events).

    Args:
        trace_file: Path to the trace JSON file

    Returns:
        Dictionary with metadata (e.g., beginningOfTime)
    """
    with open(trace_file, "rb") as f:
        parser = ijson.parse(f)
        metadata = {}
        for prefix, event, value in parser:
            if prefix == "beginningOfTime":
                metadata["beginningOfTime"] = value
            # Stop after getting metadata, don't parse events
            if prefix.startswith("traceEvents"):
                break
        return metadata


def filter_events(
    events: Iterator[Dict[str, Any]],
    event_names: Optional[List[str]] = None,
    min_duration: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Filter events by name and/or minimum duration.

    Args:
        events: Iterator of event dictionaries
        event_names: List of event names to include (None = all)
        min_duration: Minimum duration in microseconds (None = no filter)

    Yields:
        Filtered event dictionaries
    """
    for event in events:
        # Filter by name
        if event_names is not None:
            if event.get("name") not in event_names:
                continue

        # Filter by duration
        if min_duration is not None:
            if event.get("dur", 0) < min_duration:
                continue

        yield event


def get_template_events(events: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """
    Filter for template-related events.

    Args:
        events: Iterator of event dictionaries

    Yields:
        Template instantiation events
    """
    template_event_names = [
        "InstantiateClass",
        "InstantiateFunction",
        "InstantiateVariable",
        "ParseTemplate",
    ]
    return filter_events(events, event_names=template_event_names)


def aggregate_by_name(events: Iterator[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate events by name, computing total duration and count.

    Args:
        events: Iterator of event dictionaries

    Returns:
        Dictionary mapping event names to aggregated statistics
    """
    aggregated = {}

    for event in events:
        name = event.get("name", "Unknown")
        duration = event.get("dur", 0)

        if name not in aggregated:
            aggregated[name] = {
                "count": 0,
                "total_duration": 0,
                "max_duration": 0,
                "min_duration": float("inf"),
            }

        aggregated[name]["count"] += 1
        aggregated[name]["total_duration"] += duration
        aggregated[name]["max_duration"] = max(
            aggregated[name]["max_duration"], duration
        )
        aggregated[name]["min_duration"] = min(
            aggregated[name]["min_duration"], duration
        )

    # Calculate averages
    for name, stats in aggregated.items():
        if stats["count"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
        if stats["min_duration"] == float("inf"):
            stats["min_duration"] = 0

    return aggregated


def get_top_events(
    events: Iterator[Dict[str, Any]], n: int = 10, sort_by: str = "dur"
) -> List[Dict[str, Any]]:
    """
    Get the top N events by a specified field.

    Args:
        events: Iterator of event dictionaries
        n: Number of top events to return
        sort_by: Field to sort by (default: 'dur' for duration)

    Returns:
        List of top N events
    """
    # We need to materialize events for sorting
    # This is safe for top-N queries as we only keep N items
    import heapq

    # Use a min-heap to keep only top N items
    # Include a counter to break ties and avoid comparing dicts
    top_n = []
    counter = 0

    for event in events:
        value = event.get(sort_by, 0)
        if len(top_n) < n:
            heapq.heappush(top_n, (value, counter, event))
            counter += 1
        elif value > top_n[0][0]:
            heapq.heapreplace(top_n, (value, counter, event))
            counter += 1

    # Sort in descending order
    return [event for _, _, event in sorted(top_n, reverse=True)]


def extract_template_detail(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract the template name/detail from an event's args.

    Args:
        event: Event dictionary

    Returns:
        Template detail string or None
    """
    args = event.get("args", {})
    return args.get("detail")


def microseconds_to_seconds(microseconds: int) -> float:
    """Convert microseconds to seconds."""
    return microseconds / 1_000_000


def microseconds_to_milliseconds(microseconds: int) -> float:
    """Convert microseconds to milliseconds."""
    return microseconds / 1_000
