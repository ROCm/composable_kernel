# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Perfetto UI display utilities for Jupyter notebooks.

Provides functions to display Chrome Trace data in Perfetto UI
directly within Jupyter notebooks.
"""

import json
import base64
from typing import Dict, Any, Optional


def display_perfetto(trace_data: Dict[str, Any], height: int = 600):
    """
    Display Perfetto UI in Jupyter notebook with embedded trace data.

    Args:
        trace_data: Chrome Trace Event Format dictionary
        height: Height of the IFrame in pixels (default: 600)

    Returns:
        IPython IFrame object for display in notebook

    Example:
        >>> from trace_analysis import NinjaLogParser, ChromeTraceExporter
        >>> from trace_analysis.perfetto_display import display_perfetto
        >>> builds_df = NinjaLogParser.to_dataframe(builds)
        >>> trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)
        >>> display_perfetto(trace_data)

    Note:
        This function requires IPython to be installed (available in Jupyter).
        The trace data is base64-encoded and embedded in the Perfetto UI URL.
        For very large traces (>10MB), consider using save_and_link() instead.
    """
    try:
        from IPython.display import IFrame
    except ImportError:
        raise ImportError(
            "IPython is required for display_perfetto(). "
            "Install it with: pip install ipython"
        )

    # Convert trace to JSON string
    trace_json = json.dumps(trace_data)

    # Base64 encode for URL
    trace_b64 = base64.b64encode(trace_json.encode()).decode()

    # Perfetto UI URL with embedded trace
    perfetto_url = f"https://ui.perfetto.dev/#!/?s={trace_b64}"

    # Display in IFrame
    return IFrame(perfetto_url, width="100%", height=height)


def save_and_link(
    trace_data: Dict[str, Any], output_path: str, link_text: Optional[str] = None
):
    """
    Save trace to file and display a link to open in Perfetto UI.

    This is useful for large traces that are too big to embed in a URL.

    Args:
        trace_data: Chrome Trace Event Format dictionary
        output_path: Path to save the trace file
        link_text: Custom link text (default: "Open trace in Perfetto UI")

    Returns:
        IPython HTML object with download link and instructions

    Example:
        >>> save_and_link(trace_data, '../data/build_trace.json')

    Note:
        The user will need to manually upload the saved file to
        https://ui.perfetto.dev
    """
    try:
        from IPython.display import HTML
    except ImportError:
        raise ImportError(
            "IPython is required for save_and_link(). "
            "Install it with: pip install ipython"
        )

    # Save trace to file
    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    if link_text is None:
        link_text = "Open trace in Perfetto UI"

    # Create HTML with instructions
    html = f"""
    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h4>Trace saved to: <code>{output_path}</code></h4>
        <p>To view in Perfetto UI:</p>
        <ol>
            <li>Go to <a href="https://ui.perfetto.dev" target="_blank">{link_text}</a></li>
            <li>Click "Open trace file" and select: <code>{output_path}</code></li>
        </ol>
        <p><em>Or drag and drop the file directly into the Perfetto UI.</em></p>
    </div>
    """

    return HTML(html)


def get_trace_summary(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary statistics from trace data.

    Args:
        trace_data: Chrome Trace Event Format dictionary

    Returns:
        Dictionary with summary statistics

    Example:
        >>> summary = get_trace_summary(trace_data)
        >>> print(f"Total events: {summary['event_count']}")
        >>> print(f"Total time: {summary['total_duration_s']:.2f}s")
    """
    events = trace_data.get("traceEvents", [])

    if not events:
        return {
            "event_count": 0,
            "total_duration_s": 0.0,
            "categories": {},
            "worker_count": 0,
        }

    # Count events by category
    categories = {}
    total_duration_us = 0
    worker_ids = set()

    for event in events:
        cat = event.get("cat", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

        dur = event.get("dur", 0)
        total_duration_us += dur

        tid = event.get("tid")
        if tid is not None:
            worker_ids.add(tid)

    return {
        "event_count": len(events),
        "total_duration_s": total_duration_us / 1e6,
        "categories": categories,
        "worker_count": len(worker_ids),
    }


def print_trace_summary(trace_data: Dict[str, Any]) -> None:
    """
    Print a formatted summary of trace data.

    Args:
        trace_data: Chrome Trace Event Format dictionary

    Example:
        >>> print_trace_summary(trace_data)
        === Trace Summary ===
        Total events: 1,234
        Total duration: 123.45s
        Workers: 8
        ...
    """
    summary = get_trace_summary(trace_data)

    print("=== Trace Summary ===")
    print(f"Total events: {summary['event_count']:,}")
    print(f"Total duration: {summary['total_duration_s']:.2f}s")
    print(f"Workers: {summary['worker_count']}")

    if summary["categories"]:
        print("\nEvents by category:")
        for cat, count in sorted(
            summary["categories"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {cat:15} {count:6,} events")
