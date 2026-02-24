#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "jinja2>=3.0.0",
# ]
# ///
"""
Build Time Analysis Tool for Composable Kernel

Analyzes Clang -ftime-trace output to identify template instantiation
bottlenecks and generate comprehensive build time reports.
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("Error: jinja2 is required but not installed.", file=sys.stderr)
    print("Install with: apt-get install python3-jinja2", file=sys.stderr)
    print("Or with pip: pip install jinja2", file=sys.stderr)
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    if len(sys.argv) < 7:
        print(
            "Usage: analyze_build_trace.py <trace_files_or_dir> <output_file> <target> <granularity> <build_time> <template_dir>"
        )
        print(
            "  trace_files_or_dir: Comma-separated list of trace files OR directory containing .json files"
        )
        sys.exit(1)

    return {
        "trace_input": sys.argv[1],
        "output_file": sys.argv[2],
        "target": sys.argv[3],
        "granularity": sys.argv[4],
        "build_time": sys.argv[5],
        "template_dir": sys.argv[6],
    }


def find_trace_files(trace_input):
    """Find all trace files from input (file list, single file, or directory)."""
    trace_files = []

    # Check if it's a directory
    if os.path.isdir(trace_input):
        print(f"Scanning directory: {trace_input}")
        for root, dirs, files in os.walk(trace_input):
            for file in files:
                # Include .cpp.json and .hip.json, exclude compile_commands.json and CMake files
                if file.endswith((".cpp.json", ".hip.json")) and "CMakeFiles" in root:
                    trace_files.append(os.path.join(root, file))
        trace_files.sort()
    # Check if it's a comma-separated list
    elif "," in trace_input:
        trace_files = [f.strip() for f in trace_input.split(",")]
    # Single file
    else:
        trace_files = [trace_input]

    # Filter out non-existent files
    valid_files = [f for f in trace_files if os.path.isfile(f)]

    if not valid_files:
        print(f"Error: No valid trace files found in: {trace_input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(valid_files)} trace file(s)")
    return valid_files


def load_trace_data(trace_files):
    """Load and parse multiple trace JSON files."""
    all_data = []

    for trace_file in trace_files:
        print(f"  Loading: {trace_file}")
        try:
            with open(trace_file, "r") as f:
                data = json.load(f)
                # Get file basename for tracking
                file_name = os.path.basename(trace_file)
                all_data.append({"file": file_name, "path": trace_file, "data": data})
        except Exception as e:
            print(f"  Warning: Failed to load {trace_file}: {e}", file=sys.stderr)

    return all_data


def process_events(all_trace_data):
    """Process trace events from multiple files and extract statistics."""
    print("Processing events from all files...")

    template_stats = defaultdict(lambda: {"count": 0, "total_dur": 0})
    phase_stats = defaultdict(int)
    top_individual = []
    file_stats = []
    total_events = 0

    for trace_info in all_trace_data:
        file_name = trace_info["file"]
        data = trace_info["data"]
        events = data.get("traceEvents", [])

        file_template_time = 0
        file_event_count = len(events)
        total_events += file_event_count

        print(f"  Processing {file_name}: {file_event_count:,} events")

        for event in events:
            name = event.get("name", "")
            dur = int(event.get("dur", 0))  # Keep as integer microseconds

            if name and dur > 0:
                phase_stats[name] += dur

            if name in ["InstantiateFunction", "InstantiateClass"]:
                detail = event.get("args", {}).get("detail", "")
                top_individual.append(
                    {"detail": detail, "dur": dur, "type": name, "file": file_name}
                )

                file_template_time += dur

                # Extract template name (everything before '<' or '(')
                match = re.match(r"^([^<(]+)", detail)
                if match:
                    template_name = match.group(1).strip()
                    # Normalize template names
                    template_name = re.sub(r"^ck::", "", template_name)
                    template_name = re.sub(r"^std::", "std::", template_name)

                    template_stats[template_name]["count"] += 1
                    template_stats[template_name]["total_dur"] += dur

        file_stats.append(
            {
                "name": file_name,
                "events": file_event_count,
                "template_time": file_template_time,
            }
        )

    return template_stats, phase_stats, top_individual, file_stats, total_events


def prepare_template_data(template_stats, phase_stats, top_individual, file_stats):
    """Prepare and calculate derived statistics for template rendering."""
    print("Sorting data...")

    # Sort data
    sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1], reverse=True)
    top_individual.sort(key=lambda x: x["dur"], reverse=True)
    file_stats.sort(key=lambda x: x["template_time"], reverse=True)

    # Calculate totals
    total_template_time = sum(s["total_dur"] for s in template_stats.values())
    total_trace_time = sum(phase_stats.values())
    total_inst = sum(s["count"] for s in template_stats.values())

    # Prepare templates by time with calculated fields
    templates_by_time = []
    for name, stats in sorted(
        template_stats.items(), key=lambda x: x[1]["total_dur"], reverse=True
    ):
        templates_by_time.append(
            (
                name,
                {
                    "count": stats["count"],
                    "total_dur": stats["total_dur"],
                    "avg": stats["total_dur"] // stats["count"]
                    if stats["count"] > 0
                    else 0,
                    "pct": 100 * stats["total_dur"] / total_template_time
                    if total_template_time > 0
                    else 0,
                },
            )
        )

    # Prepare templates by count
    templates_by_count = []
    for name, stats in sorted(
        template_stats.items(), key=lambda x: x[1]["count"], reverse=True
    ):
        templates_by_count.append(
            (
                name,
                {
                    "count": stats["count"],
                    "total_dur": stats["total_dur"],
                    "avg": stats["total_dur"] // stats["count"]
                    if stats["count"] > 0
                    else 0,
                },
            )
        )

    # Add friendly type names to individual instantiations
    for inst in top_individual:
        inst["inst_type"] = "Func" if inst["type"] == "InstantiateFunction" else "Class"

    # Calculate additional metrics
    median_count = 0
    if len(template_stats) > 0:
        median_count = sorted([s["count"] for s in template_stats.values()])[
            len(template_stats) // 2
        ]

    top10_pct = 0
    if len(templates_by_time) >= 10:
        top10_pct = (
            100
            * sum(s[1]["total_dur"] for s in templates_by_time[:10])
            / total_template_time
        )

    return {
        "sorted_phases": sorted_phases,
        "top_individual": top_individual,
        "templates_by_time": templates_by_time,
        "templates_by_count": templates_by_count,
        "total_template_time": total_template_time,
        "total_trace_time": total_trace_time,
        "total_inst": total_inst,
        "median_count": median_count,
        "top10_pct": top10_pct,
        "unique_families": len(template_stats),
        "file_stats": file_stats,
    }


def setup_jinja_environment(template_dir):
    """Set up Jinja2 environment with custom filters."""
    env = Environment(loader=FileSystemLoader(template_dir))

    def format_number(value):
        """Format number with thousand separators."""
        return f"{value:,}"

    def truncate(value, length):
        """Truncate string to length with ellipsis."""
        if len(value) > length:
            return value[: length - 3] + "..."
        return value

    def pad(value, length):
        """Pad string to specified length."""
        return f"{value:<{length}}"

    def us_to_ms(value):
        """Convert microseconds to milliseconds."""
        return value / 1000.0

    def us_to_s(value):
        """Convert microseconds to seconds."""
        return value / 1000000.0

    env.filters["format_number"] = format_number
    env.filters["truncate"] = truncate
    env.filters["pad"] = pad
    env.filters["us_to_ms"] = us_to_ms
    env.filters["us_to_s"] = us_to_s

    return env


def generate_report(env, data, args, total_events, num_files):
    """Generate the final report using Jinja2 template."""
    print("Rendering report with Jinja2...")

    template = env.get_template("build_analysis_report.md.jinja")

    report_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        target=args["target"],
        granularity=args["granularity"],
        build_time=args["build_time"],
        total_events=total_events,
        num_files=num_files,
        total_instantiations=data["total_inst"],
        unique_families=data["unique_families"],
        total_trace_time=data["total_trace_time"],
        total_template_time=data["total_template_time"],
        phases=data["sorted_phases"],
        top_individual=data["top_individual"],
        templates_by_time=data["templates_by_time"],
        templates_by_count=data["templates_by_count"],
        median_count=data["median_count"],
        top10_pct=data["top10_pct"],
        file_stats=data["file_stats"],
    )

    return report_content


def main():
    """Main entry point for the analysis tool."""
    args = parse_arguments()

    # Find and load trace files
    trace_files = find_trace_files(args["trace_input"])
    all_trace_data = load_trace_data(trace_files)

    # Process events from all files
    template_stats, phase_stats, top_individual, file_stats, total_events = (
        process_events(all_trace_data)
    )

    # Prepare template data
    data = prepare_template_data(
        template_stats, phase_stats, top_individual, file_stats
    )

    # Setup Jinja2 environment
    env = setup_jinja_environment(args["template_dir"])

    # Generate report
    report_content = generate_report(env, data, args, total_events, len(all_trace_data))

    # Write output
    with open(args["output_file"], "w") as f:
        f.write(report_content)

    print(f"Report generated: {args['output_file']}")
    print(f"Report size: {len(report_content):,} bytes")
    print(f"Analyzed {len(all_trace_data)} file(s) with {total_events:,} total events")


if __name__ == "__main__":
    main()
