#!/usr/bin/env python3
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
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
        print("Usage: analyze_build_trace.py <trace_file> <output_file> <target> <granularity> <build_time> <template_dir>")
        sys.exit(1)

    return {
        'trace_file': sys.argv[1],
        'output_file': sys.argv[2],
        'target': sys.argv[3],
        'granularity': sys.argv[4],
        'build_time': sys.argv[5],
        'template_dir': sys.argv[6],
    }


def load_trace_data(trace_file):
    """Load and parse the trace JSON file."""
    print(f'Loading trace file: {trace_file}')
    with open(trace_file, 'r') as f:
        return json.load(f)


def process_events(data):
    """Process trace events and extract template instantiation statistics."""
    print('Processing events...')

    template_stats = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})
    phase_stats = defaultdict(float)
    top_individual = []

    for event in data.get('traceEvents', []):
        name = event.get('name', '')
        dur = event.get('dur', 0) / 1000.0  # Convert to milliseconds

        if name and dur > 0:
            phase_stats[name] += dur

        if name in ['InstantiateFunction', 'InstantiateClass']:
            detail = event.get('args', {}).get('detail', '')
            top_individual.append({
                'detail': detail,
                'dur': dur,
                'type': name
            })

            # Extract template name (everything before '<' or '(')
            match = re.match(r'^([^<(]+)', detail)
            if match:
                template_name = match.group(1).strip()
                # Normalize template names
                template_name = re.sub(r'^ck::', '', template_name)
                template_name = re.sub(r'^std::', 'std::', template_name)

                template_stats[template_name]['count'] += 1
                template_stats[template_name]['total_dur'] += dur

    return template_stats, phase_stats, top_individual


def prepare_template_data(template_stats, phase_stats, top_individual):
    """Prepare and calculate derived statistics for template rendering."""
    print('Sorting data...')

    # Sort data
    sorted_phases = sorted(phase_stats.items(), key=lambda x: x[1], reverse=True)
    top_individual.sort(key=lambda x: x['dur'], reverse=True)

    # Calculate totals
    total_template_time = sum(s['total_dur'] for s in template_stats.values())
    total_trace_time = sum(phase_stats.values())
    total_inst = sum(s['count'] for s in template_stats.values())

    # Prepare templates by time with calculated fields
    templates_by_time = []
    for name, stats in sorted(template_stats.items(), key=lambda x: x[1]['total_dur'], reverse=True):
        templates_by_time.append((name, {
            'count': stats['count'],
            'total_dur': stats['total_dur'],
            'avg': stats['total_dur'] / stats['count'] if stats['count'] > 0 else 0,
            'pct': 100 * stats['total_dur'] / total_template_time if total_template_time > 0 else 0
        }))

    # Prepare templates by count
    templates_by_count = []
    for name, stats in sorted(template_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        templates_by_count.append((name, {
            'count': stats['count'],
            'total_dur': stats['total_dur'],
            'avg': stats['total_dur'] / stats['count'] if stats['count'] > 0 else 0
        }))

    # Add friendly type names to individual instantiations
    for inst in top_individual:
        inst['inst_type'] = 'Func' if inst['type'] == 'InstantiateFunction' else 'Class'

    # Calculate additional metrics
    median_count = 0
    if len(template_stats) > 0:
        median_count = sorted([s["count"] for s in template_stats.values()])[len(template_stats) // 2]

    top10_pct = 0
    if len(templates_by_time) >= 10:
        top10_pct = 100 * sum(s[1]["total_dur"] for s in templates_by_time[:10]) / total_template_time

    return {
        'sorted_phases': sorted_phases,
        'top_individual': top_individual,
        'templates_by_time': templates_by_time,
        'templates_by_count': templates_by_count,
        'total_template_time': total_template_time,
        'total_trace_time': total_trace_time,
        'total_inst': total_inst,
        'median_count': median_count,
        'top10_pct': top10_pct,
        'unique_families': len(template_stats),
    }


def setup_jinja_environment(template_dir):
    """Set up Jinja2 environment with custom filters."""
    env = Environment(loader=FileSystemLoader(template_dir))

    def format_number(value):
        """Format number with thousand separators."""
        return f'{value:,}'

    def truncate(value, length):
        """Truncate string to length with ellipsis."""
        if len(value) > length:
            return value[:length - 3] + '...'
        return value

    def pad(value, length):
        """Pad string to specified length."""
        return f'{value:<{length}}'

    env.filters['format_number'] = format_number
    env.filters['truncate'] = truncate
    env.filters['pad'] = pad

    return env


def generate_report(env, data, args, total_events):
    """Generate the final report using Jinja2 template."""
    print('Rendering report with Jinja2...')

    template = env.get_template('build_analysis_report.md.jinja')

    report_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        target=args['target'],
        granularity=args['granularity'],
        build_time=args['build_time'],
        trace_time_sec=f'{data["total_trace_time"] / 1000:.1f}',
        template_time_sec=f'{data["total_template_time"] / 1000:.1f}',
        template_pct=f'{100 * data["total_template_time"] / data["total_trace_time"]:.1f}',
        total_events=total_events,
        total_instantiations=data['total_inst'],
        unique_families=data['unique_families'],
        total_trace_time=data['total_trace_time'],
        total_template_time=data['total_template_time'],
        phases=data['sorted_phases'],
        top_individual=data['top_individual'],
        templates_by_time=data['templates_by_time'],
        templates_by_count=data['templates_by_count'],
        median_count=data['median_count'],
        top10_pct=f'{data["top10_pct"]:.1f}'
    )

    return report_content


def main():
    """Main entry point for the analysis tool."""
    args = parse_arguments()

    # Load trace data
    trace_data = load_trace_data(args['trace_file'])
    total_events = len(trace_data.get('traceEvents', []))

    # Process events
    template_stats, phase_stats, top_individual = process_events(trace_data)

    # Prepare template data
    data = prepare_template_data(template_stats, phase_stats, top_individual)

    # Setup Jinja2 environment
    env = setup_jinja_environment(args['template_dir'])

    # Generate report
    report_content = generate_report(env, data, args, total_events)

    # Write output
    with open(args['output_file'], 'w') as f:
        f.write(report_content)

    print(f'Report generated: {args["output_file"]}')
    print(f'Report size: {len(report_content)} bytes')


if __name__ == '__main__':
    main()
