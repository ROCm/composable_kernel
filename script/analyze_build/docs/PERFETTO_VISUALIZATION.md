# Perfetto Visualization Guide

This guide shows how to visualize ninja build timelines in Perfetto UI using the `trace_analysis` library.

## Quick Start

### Command Line Usage

```bash
# Run the example script
python examples/perfetto_visualization_example.py path/to/.ninja_log

# This will:
# 1. Parse the ninja log
# 2. Assign workers for parallelism visualization
# 3. Export to Chrome Trace format
# 4. Save to build_trace.json
```

### Jupyter Notebook Usage

```python
from pathlib import Path
from trace_analysis import NinjaLogParser, ChromeTraceExporter
from trace_analysis.perfetto_display import display_perfetto, print_trace_summary

# Parse ninja log
builds = NinjaLogParser.parse(Path('build/.ninja_log'))
builds_df = NinjaLogParser.to_dataframe(builds)
builds_df = NinjaLogParser.assign_workers(builds_df)

# Export to Chrome Trace format
trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)

# Print summary
print_trace_summary(trace_data)

# Display in Perfetto UI (embedded in notebook)
display_perfetto(trace_data)

# Or save to file for large traces
from trace_analysis.perfetto_display import save_and_link
save_and_link(trace_data, '../data/build_trace.json')
```

## What You Get

The Chrome Trace export provides:

- **Build Timeline**: Visual representation of when each target was built
- **Parallelism Analysis**: See how many workers were active at any time
- **Category Breakdown**: Targets categorized by type (compile, link, archive, etc.)
- **Duration Analysis**: Identify slow compilation units
- **Critical Path**: Understand build dependencies and bottlenecks

## Viewing in Perfetto UI

### Option 1: Embedded in Jupyter (Small Traces)

For traces < 10MB, use `display_perfetto()` to embed directly in the notebook:

```python
display_perfetto(trace_data, height=600)
```

### Option 2: Manual Upload (Large Traces)

For larger traces, save to file and upload manually:

```python
ChromeTraceExporter.export_to_file(trace_data, 'build_trace.json')
```

Then:
1. Go to https://ui.perfetto.dev
2. Click "Open trace file"
3. Select your `build_trace.json`

Or drag and drop the file directly into Perfetto UI.

## DataFrame Schema

The `builds_df` DataFrame has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `target` | str | Build target name (e.g., "obj/foo.o") |
| `start_ms` | int64 | Start time in milliseconds since epoch |
| `end_ms` | int64 | End time in milliseconds since epoch |
| `duration_ms` | int32 | Build duration in milliseconds |
| `cmd_hash` | str | Command hash from ninja |
| `worker_id` | int16 | Assigned worker ID (0-based) |

### Adding Category Column

The Chrome Trace exporter automatically categorizes targets based on file extension:

- `.o`, `.obj` → `compile`
- `.a`, `.lib` → `archive`
- `.so`, `.dll`, `.dylib` → `link_shared`
- `.exe`, `.out` → `link_executable`
- Contains "test" → `test`
- Everything else → `other`

## Chrome Trace Event Format

Each build target is exported as a Chrome Trace event:

```json
{
  "name": "obj/foo.o",
  "cat": "compile",
  "ph": "X",
  "ts": 1234567890000,
  "dur": 5000000,
  "pid": 1,
  "tid": 3,
  "args": {
    "output": "obj/foo.o",
    "duration_ms": 5000,
    "cmd_hash": "abc123"
  }
}
```

## Comparison with ninja_json_converter.py

The `trace_analysis` library provides similar functionality to `ninja_json_converter.py` but with additional features:

### Similarities
- Both parse `.ninja_log` files
- Both export to Chrome Trace Event Format
- Both can be viewed in Perfetto UI

### Differences

| Feature | ninja_json_converter.py | trace_analysis |
|---------|------------------------|----------------|
| **Primary Use** | Quick build visualization | Integrated analysis workflow |
| **Output** | Chrome Trace JSON only | DataFrames + Chrome Trace |
| **Analysis** | External (Perfetto UI) | In-notebook with pandas |
| **Template Data** | No | Yes (with -ftime-trace) |
| **Worker Assignment** | Built-in algorithm | Same algorithm, exposed as DataFrame |
| **Customization** | Command-line flags | Programmatic API |

### When to Use Each

**Use `ninja_json_converter.py` when:**
- You just want a quick visualization
- You're working from the command line
- You don't need further analysis

**Use `trace_analysis` when:**
- You want to analyze build data with pandas
- You're working in Jupyter notebooks
- You want to correlate build times with template analysis
- You need programmatic access to build data

## Examples

### Example 1: Find Slowest Builds

```python
# Get top 10 slowest builds
slowest = builds_df.nlargest(10, 'duration_ms')
print(slowest[['target', 'duration_ms', 'worker_id']])
```

### Example 2: Analyze Worker Utilization

```python
worker_stats = NinjaLogParser.compute_worker_stats(builds_df)
print(worker_stats)
```

### Example 3: Category Breakdown

```python
from trace_analysis.perfetto_display import get_trace_summary

summary = get_trace_summary(trace_data)
print(f"Total events: {summary['event_count']}")
print(f"Total duration: {summary['total_duration_s']:.2f}s")
print(f"Workers: {summary['worker_count']}")
print("\nBy category:")
for cat, count in summary['categories'].items():
    print(f"  {cat}: {count} events")
```

### Example 4: Export with Custom Process ID

```python
# Useful when combining multiple build logs
trace_data = ChromeTraceExporter.export_ninja_timeline(
    builds_df,
    process_id=2,  # Use different PID for each log
    include_metadata=True
)
```

## Troubleshooting

### Issue: Trace file too large for embedded display

**Solution**: Use `save_and_link()` instead of `display_perfetto()`:

```python
save_and_link(trace_data, 'build_trace.json')
```

### Issue: Worker IDs all show as -1

**Solution**: Make sure to call `assign_workers()`:

```python
builds_df = NinjaLogParser.assign_workers(builds_df)
```

### Issue: Import error for perfetto_display

**Solution**: The perfetto display functions are in a separate module:

```python
from trace_analysis.perfetto_display import display_perfetto
```

## See Also

- [CHROME_TRACE_EXPORT.md](CHROME_TRACE_EXPORT.md) - Full design document
- [comprehensive_example.ipynb](../notebooks/comprehensive_example.ipynb) - Complete analysis workflow
- [ninja_json_converter.py](../../ninja_json_converter.py) - Command-line alternative
