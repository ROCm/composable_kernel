# Build Trace Analysis

Tools and notebooks for analyzing Clang `-ftime-trace` build performance data from the Composable Kernel project.

## Overview

This directory contains utilities and Jupyter notebooks for analyzing the 4,484+ JSON trace files (~46 GB) generated during compilation with `-ftime-trace`. The analysis focuses on:

- Template instantiation costs
- Compilation bottlenecks
- Metaprogramming performance
- Build time optimization opportunities

## Directory Structure

```
script/build_analysis/
├── notebooks/           # Jupyter notebooks for analysis
│   └── 01_initial_exploration.ipynb
├── utils/              # Reusable Python utilities
│   ├── __init__.py
│   └── trace_parser.py
├── data/               # Cached/generated data (gitignored)
└── README.md
```

## Prerequisites

Install required Python packages:

```bash
pip install ijson pandas matplotlib seaborn jupyter
```

Or if using the project's virtual environment:

```bash
source .venv/bin/activate  # Activate the venv
pip install ijson pandas matplotlib seaborn jupyter
```

## Quick Start

1. **Navigate to the notebooks directory:**
   ```bash
   cd script/build_analysis/notebooks
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open `01_initial_exploration.ipynb`** and run the cells to analyze a sample trace file.

## Notebooks

### 01_initial_exploration.ipynb

Initial exploration of trace data demonstrating:
- Safe streaming JSON parsing (memory-efficient)
- Event type distribution analysis
- Template instantiation identification
- Duration-based performance metrics
- Basic visualizations

**Key Features:**
- Analyzes a single file as a test case
- Identifies top time-consuming events
- Finds most frequently instantiated templates
- Generates summary statistics

## Utilities

### trace_parser.py

Core utilities for working with trace files:

- `iter_trace_files()` - Iterate over trace files in a directory
- `stream_events()` - Stream events from a file without loading into memory
- `load_trace_metadata()` - Load file metadata only
- `filter_events()` - Filter events by name or duration
- `get_template_events()` - Extract template-related events
- `aggregate_by_name()` - Compute statistics by event type
- `get_top_events()` - Find top N events by any field
- `extract_template_detail()` - Get template name from event

**Example usage:**

```python
from pathlib import Path
from utils.trace_parser import stream_events, get_template_events

# Stream events from a trace file
trace_file = Path('../../build-trace/some_file.json')
for event in stream_events(trace_file):
    print(event['name'], event.get('dur', 0))

# Get only template instantiation events
for event in get_template_events(stream_events(trace_file)):
    print(event['args'].get('detail'))
```

## Data Format

The trace files use the [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview):

```json
{
  "traceEvents": [
    {
      "pid": 1234,
      "tid": 1234,
      "ts": 1000,
      "dur": 500,
      "ph": "X",
      "name": "InstantiateFunction",
      "args": {
        "detail": "template_name<Args...>"
      }
    }
  ],
  "beginningOfTime": 1234567890
}
```

**Key fields:**
- `name`: Event type (e.g., "InstantiateClass", "ParseFunctionDefinition")
- `dur`: Duration in microseconds
- `ts`: Timestamp in microseconds
- `args.detail`: Additional information (e.g., template name)

## Memory Safety

All utilities are designed to handle large files safely:

- **Streaming parsing**: Uses `ijson` to parse JSON incrementally
- **Iterator-based**: Process events one at a time
- **Minimal materialization**: Only load data into memory when necessary
- **Top-N queries**: Use heaps to avoid loading all events

This allows analysis of 4+ MB JSON files without memory issues.

## Analysis Workflow

1. **Exploration** (Notebook 01): Understand data structure and identify patterns
2. **Template Analysis** (Future): Deep dive into template instantiation costs
3. **Visualization** (Future): Create interactive charts and timelines
4. **Optimization** (Future): Identify and track improvements

## Tips

- **Start small**: Analyze a single file first to understand the data
- **Use streaming**: Always use `stream_events()` for large files
- **Cache results**: Save aggregated data to `data/` for faster re-analysis
- **Filter early**: Use `filter_events()` to reduce data volume
- **Sample wisely**: For initial exploration, analyze a representative subset

## Future Enhancements

Planned additions:
- Multi-file aggregation across entire build
- Template dependency graph visualization
- Comparative analysis (before/after optimizations)
- Interactive flame graphs
- Build time regression detection
- Automated reporting

## Contributing

When adding new notebooks or utilities:
1. Follow the streaming/iterator pattern for memory safety
2. Document functions with clear docstrings
3. Include usage examples
4. Add visualizations where helpful
5. Update this README

## References

- [Clang Time Trace Documentation](https://releases.llvm.org/11.0.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-ftime-trace)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
- [Template Metaprogramming Performance](https://www.youtube.com/watch?v=vwrXHznaYLA)
