# Build Trace Analysis

Simple to use, fast python tools for analyzing Clang `-ftime-trace` build performance data.

## Overview

We're kicking off a systematic effort to dramatically reduce CK and CK-Tile build times, [#3575](https://github.com/ROCm/composable_kernel/issues/3575). A key part of this work is improving our C++ metaprogramming to reduce the burden on the compiler.

In order to prioritize work and measure our progress, we need data on template instantiation. For single files, Clang's `-ftime-trace` build performance data is easy to analyze with the Perfetto UI. The problem we are solving here is how to analyze instantiation data across thousands of compilation units.

The python code in this directory provides helper functions to quickly load JSON files into pandas DataFrames that can be used for analysis in Jupyter notebooks.

## Directory Structure

```
script/analyze_build/
├── trace_analysis/              # Core library
│   ├── __init__.py              # Main exports
│   ├── parse_file.py            # Fast parsing of JSON trace files
│   ├── template_analysis.py     # Template instantiation analysis
│   ├── template_parser.py       # Template name parsing utilities
│   └── phase_breakdown.py       # Compilation phase breakdown
├── notebooks/                   # Jupyter notebooks for analysis
│   └── file_analysis_example.ipynb  # Template analysis example
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Python Requirements

See `requirements.txt` for the complete list of dependencies:
* **pandas** - DataFrame manipulation and analysis
* **orjson** - Fast JSON parsing for trace files
* **plotly** - Interactive visualizations (sunburst, treemap)
* **nbformat** - Jupyter notebook format support
* **ipykernel** - Kernel for running notebooks in VSCode/Jupyter
* **kaleido** - Static image export from Plotly charts
* **jupyter** - Full Jupyter environment

## Quick Start

### Setup

1. Create a virtual environment (recommended):
```bash
cd script/analyze_build
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install VSCode extensions if you want to run notebooks in VSCode:
   * Jupyter
   * Data Wrangler (interact with Pandas DataFrames)

### Analyzing a Single File

Use the `parse_file` function to load a `-ftime-trace` JSON file into a Pandas DataFrame:

```python
from trace_analysis import parse_file

# Parse the trace file
df = parse_file('path/to/trace.json')

# View basic info
print(f"Total events: {len(df)}")
print(df.columns)

# Analyze duration statistics
print(df['dur'].describe())
```

### Extracting Compilation Metadata

Get high-level metadata about the compilation:

```python
from trace_analysis import get_metadata

# Extract metadata from trace file
metadata = get_metadata('trace.json')

print(f"Source file: {metadata['source_file']}")
print(f"Compilation time: {metadata['total_wall_time_s']:.2f}s")
print(f"Started: {metadata['wall_start_datetime']}")
print(f"Ended: {metadata['wall_end_datetime']}")
```

The metadata includes:
- `source_file`: Main .cpp/.c file being compiled
- `time_granularity`: Time unit used ("microseconds")
- `beginning_of_time`: Epoch timestamp in microseconds
- `wall_start_time`: Wall clock start (microseconds since epoch)
- `wall_end_time`: Wall clock end (microseconds since epoch)
- `wall_start_datetime`: Human-readable start time
- `wall_end_datetime`: Human-readable end time
- `total_wall_time_us`: Total compilation time in microseconds
- `total_wall_time_s`: Total compilation time in seconds

### Template Instantiation Analysis

The module includes specialized functions for analyzing C++ template instantiation costs:

```python
from trace_analysis import (
    parse_file,
    get_template_instantiation_events,
    get_phase_breakdown,
)

df = parse_file('trace.json')

# Get all template instantiation events with parsed template information
template_events = get_template_instantiation_events(df)

# The returned DataFrame includes parsed columns:
# - namespace: Top-level namespace (e.g., 'std', 'ck')
# - template_name: Template name without parameters
# - full_qualified_name: Full namespace::template_name
# - param_count: Number of template parameters
# - is_ck_type: Boolean indicating CK library types
# - is_nested: Boolean indicating nested templates

# Find slowest template instantiations
top_templates = template_events.nlargest(20, 'dur')
print(top_templates[['template_name', 'namespace', 'param_count', 'dur']])

# Analyze by namespace
namespace_summary = template_events.groupby('namespace').agg({
    'dur': ['count', 'sum', 'mean']
})
print(namespace_summary)
```

### Compilation Phase Breakdown

Analyze how compilation time is distributed across different phases:

```python
from trace_analysis import get_phase_breakdown, PhaseBreakdown

df = parse_file('trace.json')

# Get hierarchical phase breakdown
breakdown = get_phase_breakdown(df)

# Display in Jupyter (automatic rich HTML display)
display(breakdown)

# Print text representation
print(breakdown)

# Access the underlying DataFrame
print(breakdown.df)

# Convert to plotly format for visualization
import plotly.express as px
data = breakdown.to_plotly()
fig = px.sunburst(**data)
fig.show()
```

The `PhaseBreakdown` class provides:
- Hierarchical breakdown of compilation phases
- Automatic calculation of "Other" residual time at each level
- Validation that children don't exceed parent durations
- Multiple output formats (text, DataFrame, Plotly)

## DataFrame Schema

The parsed DataFrame contains the following columns from the `-ftime-trace` format:

- `name`: Event name (function, template instantiation, etc.)
- `ph`: Phase character ('X' for complete, 'B' for begin, 'E' for end, 'i' for instant)
- `ts`: Timestamp in microseconds
- `dur`: Duration in microseconds (for complete events)
- `pid`: Process ID
- `tid`: Thread ID
- `arg_*`: Flattened arguments from the event's `args` field

### Template Event Columns

When using `get_template_instantiation_events()`, additional parsed columns are included:

- `namespace`: Top-level namespace extracted from the template name
- `template_name`: Template name without namespace or parameters
- `full_qualified_name`: Complete namespace::template_name
- `param_count`: Number of template parameters
- `is_ck_type`: Boolean flag for CK library types (namespace starts with 'ck')
- `is_nested`: Boolean flag indicating nested template instantiations

## Use in Jupyter Notebooks

The module is designed to work seamlessly in Jupyter notebooks. See `notebooks/file_analysis_example.ipynb` for a complete example workflow that demonstrates:

- Loading and parsing trace files
- Extracting compilation metadata
- Analyzing phase breakdown with visualizations
- Template instantiation analysis with parsed columns
- Filtering and grouping by namespace
- Identifying CK-specific template costs

To use in a notebook:

```python
import sys
from pathlib import Path

# Add trace_analysis to path
sys.path.insert(0, str(Path.cwd().parent))

from trace_analysis import (
    parse_file,
    get_metadata,
    get_template_instantiation_events,
    get_phase_breakdown,
)

# Load and analyze
df = parse_file('path/to/trace.json')
breakdown = get_phase_breakdown(df)
templates = get_template_instantiation_events(df)

# Visualize
import plotly.express as px
fig = px.sunburst(**breakdown.to_plotly())
fig.show()
```

## API Reference

### Core Functions

- `parse_file(filepath)`: Parse a `-ftime-trace` JSON file into a pandas DataFrame
- `get_metadata(filepath_or_df)`: Extract compilation metadata from trace file or DataFrame

### Template Analysis

- `get_template_instantiation_events(df)`: Filter to template instantiation events with parsed template information

### Phase Breakdown

- `get_phase_breakdown(df)`: Generate hierarchical compilation phase breakdown
- `PhaseBreakdown`: Class representing phase breakdown with multiple output formats

## Contributing

This is an experimental project for analyzing and improving C++ metaprogramming build times. Contributions are welcome! When adding new analysis functions:

1. Add the function to the appropriate module in `trace_analysis/`
2. Export it in `__init__.py`
3. Update this README with usage examples
4. Consider adding a notebook example if the feature is substantial

## License

Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
