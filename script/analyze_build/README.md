# Build Trace Analysis

Simple, fast tools for analyzing Clang `-ftime-trace` build performance data.

## Overview

This directory provides straightforward Python tools for analyzing the JSON trace files generated during compilation with `-ftime-trace`. The focus is on simplicity and speed - no caching, no complexity, just fast parallel I/O and pandas DataFrames.

**Key principle: Fresh analysis every time is faster and simpler than managing caches.**

## Quick Start

```bash
# Analyze all trace files in a directory
cd script/analyze_build/examples
python analyze_build.py ../../build-trace

# Analyze a single file
python analyze_file.py ../../build-trace/some_file.json
```

## Installation

Install required Python packages:

```bash
pip install pandas orjson tqdm
```

**Performance Note**: `orjson` provides a 1.65x speedup in JSON parsing. The parser automatically uses it if available, otherwise falls back to the standard library.

## Directory Structure

```
script/analyze_build/
├── trace_analysis/          # Core library
│   ├── __init__.py         # Main exports
│   ├── models.py           # TraceFile model
│   ├── parser.py           # Fast JSON parsing
│   └── transformer.py      # DataFrame conversion
├── examples/
│   ├── analyze_build.py    # Analyze all files in a directory
│   └── analyze_file.py     # Analyze a single file
├── notebooks/              # Jupyter notebooks for analysis
│   └── (existing notebooks)
└── README.md               # This file
```

## Usage

### Command-Line Analysis

**Analyze all trace files:**

```bash
python examples/analyze_build.py ../../build-trace
```

This will:
- Find all `.json` files recursively
- Process them in parallel using all CPU cores
- Display comprehensive build statistics
- Show top event types, slowest files, and template analysis

**Analyze a single file:**

```bash
python examples/analyze_file.py ../../build-trace/some_file.json
```

### Python API

```python
from pathlib import Path
from trace_analysis import TraceFile, TraceParser, TraceTransformer

# Parse a single file
trace_file = TraceFile.from_path(Path("build.json"))
events = TraceParser.parse(trace_file)

# Convert to DataFrames
events_df = TraceTransformer.to_events_dataframe(events)
templates_df = TraceTransformer.to_templates_dataframe(events)

# Analyze
print(f"Total events: {len(events_df):,}")
print(f"Total time: {events_df['dur'].sum() / 1e6:.2f}s")
print(f"Template time: {templates_df['dur'].sum() / 1e6:.2f}s")
```

### Jupyter Notebooks

For interactive analysis, see the comprehensive example notebook:

**[notebooks/comprehensive_example.ipynb](notebooks/comprehensive_example.ipynb)** - Complete guide covering:
- Single file analysis with detailed explanations
- Multi-file parallel processing
- Build-wide statistics and template analysis
- Advanced analysis patterns (optimization targets, distributions, etc.)
- Practical recommendations for improving build times

Quick example for custom notebooks:

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from trace_analysis import TraceFile, TraceParser, TraceTransformer
import pandas as pd

def process_file(json_path):
    trace_file = TraceFile.from_path(json_path)
    events = TraceParser.parse(trace_file)
    return TraceTransformer.to_events_dataframe(events)

# Process all files in parallel
trace_dir = Path("../../build-trace")
json_files = list(trace_dir.rglob("*.json"))

with ProcessPoolExecutor() as executor:
    dfs = list(executor.map(process_file, json_files))

# Combine and analyze
events_df = pd.concat(dfs, ignore_index=True)

# Top event types
event_totals = events_df.groupby('name')['dur'].sum().sort_values(ascending=False)
print(event_totals.head(10))
```

## Performance

**Typical performance on 4,484 trace files (~46 GB):**
- Parsing: ~26 seconds (174 files/sec)
- Memory: ~1-2 GB
- Throughput: I/O limited (uses all CPU cores)

**Why no caching?**
- Fresh analysis is faster than cache management overhead
- Simpler code (60% less code than cached version)
- No cache invalidation issues
- Catches changes immediately

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

## Library Components

### TraceFile

Simple model for trace file metadata:

```python
@dataclass
class TraceFile:
    path: Path
    size_bytes: int
    mtime_ns: int
    
    @classmethod
    def from_path(cls, path: Path) -> "TraceFile"
```

### TraceParser

Fast JSON parsing with orjson support:

```python
class TraceParser:
    @staticmethod
    def parse(trace_file: TraceFile) -> List[Dict[str, Any]]
```

Automatically uses `orjson` if available for 1.65x speedup.

### TraceTransformer

Convert parsed events to pandas DataFrames:

```python
class TraceTransformer:
    @staticmethod
    def to_events_dataframe(events: List[Dict]) -> pd.DataFrame
    
    @staticmethod
    def to_templates_dataframe(events: List[Dict]) -> pd.DataFrame
```

The events DataFrame includes all events with optimized dtypes. The templates DataFrame filters to template-related events and extracts template details.

## Analysis Examples

### Find Most Expensive Event Types

```python
event_totals = events_df.groupby('name')['dur'].sum()
top_events = event_totals.sort_values(ascending=False).head(10)
print(top_events / 1e6)  # Convert to seconds
```

### Find Slowest Files

```python
file_totals = events_df.groupby('file_name')['dur'].sum()
slowest = file_totals.sort_values(ascending=False).head(10)
print(slowest / 1e6)  # Convert to seconds
```

### Analyze Template Instantiations

```python
# Most frequently instantiated
template_counts = templates_df['template_detail'].value_counts()
print(template_counts.head(10))

# Most expensive by total time
template_totals = templates_df.groupby('template_detail')['dur'].sum()
print(template_totals.sort_values(ascending=False).head(10) / 1e6)

# Template time percentage
total_time = events_df['dur'].sum()
template_time = templates_df['dur'].sum()
print(f"Template time: {(template_time / total_time) * 100:.1f}%")
```

## Tips

- **Use all CPU cores**: The tools automatically use all available cores for parallel processing
- **Memory is cheap**: 1-2GB for 4,484 files is acceptable on modern systems
- **Fresh is fast**: No cache overhead means consistent ~26s analysis time
- **Jupyter-friendly**: Progress bars work automatically in notebooks
- **Simple is better**: One straightforward approach, not multiple complex paths

## References

- [Clang Time Trace Documentation](https://releases.llvm.org/11.0.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-ftime-trace)
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
- [Template Metaprogramming Performance](https://www.youtube.com/watch?v=vwrXHznaYLA)
