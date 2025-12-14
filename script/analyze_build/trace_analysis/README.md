# Trace Analysis Library

Modern data pipeline for analyzing Clang `-ftime-trace` build performance data. Provides fast JSON parsing and conversion to pandas DataFrames for immediate analysis in scripts and Jupyter notebooks.

## Design Principles

1. **Modular, modern data pipeline architecture** - Clean separation between parsing, transformation, and analysis
2. **Fast path to pandas DataFrames** - Get to analyzable data structures quickly
3. **Simplicity and efficiency** - Straightforward code with optimized data structures
4. **Concurrent processing** - Leverage parallelism where it helps performance
5. **Flexible integration** - Standalone library that works well in scripts and interactive sessions

## Quick Start

```python
from pathlib import Path
from trace_analysis import TraceFile, TraceParser, TraceTransformer

# Parse a trace file
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

## Module Overview

### models.py - Data Models

**TraceFile**: Lightweight metadata wrapper for trace files.

```python
@dataclass
class TraceFile:
    path: Path
    size_bytes: int
    mtime_ns: int
    
    @classmethod
    def from_path(cls, path: Path) -> "TraceFile"
```

**Usage:**
```python
trace_file = TraceFile.from_path(Path("build.json"))
print(f"File: {trace_file.name}, Size: {trace_file.size_bytes:,} bytes")
```

### parser.py - JSON Parsing

**TraceParser**: Fast JSON parsing with automatic orjson optimization.

**Key Methods:**

```python
@staticmethod
def parse(trace_file: TraceFile) -> List[Dict[str, Any]]
    """Parse trace file and return all events."""

@staticmethod
def is_template_event(event: Dict[str, Any]) -> bool
    """Check if event is template-related."""

@staticmethod
def extract_template_detail(event: Dict[str, Any]) -> str
    """Extract template detail from event."""
```

**Performance Notes:**
- Automatically uses `orjson` if available (1.65x faster than stdlib json)
- Gracefully falls back to standard library
- Optimized for single-line JSON format used by -ftime-trace

**Usage:**
```python
events = TraceParser.parse(trace_file)
template_events = [e for e in events if TraceParser.is_template_event(e)]
```

### transformer.py - DataFrame Conversion

**TraceTransformer**: Convert parsed events to optimized pandas DataFrames.

**Key Methods:**

```python
@staticmethod
def to_events_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame
    """Convert all events to DataFrame with optimized dtypes."""

@staticmethod
def to_templates_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame
    """Convert template events to DataFrame with template details."""

@staticmethod
def compute_file_stats(events_df: pd.DataFrame, templates_df: pd.DataFrame, 
                       file_name: str) -> Dict[str, Any]
    """Compute summary statistics for a file."""

@staticmethod
def aggregate_event_types(events_df: pd.DataFrame) -> pd.DataFrame
    """Aggregate events by type with count, sum, mean, max."""

@staticmethod
def aggregate_templates(templates_df: pd.DataFrame) -> pd.DataFrame
    """Aggregate template instantiations by detail."""
```

**Usage:**
```python
# Convert to DataFrames
events_df = TraceTransformer.to_events_dataframe(events)
templates_df = TraceTransformer.to_templates_dataframe(events)

# Aggregate analysis
event_summary = TraceTransformer.aggregate_event_types(events_df)
template_summary = TraceTransformer.aggregate_templates(templates_df)
```

## DataFrame Schemas

> **Note**: The DataFrame schemas are evolving as analysis needs develop. The current focus is on fast intake of raw JSON data into a simple, analyzable structure.

### Events DataFrame

Columns and optimized dtypes:

| Column | Type | Description |
|--------|------|-------------|
| `name` | category | Event type (e.g., "InstantiateFunction") |
| `dur` | int64 | Duration in microseconds |
| `ts` | int64 | Timestamp in microseconds |
| `pid` | int32 | Process ID |
| `tid` | int32 | Thread ID |
| `ph` | category | Phase (usually "X" for complete events) |

**Example:**
```python
events_df.head()
#                      name      dur         ts   pid   tid ph
# 0  InstantiateFunction  12500  1000000  1234  1234  X
# 1       ParseClass       8300  1012500  1234  1234  X
```

### Templates DataFrame

Columns for template-specific analysis:

| Column | Type | Description |
|--------|------|-------------|
| `name` | category | Template event type |
| `dur` | int64 | Duration in microseconds |
| `template_detail` | object | Full template signature |

**Example:**
```python
templates_df.head()
#                  name    dur                    template_detail
# 0  InstantiateFunction  12500  std::vector<int, std::allocator<int>>
# 1   InstantiateClass    8300  MyClass<double, 3>
```

## Parallel Processing Pattern

For analyzing multiple files concurrently:

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def process_file(json_path: Path):
    trace_file = TraceFile.from_path(json_path)
    events = TraceParser.parse(trace_file)
    return TraceTransformer.to_events_dataframe(events)

# Process all files in parallel
trace_dir = Path("../../build-trace")
json_files = list(trace_dir.rglob("*.json"))

with ProcessPoolExecutor() as executor:
    dfs = list(executor.map(process_file, json_files))

# Combine results
all_events = pd.concat(dfs, ignore_index=True)
```

## Analysis Examples

### Top Event Types by Duration

```python
event_totals = events_df.groupby('name', observed=True)['dur'].sum()
top_events = event_totals.sort_values(ascending=False).head(10)
print(top_events / 1e6)  # Convert to seconds
```

### Most Expensive Templates

```python
template_totals = templates_df.groupby('template_detail')['dur'].agg(['sum', 'count', 'mean'])
expensive = template_totals.sort_values('sum', ascending=False).head(10)
print(expensive)
```

### Template Time Percentage

```python
total_time = events_df['dur'].sum()
template_time = templates_df['dur'].sum()
print(f"Template instantiation: {(template_time / total_time) * 100:.1f}% of build time")
```

## Future Directions

This library is actively evolving:

- **Schema evolution**: DataFrame schemas will expand as analysis needs develop
- **Visualization helpers**: Planned addition of common visualization utilities
- **Analysis utilities**: Additional aggregation and analysis functions as patterns emerge

The core pipeline architecture (parse → transform → analyze) will remain stable while specific schemas and utilities evolve based on practical analysis needs.

## See Also

- [Main README](../README.md) - Overall project documentation and usage examples
- [examples/](../examples/) - Complete analysis scripts
- [Clang -ftime-trace docs](https://releases.llvm.org/11.0.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-ftime-trace)
