# Chrome Trace Export for Cross-Validation

**Status**: Design Document  
**Author**: Build Analysis Team  
**Date**: January 2026  
**Version**: 1.0

## Executive Summary

This document proposes adding Chrome Trace Event Format export capabilities to the `analyze_build` library to enable cross-validation with the existing `ninja_json_converter.py` tool. The two tools serve complementary purposes and this enhancement will allow verification of data consistency between them.

## Background

### Current State: Two Complementary Tools

The project currently has two distinct build analysis tools:

#### 1. `ninja_json_converter.py` - Build System Monitoring
- **Purpose**: Monitor build-level parallelism and efficiency
- **Primary Users**: Build engineers, CI/CD optimization teams
- **Key Metrics**: Worker utilization, critical path, slow compilation units
- **Output Format**: Chrome Trace Event Format (JSON)
- **Granularity**: File-level (compilation units)
- **Visualization**: Perfetto / Chrome Tracing UI
- **Use Cases**:
  - Is our build sharding efficient?
  - Which files are compilation bottlenecks?
  - How well are we utilizing available CPU cores?
  - What's the critical path in our build?

#### 2. `analyze_build` Library - Compiler Performance Analysis
- **Purpose**: Deep analysis of C++ template metaprogramming costs
- **Primary Users**: C++ developers, library maintainers, performance engineers
- **Key Metrics**: Template instantiation times, template relationships, compiler event breakdown
- **Output Format**: Pandas DataFrames for statistical analysis
- **Granularity**: Template-level and compiler event-level (within compilation)
- **Visualization**: Jupyter notebooks with statistical analysis
- **Use Cases**:
  - Which templates are most expensive to instantiate?
  - What are the template dependency relationships?
  - How can we optimize our metaprogramming patterns?
  - How can we measure improved build times with better metaprogramming?
  - What percentage of build time is template instantiation?

### The Problem: Need for Cross-Validation

Currently, these tools operate independently with no mechanism to verify consistency. This creates several challenges:

1. **Data Accuracy**: No way to verify both tools are parsing the same underlying data correctly
2. **Discrepancy Detection**: When numbers differ, unclear which tool is correct
3. **Cross-Referencing**: Difficult to correlate findings (e.g., "slow file in ninja" vs "high template time in analyzer")
4. **Debugging**: Hard to diagnose when tools report different build times
5. **Trust**: Users may question which tool's numbers to believe

## Goals and Non-Goals

### Primary Goals

1. **Enable Cross-Validation**: Export analyze_build data to Chrome Trace format for comparison with ninja_json_converter
2. **Verify Consistency**: Provide utilities to compare outputs and identify discrepancies
3. **Sanity Checking**: Quick visual verification in Perfetto that data looks correct
4. **Cross-Reference Findings**: Correlate slow files with expensive templates

### Secondary Goals

1. **Template Event Visualization**: Optionally export template instantiation events as additional trace layer
2. **Debugging Support**: Help diagnose when tools report different results
3. **Documentation**: Clear workflow for validation process

### Explicit Non-Goals

1. **Not Replacing ninja_json_converter**: The tools serve different purposes and both should continue to exist
2. **Not Full-Featured Visualization**: analyze_build focuses on statistical analysis, not interactive timelines
3. **Not Advanced Timeline Features**: Keep it simple - just export for validation
4. **Not Multi-Build Comparison**: ninja_json_converter already handles this well

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    analyze_build Library                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ NinjaParser  │─────▶│  builds_df   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│  ┌──────────────┐      ┌──────▼───────┐                     │
│  │ TraceParser  │─────▶│  events_df   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                        ┌──────▼───────────┐                 │
│                        │ ChromeTraceExporter│                │
│                        └──────┬───────────┘                 │
│                               │                              │
│                        ┌──────▼───────────┐                 │
│                        │  trace_events    │                 │
│                        │  (Chrome Format) │                 │
│                        └──────┬───────────┘                 │
│                               │                              │
└───────────────────────────────┼──────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Validation Utilities  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ ninja_json_converter   │
                    │      output            │
                    └────────────────────────┘
```

### New Module: `trace_analysis/chrome_trace.py`

```python
"""
Chrome Trace Event Format export for cross-validation.

Exports trace analysis data to Chrome Trace Event Format compatible
with ninja_json_converter.py output for validation purposes.
"""

from typing import Dict, List, Optional, Any
import pandas as pd


class ChromeTraceExporter:
    """Export trace analysis data to Chrome Trace Event Format."""
    
    @staticmethod
    def export_ninja_timeline(
        builds_df: pd.DataFrame,
        process_id: int = 1,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Export ninja build timeline to Chrome Trace format.
        
        Creates trace events compatible with ninja_json_converter.py output
        for cross-validation purposes.
        
        Args:
            builds_df: DataFrame with columns: target, start_ms, end_ms, 
                      duration_ms, worker_id, (optional) category
            process_id: Process ID for trace events (default: 1)
            include_metadata: Include trace metadata (default: True)
            
        Returns:
            Dictionary in Chrome Trace Event Format:
            {
                'traceEvents': [...],
                'displayTimeUnit': 'ms',
                'otherData': {...}
            }
            
        Example:
            >>> trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)
            >>> with open('trace.json', 'w') as f:
            ...     json.dump(trace_data, f)
        """
        
    @staticmethod
    def export_template_events(
        instantiations_df: pd.DataFrame,
        templates_df: pd.DataFrame,
        builds_df: pd.DataFrame,
        process_id: int = 1,
        granularity_us: int = 50000
    ) -> Dict[str, Any]:
        """
        Export template instantiation events as Chrome Trace layer.
        
        Creates template-level trace events that can be overlaid on the
        ninja build timeline for detailed compiler analysis.
        
        Args:
            instantiations_df: Template instantiation events
            templates_df: Template definitions
            builds_df: Ninja builds (for timing alignment)
            process_id: Process ID for trace events
            granularity_us: Minimum duration threshold in microseconds
            
        Returns:
            Chrome Trace Event Format dictionary with template events
            
        Note:
            Template events are aligned with ninja build timing and
            filtered by granularity threshold to reduce trace size.
        """
        
    @staticmethod
    def merge_traces(
        ninja_trace: Dict[str, Any],
        template_trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge ninja and template traces into single trace file.
        
        Combines build-level and template-level events for unified
        visualization in Perfetto.
        
        Args:
            ninja_trace: Ninja build timeline trace
            template_trace: Template instantiation trace
            
        Returns:
            Merged trace with both event types
        """
```

### New Module: `trace_analysis/validation.py`

```python
"""
Validation utilities for cross-checking trace analysis tools.

Compares outputs from analyze_build and ninja_json_converter to
verify data consistency and identify discrepancies.
"""

from typing import Dict, List, Any, Optional
import pandas as pd


class TraceValidator:
    """Validate consistency between trace analysis tools."""
    
    @staticmethod
    def compare_traces(
        analyzer_trace: Dict[str, Any],
        ninja_converter_trace: Dict[str, Any],
        tolerance_ms: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compare Chrome Trace outputs from both tools.
        
        Validates that analyze_build and ninja_json_converter produce
        consistent results from the same underlying data.
        
        Args:
            analyzer_trace: Trace from ChromeTraceExporter
            ninja_converter_trace: Trace from ninja_json_converter.py
            tolerance_ms: Acceptable time difference in milliseconds
            
        Returns:
            Validation report:
            {
                'total_time_match': bool,
                'total_time_diff_ms': float,
                'event_count_match': bool,
                'event_count_diff': int,
                'file_discrepancies': [
                    {
                        'file': str,
                        'analyzer_ms': float,
                        'ninja_ms': float,
                        'diff_ms': float,
                        'diff_pct': float
                    }
                ],
                'summary': str
            }
        """
        
    @staticmethod
    def validate_ninja_log_parsing(
        builds_df: pd.DataFrame,
        ninja_log_path: str
    ) -> Dict[str, Any]:
        """
        Validate that NinjaLogParser correctly parsed .ninja_log.
        
        Cross-checks parsed DataFrame against raw .ninja_log file
        to ensure no data loss or corruption.
        
        Args:
            builds_df: Parsed builds DataFrame
            ninja_log_path: Path to original .ninja_log file
            
        Returns:
            Validation report with any parsing issues
        """
        
    @staticmethod
    def generate_validation_report(
        validation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate human-readable validation report.
        
        Creates formatted report of validation results for review.
        
        Args:
            validation_results: Results from compare_traces()
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
```

### Data Flow

```
1. Parse .ninja_log
   └─> NinjaLogParser.parse() -> builds_df

2. Export to Chrome Trace
   └─> ChromeTraceExporter.export_ninja_timeline(builds_df) -> trace_data

3. Save trace file
   └─> json.dump(trace_data, 'analyzer_trace.json')

4. Generate ninja_json_converter trace (separately)
   └─> python ninja_json_converter.py .ninja_log -o ninja_trace.json

5. Validate consistency
   └─> TraceValidator.compare_traces(analyzer_trace, ninja_trace) -> report

6. Review discrepancies
   └─> TraceValidator.generate_validation_report(report)
```

## Chrome Trace Event Format Specification

### Event Structure

Each trace event follows the Chrome Trace Event Format:

```json
{
  "name": "target_name.o",
  "cat": "compile",
  "ph": "X",
  "ts": 1234567890,
  "dur": 5000000,
  "pid": 1,
  "tid": 3,
  "args": {
    "output": "target_name.o",
    "duration_ms": 5000,
    "cmd_hash": "abc123"
  }
}
```

**Field Descriptions:**
- `name`: Target name (file being built)
- `cat`: Category (compile, link_shared, link_executable, archive, test, other)
- `ph`: Phase ("X" for complete events)
- `ts`: Timestamp in microseconds
- `dur`: Duration in microseconds
- `pid`: Process ID (1 for ninja builds)
- `tid`: Thread ID (worker ID)
- `args`: Additional metadata

### Trace File Structure

```json
{
  "traceEvents": [
    { /* event 1 */ },
    { /* event 2 */ },
    ...
  ],
  "displayTimeUnit": "ms",
  "otherData": {
    "version": "1.0",
    "generator": "trace_analysis",
    "source": "analyze_build"
  }
}
```

### Compatibility with ninja_json_converter

The export format must be **byte-for-byte compatible** with ninja_json_converter output for the same input data, with these exceptions:

**Acceptable Differences:**
- `otherData.generator`: Different tool name
- Event ordering: May differ if timestamps are identical
- Floating point precision: ±0.001ms acceptable

**Must Match Exactly:**
- Total build time
- Per-file durations (within tolerance)
- Worker assignments
- Event counts
- Category assignments

## Validation Strategy

### Validation Checks

1. **Total Build Time**
   - Sum of all event durations
   - Should match within ±1ms (rounding tolerance)

2. **Event Count**
   - Number of trace events
   - Should match exactly

3. **Per-File Duration**
   - Duration for each compilation unit
   - Should match within ±1ms per file

4. **Worker Assignment**
   - Thread ID (worker) for each event
   - Should match exactly (deterministic algorithm)

5. **Category Assignment**
   - Event category based on file extension
   - Should match exactly

### Expected Discrepancies

Some differences are expected and acceptable:

1. **Timestamp Precision**: Microsecond rounding differences
2. **Event Ordering**: When timestamps are identical
3. **Metadata Fields**: Different tool names, versions
4. **Floating Point**: Minor precision differences (< 0.001ms)

### Validation Workflow

```python
# 1. Generate trace from analyze_build
from trace_analysis import NinjaLogParser, ChromeTraceExporter
import json

builds = NinjaLogParser.parse(Path('.ninja_log'))
builds_df = NinjaLogParser.to_dataframe(builds)
analyzer_trace = ChromeTraceExporter.export_ninja_timeline(builds_df)

with open('analyzer_trace.json', 'w') as f:
    json.dump(analyzer_trace, f)

# 2. Generate trace from ninja_json_converter (shell)
# $ python script/ninja_json_converter.py .ninja_log -o ninja_trace.json

# 3. Load both traces
with open('ninja_trace.json') as f:
    ninja_trace = json.load(f)

# 4. Validate
from trace_analysis import TraceValidator

report = TraceValidator.compare_traces(analyzer_trace, ninja_trace)

# 5. Review results
print(TraceValidator.generate_validation_report(report))
```

### Validation Report Format

```
=== Trace Validation Report ===

Overall Status: PASS / FAIL

Build Statistics:
  Total Events: 1,234 (analyzer) vs 1,234 (ninja) ✓
  Total Time: 123.456s (analyzer) vs 123.457s (ninja) ✓ (diff: 0.001s)
  
Worker Assignment:
  Match Rate: 100% (1,234/1,234 events) ✓
  
Per-File Duration:
  Files Checked: 1,234
  Exact Matches: 1,230 (99.7%)
  Within Tolerance: 4 (0.3%)
  Outside Tolerance: 0 (0.0%) ✓
  
Discrepancies:
  file1.o: 1234ms (analyzer) vs 1235ms (ninja) - diff: 1ms (0.08%)
  file2.o: 5678ms (analyzer) vs 5677ms (ninja) - diff: 1ms (0.02%)
  
Conclusion: Tools are consistent within acceptable tolerance.
```

## Implementation Plan

### Phase 1: Basic Export (Week 1)

**Deliverables:**
- `trace_analysis/chrome_trace.py` with `export_ninja_timeline()`
- Unit tests for Chrome Trace format
- Integration test comparing with ninja_json_converter

**Tasks:**
- [ ] Implement ChromeTraceExporter class
- [ ] Add event categorization logic
- [ ] Write unit tests for event generation
- [ ] Test with sample .ninja_log files
- [ ] Verify format matches ninja_json_converter exactly

**Success Criteria:**
- Exports valid Chrome Trace JSON
- Loads correctly in Perfetto
- Matches ninja_json_converter output for same input

### Phase 2: Validation Utilities (Week 1-2)

**Deliverables:**
- `trace_analysis/validation.py` with comparison utilities
- Validation report generator
- Documentation of validation workflow

**Tasks:**
- [ ] Implement TraceValidator class
- [ ] Add comparison algorithms
- [ ] Create validation report formatter
- [ ] Write tests for validation logic
- [ ] Document expected discrepancies

**Success Criteria:**
- Accurately identifies discrepancies
- Generates clear validation reports
- Handles edge cases gracefully

### Phase 3: Template Event Export (Week 2)

**Deliverables:**
- Template event export in `chrome_trace.py`
- Merged trace generation
- Examples in notebook

**Tasks:**
- [ ] Implement `export_template_events()`
- [ ] Add timing alignment logic
- [ ] Implement granularity filtering
- [ ] Add merge functionality
- [ ] Test with real -ftime-trace data

**Success Criteria:**
- Template events align with ninja timeline
- Granularity filtering works correctly
- Merged traces load in Perfetto

### Phase 4: Documentation & Examples (Week 2-3)

**Deliverables:**
- Updated README with validation workflow
- Notebook section demonstrating export
- API documentation
- Validation guide

**Tasks:**
- [ ] Add notebook section for Chrome Trace export
- [ ] Document validation workflow
- [ ] Create troubleshooting guide
- [ ] Add API documentation
- [ ] Write migration guide for ninja_json_converter users

**Success Criteria:**
- Clear documentation of validation process
- Working examples in notebook
- Users can successfully validate traces

## Testing Strategy

### Unit Tests

```python
# test_chrome_trace.py

def test_export_ninja_timeline_format():
    """Verify Chrome Trace format is valid."""
    
def test_export_ninja_timeline_compatibility():
    """Verify compatibility with ninja_json_converter."""
    
def test_event_categorization():
    """Verify file extension -> category mapping."""
    
def test_worker_assignment():
    """Verify worker IDs match ninja_json_converter."""
```

### Integration Tests

```python
# test_validation.py

def test_compare_identical_traces():
    """Validation passes for identical traces."""
    
def test_detect_discrepancies():
    """Validation detects timing differences."""
    
def test_tolerance_handling():
    """Small differences within tolerance pass."""
```

### Validation Tests

```python
# test_cross_validation.py

def test_real_ninja_log():
    """Compare with actual ninja_json_converter output."""
    
def test_large_build():
    """Handle large builds (1000+ files)."""
    
def test_incremental_build():
    """Handle incremental build scenarios."""
```

## Usage Examples

### Basic Export

```python
from pathlib import Path
from trace_analysis import NinjaLogParser, ChromeTraceExporter
import json

# Parse ninja log
builds = NinjaLogParser.parse(Path('build/.ninja_log'))
builds_df = NinjaLogParser.to_dataframe(builds)

# Export to Chrome Trace
trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)

# Save for Perfetto
with open('build_trace.json', 'w') as f:
    json.dump(trace_data, f)

print("Open build_trace.json in chrome://tracing or https://ui.perfetto.dev")
```

### Cross-Validation

```python
from trace_analysis import ChromeTraceExporter, TraceValidator
import json
import subprocess

# Generate trace from analyze_build
analyzer_trace = ChromeTraceExporter.export_ninja_timeline(builds_df)

# Generate trace from ninja_json_converter
subprocess.run([
    'python', 'script/ninja_json_converter.py',
    'build/.ninja_log',
    '-o', 'ninja_trace.json'
])

# Load ninja_json_converter output
with open('ninja_trace.json') as f:
    ninja_trace = json.load(f)

# Validate
report = TraceValidator.compare_traces(analyzer_trace, ninja_trace)

# Print report
print(TraceValidator.generate_validation_report(report))

# Check if validation passed
if report['total_time_match'] and report['event_count_match']:
    print("✓ Validation PASSED - Tools are consistent")
else:
    print("✗ Validation FAILED - Discrepancies found")
    for disc in report['file_discrepancies']:
        print(f"  {disc['file']}: {disc['diff_ms']}ms difference")
```

### Template Event Export

```python
from trace_analysis import (
    TraceParser, TraceTransformer, 
    ChromeTraceExporter, find_trace_files
)

# Parse -ftime-trace files
trace_files = find_trace_files(Path('build'))
all_events = []
all_instantiations = []

for trace_file in trace_files:
    events = TraceParser.parse(trace_file)
    schema = TraceTransformer.to_enhanced_schema(events, file_id=0)
    all_instantiations.append(schema['instantiations'])

instantiations_df = pd.concat(all_instantiations, ignore_index=True)

# Export template events
template_trace = ChromeTraceExporter.export_template_events(
    instantiations_df,
    templates_df,
    builds_df,
    granularity_us=50000  # Only events > 50ms
)

# Merge with ninja timeline
merged_trace = ChromeTraceExporter.merge_traces(
    ninja_trace,
    template_trace
)

# Save merged trace
with open('merged_trace.json', 'w') as f:
    json.dump(merged_trace, f)
```

### Notebook Integration

```python
# In comprehensive_example.ipynb

## Chrome Trace Export for Validation

# Export ninja timeline
from trace_analysis import ChromeTraceExporter
import json

trace_data = ChromeTraceExporter.export_ninja_timeline(builds_df)

# Save trace
with open('../data/analyzer_trace.json', 'w') as f:
    json.dump(trace_data, f, indent=2)

print(f"Exported {len(trace_data['traceEvents'])} events")
print(f"Total build time: {sum(e['dur'] for e in trace_data['traceEvents']) / 1e6:.2f}s")

# Validate against ninja_json_converter
# (Assuming ninja_trace.json was generated separately)
with open('../data/ninja_trace.json') as f:
    ninja_trace = json.load(f)

from trace_analysis import TraceValidator

report = TraceValidator.compare_traces(trace_data, ninja_trace)
print(TraceValidator.generate_validation_report(report))
```

## Open Questions

### Critical Questions

1. **Data Consistency**
   - Q: Do you currently see discrepancies between the tools?
   - Q: What tolerance is acceptable? (±1ms suggested)
   - Q: Are there known sources of differences?

2. **Validation Workflow**
   - Q: How often do you need to cross-validate?
   - Q: Should this be automated in CI?
   - Q: What triggers a validation run?

3. **Template Event Export**
   - Q: Should template events be in same file as ninja events?
   - Q: Or separate files for different analysis?
   - Q: Priority: High, Medium, or Low?

### Technical Questions

4. **Output Format**
   - Q: Must we match ninja_json_converter format exactly?
   - Q: Or can we use enhanced format with metadata?
   - Q: Is backward compatibility required?

5. **Performance**
   - Q: What's the largest build to support?
   - Q: Number of targets? (hundreds, thousands, tens of thousands?)
   - Q: Should we implement sampling for huge builds?

## Success Metrics

### Functional Metrics

- ✅ Exports valid Chrome Trace JSON
- ✅ Loads correctly in Perfetto
- ✅ Matches ninja_json_converter output (within tolerance)
- ✅ Validation detects discrepancies accurately
- ✅ Clear validation reports

### Quality Metrics

- ✅ 100% unit test coverage for new modules
- ✅ Integration tests with real data pass
- ✅ Documentation complete and clear
- ✅ Examples work in notebook

### Performance Metrics

- ✅ Export completes in < 1s for 1000 files
- ✅ Validation completes in < 5s for 1000 files
- ✅ Memory usage < 100MB for typical builds

## Future Enhancements

### Potential Phase 2 Features

1. **Automated Validation in CI**
   - Run validation on every build
   - Fail CI if discrepancies exceed threshold
   - Track validation metrics over time

2. **Differential Analysis**
   - Compare traces from different builds
   - Identify performance regressions
   - Track optimization progress

3. **Enhanced Visualization**
   - Plotly timeline charts in notebooks
   - Interactive exploration of discrepancies
   - Side-by-side comparison views

4. **Template Optimization Recommendations**
   - Correlate slow files with expensive templates
   - Suggest optimization targets
   - Estimate potential improvements

## References

- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
- [Perfetto UI](https://ui.perfetto.dev)
- [Clang -ftime-trace Documentation](https://releases.llvm.org/11.0.0/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-ftime-trace)
- [Ninja Build System](https://ninja-build.org/)

## Appendix A: Chrome Trace Event Format Details

### Complete Event Structure

```json
{
  "name": "event_name",
  "cat": "category",
  "ph": "X",
  "ts": 1234567890,
  "dur": 5000000,
  "pid": 1,
  "tid": 3,
  "args": {
    "custom_field": "value"
  }
}
```

### Phase Types

- `X`: Complete event (has duration)
- `B`: Begin event
- `E`: End event
- `i`: Instant event
- `M`: Metadata event

For build traces, we use `X` (complete events) exclusively.

### Category Conventions

Standard categories for build events:

- `compile`: Compilation of source files (.o, .obj)
- `link_shared`: Shared library linking (.so, .dll, .dylib)
- `link_executable`: Executable linking (.exe, .out)
- `archive`: Static library creation (.a, .lib)
- `test`: Test execution
- `other`: Other build steps

## Appendix B: Validation Algorithm

### Comparison Algorithm

```python
def compare_events(event1, event2, tolerance_ms=1.0):
    """Compare two trace events for equivalence."""
    
    # Must match exactly
    if event1['name'] != event2['name']:
        return False, "Name mismatch"
    if event1['tid'] != event2['tid']:
        return False, "Worker ID mismatch"
    if event1['cat'] != event2['cat']:
        return False, "Category mismatch"
    
    # Must match within tolerance
    dur1_ms = event1['dur'] / 1000
    dur2_ms = event2['dur'] / 1000
    diff_ms = abs(dur1_ms - dur2_ms)
    
    if diff_ms > tolerance_ms:
        return False, f"Duration mismatch: {diff_ms}ms"
    
    return True, "Match"
```

### Discrepancy Categorization

**Critical**: Must be fixed
- Total time difference > 1%
- Event count mismatch
- Worker assignment errors

**Warning**: Should investigate
- Per-file duration > 1ms difference
- Category mismatches
- Timestamp ordering issues

**Info**: Acceptable
- Floating point precision differences
- Metadata differences
- Event ordering when timestamps identical

## Appendix C: Migration Guide

### For ninja_json_converter Users

If you currently use `ninja_json_converter.py`, you can continue to do so. The new Chrome Trace export in `analyze_build` is complementary, not a replacement.

**When to use ninja_json_converter:**
- Quick build timeline visualization
- Build system optimization
- CI/CD monitoring
- Multi-build comparison

**When to use analyze_build Chrome Trace export:**
- Cross-validation with template analysis
- Verifying data consistency
- Debugging discrepancies
- Correlating build and template metrics

**Using both together:**
```bash
# Generate trace from ninja_json_converter
python script/ninja_json_converter.py build/.ninja_log -o ninja_trace.json

# Generate trace from analyze_build
python -c "
from pathlib import Path
from trace_analysis import NinjaLogParser, ChromeTraceExporter
import json

builds = NinjaLogParser.parse(Path('build/.ninja_log'))
builds_df = NinjaLogParser.to_dataframe(builds)
trace = ChromeTraceExporter.export_ninja_timeline(builds_df)

with open('analyzer_trace.json', 'w') as f:
    json.dump(trace, f)
"

# Compare
python -c "
from trace_analysis import TraceValidator
import json

with open('ninja_trace.json') as f:
    ninja = json.load(f)
with open('analyzer_trace.json') as f:
    analyzer = json.load(f)

report = TraceValidator.compare_traces(analyzer, ninja)
print(TraceValidator.generate_validation_report(report))
"
