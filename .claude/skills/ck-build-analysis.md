# ck-build-analysis

Analyze Composable Kernel build times using Clang's -ftime-trace profiler.

## Terminal Usage

Direct command-line usage:

```bash
# From composable_kernel directory
.claude/skills/ck-build-analysis example_convnd_fwd_xdl_fp8
.claude/skills/ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=1
.claude/skills/ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=1 --output=my_report.md

# Or add to PATH
export PATH="$PATH:$PWD/.claude/skills"
ck-build-analysis example_convnd_fwd_xdl_fp8
```

## Ask Claude

Just ask in natural language:
- "Analyze build time for example_convnd_fwd_xdl_fp8"
- "Profile the compilation of test_amdgcn_mma with 1µs granularity"
- "Generate a build time report for example_gemm_xdl"

## Commands

```
ck-build-analysis <target> [options]

Options:
  --granularity=N      Time trace granularity in microseconds (default: 500)
  --output=FILE        Output report filename (default: build_time_analysis_report.md)
  --name=NAME          Docker container name (default: from CK_CONTAINER_NAME or auto-generated)
  --no-reconfigure     Skip CMake reconfiguration if build exists
  --help               Show this help message
```

## What It Does

1. **Configures CMake** with `-ftime-trace` and custom granularity
2. **Builds the target** using Ninja in Docker
3. **Analyzes the trace** JSON file for template instantiation patterns
4. **Generates a report** with:
   - Compilation phase breakdown
   - Top expensive individual instantiations
   - Template families ranked by total time and count
   - Key insights and optimization recommendations
   - Complete statistics

## Configuration

- **Container**: Uses ck-docker container (auto-starts if needed)
- **Granularity**: Default 500µs (use 1µs for high-resolution, 100µs for medium)
- **Output**: Markdown report in project root

## Environment

```bash
export CK_CONTAINER_NAME=my_build     # Override container name
export CK_BUILD_ANALYSIS_GRANULARITY=1  # Default granularity in µs
```

## Examples

```bash
# Basic analysis with default granularity (500µs)
ck-build-analysis example_convnd_fwd_xdl_fp8

# High-resolution analysis (1µs granularity, 22x larger trace)
ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=1

# Medium-resolution analysis (100µs granularity, good balance)
ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=100

# Custom output filename
ck-build-analysis example_convnd_fwd_xdl_fp8 --output=fp8_conv_analysis.md

# Analyze test target
ck-build-analysis test_amdgcn_mma --granularity=1

# Use existing build (skip reconfigure)
ck-build-analysis example_convnd_fwd_xdl_fp8 --no-reconfigure
```

## Output

The report includes:
- **Executive Summary**: Total time, events, instantiations, unique templates
- **Compilation Phases**: InstantiateFunction, Frontend, Backend, Optimizer, etc.
- **Top 30 Individual Instantiations**: Most expensive single templates
- **Template Families**: Grouped by total time and instantiation count
- **Key Insights**: What's slow and why
- **Optimization Recommendations**: Short, medium, and long-term strategies
- **Detailed Statistics**: Averages, medians, distributions

## Granularity Trade-offs

| Granularity | Events | Trace Size | Use Case |
|-------------|--------|------------|----------|
| 500µs (default) | ~50k | 3-5 MB | Quick overview, major bottlenecks |
| 100µs | ~150k | 15-20 MB | Balanced detail and performance |
| 50µs | ~200k | 30-40 MB | Detailed analysis |
| 1µs (high-res) | ~300k | 80-100 MB | Complete picture, all instantiations |

## Notes

- Lower granularity = more events = larger files = longer analysis
- Default 500µs captures major bottlenecks (filters out 86% of instantiations)
- 1µs granularity reveals all 36,000+ instantiations but takes longer to analyze
- 100µs is a good middle ground for most use cases
