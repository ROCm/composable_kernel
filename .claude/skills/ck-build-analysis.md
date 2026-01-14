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
  --granularity=N      Time trace granularity in microseconds (default: 100)
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
export CK_CONTAINER_NAME=my_build       # Override container name
export CK_BUILD_ANALYSIS_GRANULARITY=100  # Default granularity in µs
```

## Examples

```bash
# Basic analysis with default granularity (100µs)
ck-build-analysis example_convnd_fwd_xdl_fp8

# Quick overview (500µs granularity, filters minor events)
ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=500

# High-resolution analysis (1µs granularity, complete picture)
ck-build-analysis example_convnd_fwd_xdl_fp8 --granularity=1

# Custom output filename
ck-build-analysis example_convnd_fwd_xdl_fp8 --output=fp8_conv_analysis.md

# Analyze test target
ck-build-analysis test_amdgcn_mma

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
| 500µs | ~50k | 3-5 MB | Quick overview, major bottlenecks only |
| 100µs (default) | ~150k | 15-20 MB | Balanced detail and performance |
| 50µs | ~200k | 30-40 MB | Detailed analysis |
| 1µs (high-res) | ~300k | 80-100 MB | Complete picture, all instantiations |

## Notes

- Lower granularity = more events = larger files = longer analysis
- Default 100µs provides balanced detail for most use cases
- 500µs captures only major bottlenecks (filters out 86% of instantiations)
- 1µs granularity reveals all 36,000+ instantiations but takes longer to analyze

## Implementation Details

### PEP 723 Compliance with Automatic Dependency Management

The analysis script (`analyze_build_trace.py`) is PEP 723 compliant with inline dependency metadata:

```python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "jinja2>=3.0.0",
# ]
# ///
```

**The skill automatically uses `uv run` if available**, which provides:
- ✅ Zero-configuration dependency management
- ✅ Automatic installation of jinja2 from PEP 723 metadata
- ✅ Isolated dependency environment (no system pollution)
- ✅ Fast caching for subsequent runs

### Installation Options

**Option 1: Install uv (Recommended)**
```bash
# Install uv in the Docker container (one-time setup)
docker exec ck_<container_name> bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
```

After installing `uv`, the skill will automatically use it for dependency management.

**Option 2: Use system python3 + jinja2**
```bash
# If uv is not available, install jinja2 manually
docker exec ck_<container_name> apt-get install -y python3-jinja2
```

The skill automatically detects which method is available and uses the appropriate one.

### Components

- **ck-build-analysis** - Main bash script that orchestrates Docker, CMake, and analysis
- **analyze_build_trace.py** - PEP 723 compliant Python script for trace analysis
- **templates/build_analysis_report.md.jinja** - Jinja2 template for report generation

### Standalone Usage

The Python script can also be run independently:

```bash
# With uv (recommended - auto-installs dependencies)
uv run .claude/skills/analyze_build_trace.py trace.json report.md target 100 22 templates/

# With pipx (alternative - auto-installs dependencies)
pipx run .claude/skills/analyze_build_trace.py trace.json report.md target 100 22 templates/

# With python3 (requires jinja2 pre-installed)
python3 .claude/skills/analyze_build_trace.py trace.json report.md target 100 22 templates/
```
