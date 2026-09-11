# Composable Kernel Tools

This directory contains utility tools for building, testing, and analyzing Composable Kernel.

These tools are designed to be LLM-agnostic and can be used with any AI assistant or directly from the command line.

## Available Tools

### ck-docker

Build and test composable_kernel in Docker with ROCm support.

See [README_ck-docker.md](README_ck-docker.md) for details.

**Quick start:**
```bash
# Add to PATH
export PATH="$PATH:$PWD/script/tools"

# Start container and build
ck-docker start
ck-docker build test_amdgcn_mma
ck-docker test test_amdgcn_mma
```

### ck-build-analysis

Analyze Composable Kernel build times using Clang's -ftime-trace profiler.

See [README_ck-build-analysis.md](README_ck-build-analysis.md) for details.

**Quick start:**
```bash
# Add to PATH
export PATH="$PATH:$PWD/script/tools"

# Analyze build time
ck-build-analysis example_convnd_fwd_xdl_fp8
```

## LLM Assistant Integration

These tools can be used as-is with any LLM assistant by providing the tool documentation to the assistant. The assistant can then invoke these tools on your behalf.

For example, you can ask:
- "Start the docker container"
- "Build and test test_amdgcn_mma"
- "Analyze build time for example_convnd_fwd_xdl_fp8"

The assistant will translate your natural language request into the appropriate tool invocation.

## Dependencies

- **ck-docker**: Requires Docker and ROCm-capable GPU (for running tests)
- **ck-build-analysis**: Requires Docker, automatically installs Python dependencies (jinja2) via `uv`

## Directory Structure

```
script/tools/
├── README.md                          # This file
├── README_ck-docker.md                # Documentation for ck-docker
├── README_ck-build-analysis.md        # Documentation for ck-build-analysis
├── ck-docker                          # Docker container management tool
├── ck-build-analysis                  # Build time analysis tool
├── common.sh                          # Shared utilities for bash scripts
├── analyze_build_trace.py             # Python script for trace analysis (PEP 723 compliant)
└── templates/
    └── build_analysis_report.md.jinja # Jinja2 template for analysis reports
```

## Contributing

When adding new tools to this directory:
1. Keep them LLM-agnostic (avoid hardcoding references to specific AI assistants)
2. Provide clear command-line usage documentation
3. Include examples for both CLI and LLM assistant usage
4. Follow the existing naming convention and structure
