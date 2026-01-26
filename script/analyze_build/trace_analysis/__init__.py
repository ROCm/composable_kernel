# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Build Trace Analysis - Core library for analyzing Clang -ftime-trace data.

This package provides tools to parse and analyze Clang's -ftime-trace JSON output
for build performance analysis.
"""

from .parse_file import (
    parse_file,
    get_metadata,
)

from .template_analysis import (
    get_template_instantiation_events,
)

from .phase_breakdown import (
    get_phase_breakdown,
    PhaseBreakdown,
)

from .parse_build import (
    find_trace_files,
    read_trace_files,
)

from .pipeline import (
    Pipeline,
)

from .build_helpers import (
    get_trace_file,
)

__all__ = [
    # Core parsing and filtering
    "parse_file",
    "get_metadata",
    "find_trace_files",
    "read_trace_files",
    # Pipeline processing
    "Pipeline",
    # Template analysis
    "get_template_instantiation_events",
    # Phase breakdown
    "get_phase_breakdown",
    "PhaseBreakdown",
    # Build helpers
    "get_trace_file",
]
