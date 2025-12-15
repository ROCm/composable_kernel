# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data models for trace analysis.

Data structures for representing trace files, build events, templates,
and their relationships.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib


@dataclass
class TraceFile:
    """Represents a trace file to be processed."""

    path: Path
    size_bytes: int
    mtime_ns: int

    @property
    def name(self) -> str:
        """Get the filename."""
        return self.path.name

    @classmethod
    def from_path(cls, path: Path) -> "TraceFile":
        """Create a TraceFile from a path."""
        stat = path.stat()
        return cls(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


@dataclass
class NinjaBuild:
    """
    Represents a ninja build event from .ninja_log.

    Captures build-level parallelism and timing information.
    """

    target: str
    start_ms: int
    end_ms: int
    cmd_hash: str
    worker_id: int = -1

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0


@dataclass
class CompilationTimeline:
    """
    Represents compilation timeline metadata from -ftime-trace.

    Links trace file to wall-clock timing for timeline visualization.
    """

    file_path: Path
    beginning_of_time_us: int
    total_duration_us: int
    event_count: int

    @property
    def start_time_s(self) -> float:
        """Wall-clock start time in seconds since epoch."""
        return self.beginning_of_time_us / 1e6

    @property
    def duration_s(self) -> float:
        """Total compilation duration in seconds."""
        return self.total_duration_us / 1e6

    @property
    def file_name(self) -> str:
        """Get the filename."""
        return self.file_path.name


@dataclass
class Template:
    """
    Represents a unique C++ template with parsed structure.

    Captures template hierarchy and relationships for dependency analysis.
    """

    template_id: int
    template_name: str  # Base name before '<'
    full_signature: str  # Complete template with arguments
    signature_hash: int  # For deduplication
    depth: int  # Nesting level (0 = no template args, 1+ = nested)
    arg_count: int  # Number of template arguments

    @staticmethod
    def compute_hash(signature: str) -> int:
        """Compute a hash for template signature deduplication."""
        return int(hashlib.sha256(signature.encode()).hexdigest()[:16], 16)

    @classmethod
    def from_signature(
        cls, template_id: int, signature: str, depth: int = 0, arg_count: int = 0
    ) -> "Template":
        """
        Create a Template from a signature string.

        Args:
            template_id: Unique identifier
            signature: Full template signature
            depth: Nesting depth (computed by parser)
            arg_count: Number of template arguments (computed by parser)
        """
        # Extract base name (everything before first '<')
        base_name = signature.split("<")[0].strip() if "<" in signature else signature

        return cls(
            template_id=template_id,
            template_name=base_name,
            full_signature=signature,
            signature_hash=cls.compute_hash(signature),
            depth=depth,
            arg_count=arg_count,
        )


@dataclass
class TemplateArgument:
    """
    Represents a template argument relationship.

    Links parent templates to their argument templates for dependency analysis.
    """

    parent_template_id: int
    arg_position: int  # 0-indexed position in argument list
    arg_template_id: Optional[int]  # None if not a template type
    arg_type: str  # 'template', 'primitive', 'unknown'
    arg_text: str  # Raw argument text


@dataclass
class FileMetadata:
    """
    Metadata about a compilation unit.

    Links trace files to build events and aggregates statistics.
    """

    file_id: int
    file_name: str
    file_path: Path
    total_duration_us: int
    template_count: int
    event_count: int
    beginning_of_time_us: Optional[int] = None  # Wall-clock start time

    @property
    def duration_s(self) -> float:
        """Total compilation duration in seconds."""
        return self.total_duration_us / 1e6


@dataclass
class TemplateInstantiation:
    """
    Represents a single template instantiation event.

    Links to template definition and captures timing information.
    """

    instantiation_id: int
    template_id: int
    file_id: int
    dur_us: int
    ts_us: int  # Timestamp relative to compilation start
    event_type: str  # InstantiateClass, InstantiateFunction, etc.

    @property
    def dur_ms(self) -> float:
        """Duration in milliseconds."""
        return self.dur_us / 1000.0


@dataclass
class BuildStatistics:
    """
    Aggregated statistics for build analysis.

    Provides summary metrics for reporting and visualization.
    """

    total_files: int
    total_events: int
    total_duration_us: int
    template_instantiations: int
    template_duration_us: int
    unique_templates: int
    max_template_depth: int
    worker_count: int = 0
    build_duration_s: float = 0.0

    @property
    def total_duration_s(self) -> float:
        """Total compilation time in seconds."""
        return self.total_duration_us / 1e6

    @property
    def template_percentage(self) -> float:
        """Percentage of time spent in template instantiation."""
        if self.total_duration_us == 0:
            return 0.0
        return (self.template_duration_us / self.total_duration_us) * 100

    @property
    def avg_file_duration_s(self) -> float:
        """Average compilation time per file in seconds."""
        if self.total_files == 0:
            return 0.0
        return self.total_duration_s / self.total_files
