# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data models for trace analysis.

Simple data structures for representing trace files.
"""

from dataclasses import dataclass
from pathlib import Path


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
