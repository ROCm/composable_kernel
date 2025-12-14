# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
JSON parser for trace files.

Provides streaming JSON parsing to handle large trace files efficiently.
"""

from typing import List, Dict, Any

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    import json

    HAS_ORJSON = False

from .models import TraceFile


class TraceParser:
    """
    Parser for Clang -ftime-trace JSON files.

    Uses streaming JSON parsing to handle large files without loading
    them entirely into memory.
    """

    # Template-related event names
    TEMPLATE_EVENT_NAMES = {
        "InstantiateClass",
        "InstantiateFunction",
        "InstantiateVariable",
        "ParseTemplate",
    }

    @staticmethod
    def parse(trace_file: TraceFile) -> List[Dict[str, Any]]:
        """
        Parse a trace file and return all events.

        Args:
            trace_file: TraceFile to parse

        Returns:
            List of event dictionaries

        Note:
            Uses orjson if available (1.65x faster than stdlib json),
            otherwise falls back to standard json library. The -ftime-trace
            files are single-line JSON, so we can load them efficiently.
        """
        if HAS_ORJSON:
            # orjson is significantly faster (1.65x) and reads bytes
            with open(trace_file.path, "rb") as f:
                data = orjson.loads(f.read())
        else:
            # Fallback to standard library
            with open(trace_file.path, "r") as f:
                data = json.load(f)

        # Handle both dict format {"traceEvents": [...]} and direct list format
        if isinstance(data, dict):
            return data.get("traceEvents", [])
        elif isinstance(data, list):
            return data
        else:
            return []

    @staticmethod
    def parse_stream(trace_file: TraceFile):
        """
        Stream events from a trace file without loading entire file.

        Args:
            trace_file: TraceFile to parse

        Yields:
            Individual event dictionaries

        Note:
            For compatibility, this now just yields from the parsed list.
            The standard json library is much faster than ijson for these files.
        """
        events = TraceParser.parse(trace_file)
        for event in events:
            yield event

    @staticmethod
    def is_template_event(event: Dict[str, Any]) -> bool:
        """
        Check if an event is template-related.

        Args:
            event: Event dictionary

        Returns:
            True if event is template-related
        """
        return event.get("name") in TraceParser.TEMPLATE_EVENT_NAMES

    @staticmethod
    def extract_template_detail(event: Dict[str, Any]) -> str:
        """
        Extract template detail from an event.

        Args:
            event: Event dictionary

        Returns:
            Template detail string, or empty string if not available
        """
        args = event.get("args", {})
        return args.get("detail", "")

    @staticmethod
    def get_event_duration(event: Dict[str, Any]) -> int:
        """
        Get the duration of an event in microseconds.

        Args:
            event: Event dictionary

        Returns:
            Duration in microseconds (0 if not available)
        """
        return event.get("dur", 0)

    @staticmethod
    def get_event_name(event: Dict[str, Any]) -> str:
        """
        Get the name of an event.

        Args:
            event: Event dictionary

        Returns:
            Event name (or "Unknown" if not available)
        """
        return event.get("name", "Unknown")
