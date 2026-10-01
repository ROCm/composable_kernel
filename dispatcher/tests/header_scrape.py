# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate code sections in CK headers for the host-compiled probe tests.

The probes compile the real header text of a function instead of a copy, so
they break when a header is refactored. A bare ``str.index`` then fails with
"substring not found" and no hint of which header or marker moved. These
helpers try the exact marker first, fall back to a whitespace-insensitive match
(clang-format re-indentation and line wrapping), and otherwise fail with the
header name and the marker that needs updating.

Comments are deliberately not stripped: some markers are comments themselves.
"""

import re
from pathlib import Path

import pytest


def read_header(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"probe header not found: {path}", pytrace=False)
    return path.read_text()


def _pattern(marker: str):
    body = r"\s+".join(re.escape(tok) for tok in marker.split())
    lead = r"[ \t]*" if marker[:1].isspace() else ""
    trail = r"\s*" if marker[-1:].isspace() else ""
    return re.compile(lead + body + trail)


def find(text: str, marker: str, start: int = 0, end=None, *, where: str) -> int:
    """Index of the first ``marker`` in ``text[start:end]``."""
    stop = len(text) if end is None else end
    pos = text.find(marker, start, stop)
    if pos >= 0:
        return pos
    match = _pattern(marker).search(text, start, stop)
    if match:
        return match.start()
    pytest.fail(
        f"{where}: marker {marker!r} not found (offset {start}..{stop}); "
        "the header was refactored, update the probe marker", pytrace=False)


def rfind(text: str, marker: str, end: int, *, where: str) -> int:
    """Index of the last ``marker`` that starts before ``end``."""
    pos = text.rfind(marker, 0, end)
    if pos >= 0:
        return pos
    matches = [m.start() for m in _pattern(marker).finditer(text, 0, end)]
    if matches:
        return matches[-1]
    pytest.fail(
        f"{where}: marker {marker!r} not found before offset {end}; "
        "the header was refactored, update the probe marker", pytrace=False)


def between(text: str, begin: str, end: str, *, where: str) -> str:
    """Text from ``begin`` up to, not including, the next ``end``."""
    start = find(text, begin, where=where)
    section = text[start:find(text, end, start + 1, where=where)]
    if not section.strip():
        pytest.fail(f"{where}: empty section between {begin!r} and {end!r}", pytrace=False)
    return section
