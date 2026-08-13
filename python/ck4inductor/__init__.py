# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
import os
import subprocess
from typing import TYPE_CHECKING

from .util import check_headers, include_roots


__all__ = ["check_headers", "include_roots", "__version__"]

if TYPE_CHECKING:
    # Bound at runtime by __getattr__ below; declared here so type checkers can
    # see the name that __all__ exports.
    __version__: str


# Needs to be manually updated at each release; it cannot be derived here. Release
# tags (therock-*) exist only in the monorepo, but the wheel is built from the
# filtered mirror, whose reachable tags stop at therock-7.10.
BASE_VERSION = "7.14"

_HASH_WIDTH = 7


def _build_hash():
    """Short commit hash of this source tree, or "" if it cannot be determined.

    Queried against the directory holding this file rather than the working
    directory, so the answer does not depend on where the caller happens to be
    standing. Note git still walks upwards from there, so an installed copy
    nested inside an unrelated repository reports that repository's commit.
    """
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                f"--short={_HASH_WIDTH}",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # No git, no .git (source tarball), or the call failed - all legitimate.
        return ""


@functools.lru_cache(None)
def _build_version():
    build_hash = _build_hash()
    # "+unknown" keeps the label honest when the hash is unavailable, rather than
    # emitting a plausible-looking one that would hide the failure.
    return f"{BASE_VERSION}+g{build_hash}" if build_hash else f"{BASE_VERSION}+unknown"


def __getattr__(name):
    # Resolve __version__ on first access rather than at import: it forks git, and
    # importing this package must not pay for a value most callers never read.
    # setuptools' `attr:` directive (pyproject.toml) triggers this at build time.
    if name == "__version__":
        return _build_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
