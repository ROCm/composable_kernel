#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ctest driver for unittest.TestCase suites that skip themselves.

``python -m unittest discover`` exits 0 when every test skipped, so ctest
records Passed for a run that executed nothing. On a CPU-only runner -- or a
gfx942 box for a gfx950-only suite -- that is a green tick backed by zero
assertions, which is exactly the signal SKIP_RETURN_CODE exists to remove.

This runs the same discovery and maps the outcome onto the convention the
script-style tests here already use:

    0   at least one test actually ran, and everything passed
    77  every collected test skipped  -> ctest "Skipped"
    1   a test failed or errored, or discovery collected nothing at all

Discovery collecting nothing is a failure rather than a skip on purpose: a
renamed file or a typo'd pattern otherwise reports Skipped forever and nobody
notices the suite left the build.

Used instead of converting the suites to the script-style-77 form so the
skip-vs-run decision stays where it belongs -- in each test's own setUp.
"""

import sys
import unittest
from pathlib import Path

_SKIP = 77


def main(argv) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <test_file_pattern.py>", file=sys.stderr)
        return 2

    pattern = argv[1]
    start_dir = str(Path(__file__).resolve().parent)

    suite = unittest.defaultTestLoader.discover(start_dir, pattern=pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        return 1

    if result.testsRun == 0:
        print(
            f"ERROR: pattern {pattern!r} collected no tests under {start_dir}. "
            "The file was probably renamed or removed -- reporting failure "
            "rather than Skipped so this does not go unnoticed.",
            file=sys.stderr,
        )
        return 1

    ran = result.testsRun - len(result.skipped)
    if ran == 0:
        reasons = sorted({reason for _, reason in result.skipped})
        print(
            f"SKIP: all {result.testsRun} test(s) in {pattern} skipped "
            f"({'; '.join(reasons)})"
        )
        return _SKIP

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
