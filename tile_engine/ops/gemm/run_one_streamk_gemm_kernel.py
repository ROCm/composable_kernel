#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compatibility entrypoint for callers of the legacy Stream-K worker.

Stream-K shares GEMM's C ABI and worker JSON protocol, including per-item
launch environments. All execution is delegated to ``run_one_gemm_kernel``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from run_one_gemm_kernel import main  # noqa: E402

if __name__ == "__main__":
    main()
