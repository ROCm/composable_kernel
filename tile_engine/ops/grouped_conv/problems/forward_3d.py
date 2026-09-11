#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""3D forward grouped convolution problem set.

Re-exports the 3D subset of forward_synthetic_extended (Di > 1 or Z > 1).
"""

from forward_synthetic_extended import TRAINING_PROBLEMS_FORWARD_SYNTHETIC

PROBLEMS_FORWARD_3D = [
    p for p in TRAINING_PROBLEMS_FORWARD_SYNTHETIC
    if getattr(p, "Di", 1) > 1 or getattr(p, "Z", 1) > 1
]


if __name__ == "__main__":
    print(f"forward 3D problems: {len(PROBLEMS_FORWARD_3D)}")