#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D forward grouped convolution problem set.

Re-exports the 2D subset of forward_synthetic_extended (Di == Z == 1).
"""

from forward_synthetic_extended import TRAINING_PROBLEMS_FORWARD_SYNTHETIC

PROBLEMS_FORWARD_2D = [
    p for p in TRAINING_PROBLEMS_FORWARD_SYNTHETIC
    if getattr(p, "Di", 1) == 1 and getattr(p, "Z", 1) == 1
]


if __name__ == "__main__":
    print(f"forward 2D problems: {len(PROBLEMS_FORWARD_2D)}")