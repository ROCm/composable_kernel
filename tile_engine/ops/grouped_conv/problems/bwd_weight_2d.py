#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D bwd_weight grouped convolution problem set.

Re-exports the 2D subset of bwd_weight_synthetic_extended (Di == Z == 1).
"""

from bwd_weight_synthetic_extended import TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC

PROBLEMS_BWD_WEIGHT_2D = [
    p for p in TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC
    if getattr(p, "Di", 1) == 1 and getattr(p, "Z", 1) == 1
]


if __name__ == "__main__":
    print(f"bwd_weight 2D problems: {len(PROBLEMS_BWD_WEIGHT_2D)}")