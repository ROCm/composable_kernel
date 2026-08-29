#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D bwd_data grouped convolution problem set.

Re-exports the 2D subset of bwd_data_synthetic_extended (Di == Z == 1).
"""

from bwd_data_synthetic_extended import TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC

PROBLEMS_BWD_DATA_2D = [
    p for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC
    if getattr(p, "Di", 1) == 1 and getattr(p, "Z", 1) == 1
]


if __name__ == "__main__":
    print(f"bwd_data 2D problems: {len(PROBLEMS_BWD_DATA_2D)}")