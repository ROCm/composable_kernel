#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""3D bwd_data grouped convolution problem set.

Re-exports the 3D subset of bwd_data_synthetic_extended (Di > 1 or Z > 1).
"""

from bwd_data_synthetic_extended import TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC

PROBLEMS_BWD_DATA_3D = [
    p for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC
    if getattr(p, "Di", 1) > 1 or getattr(p, "Z", 1) > 1
]


if __name__ == "__main__":
    print(f"bwd_data 3D problems: {len(PROBLEMS_BWD_DATA_3D)}")