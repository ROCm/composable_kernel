#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""3D bwd_weight grouped convolution problem set.

bwd_weight_synthetic_extended has no 3D shapes, so we reuse the 3D shape set
from bwd_data_synthetic_extended and rebind direction="bwd_weight" — the
underlying conv geometry is identical across variants.
"""

from dataclasses import replace

from bwd_data_synthetic_extended import TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC

PROBLEMS_BWD_WEIGHT_3D = [
    replace(p, direction="bwd_weight")
    for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC
    if getattr(p, "Di", 1) > 1 or getattr(p, "Z", 1) > 1
]


if __name__ == "__main__":
    print(f"bwd_weight 3D problems: {len(PROBLEMS_BWD_WEIGHT_3D)}")