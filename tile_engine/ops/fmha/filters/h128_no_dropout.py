# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sample filter: only h128 kernels without dropout.

Usage:
    python fmha_benchmark.py configs/receipt0_fwd.json --filter-file filters/h128_no_dropout.py
    python fmha_instance_builder.py configs/receipt0_fwd.json --filter-file filters/h128_no_dropout.py --count-only
"""


def filter_config(c) -> bool:
    """Keep only h128 kernels without dropout."""
    return c.hdim_q == 128 and not c.dropout
