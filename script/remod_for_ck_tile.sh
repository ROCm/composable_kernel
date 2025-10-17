#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run remod.py in both required locations
(cd include/ck_tile/ && python3 remod.py)
(cd example/ck_tile/ && python3 remod.py)
