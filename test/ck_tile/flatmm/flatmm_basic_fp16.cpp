// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/host.hpp"
#include "flatmm_basic.hpp"
#include "run_flatmm_example.inc"

int main() { return !run_test_cases("fp16"); }
