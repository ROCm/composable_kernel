// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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

int main() { return !run_test_cases("bf16"); }
