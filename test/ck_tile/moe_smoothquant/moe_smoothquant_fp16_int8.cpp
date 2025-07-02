// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_smoothquant.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = generate_test_cases("fp16", "int8");

    return !run_test_cases<ck_tile::half_t, ck_tile::int8_t>(test_cases);
}
