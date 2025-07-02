// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "smoothquant.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = create_test_cases("fp16");

    return !run_test_cases<ck_tile::half_t>(test_cases);
}
