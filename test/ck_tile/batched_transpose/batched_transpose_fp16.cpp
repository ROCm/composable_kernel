// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "batched_transpose.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = generate_test_cases("fp16");

    return !run_test_cases<ck_tile::fp16_t>(test_cases);
}
