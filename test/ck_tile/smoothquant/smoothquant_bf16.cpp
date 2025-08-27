// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "smoothquant.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = create_test_cases("bf16");

    return !run_test_cases<ck_tile::bf16_t>(test_cases);
}
