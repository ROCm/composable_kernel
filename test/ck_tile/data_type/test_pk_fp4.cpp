// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"

// using ck_tile::bf16x2_t;
// using ck_tile::fp16x2_t;
using ck_tile::fp32x2_t;
// using ck_tile::half_t;
using ck_tile::pk_fp4_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
using ck_tile::number;
TEST(PackedFp4, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::has_inf(), false);
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::zero(), pk_fp4_t{0b00000000});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::min(), pk_fp4_t{0b00100010});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::max(), pk_fp4_t{0b01110111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::lowest(), pk_fp4_t{0b11111111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::epsilon(), pk_fp4_t{0b00010001}); 
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::round_error(), pk_fp4_t{0b00010001});
}

TEST(PackedFp4, ConvertFP16Nearest)
{
    auto test = [](float input_0, float input_1, float ref_0, float ref_1) {
        using ck_tile::type_convert;

        fp16_t ref_fp16_0 = type_convert<fp16_t>(ref_0);
        EXPECT_EQ(type_convert<float>(ref_fp16_0), ref_0);

        fp16_t ref_fp16_1 = type_convert<fp16_t>(ref_1);
        EXPECT_EQ(type_convert<float>(ref_fp16_1), ref_1);

        const auto input_fp32x2 = fp32x2_t{input_0, input_1};
        EXPECT_EQ(input_fp32x2[0], input_0);
        EXPECT_EQ(input_fp32x2[1], input_1);

        // fp32x2 -> pk_fp4 -> fp32
        const auto output_pk_fp4 = type_convert<pk_fp4_t>(input_fp32x2);
        EXPECT_EQ(type_convert<float>(output_pk_fp4.unpack(number<0>{})), ref_0) << "Debug~~~";
        EXPECT_EQ(type_convert<float>(output_pk_fp4.unpack(number<1>{})), ref_1);

        // fp32x2 -> pk_fp4 -> fp16x2
        const auto output_fp16x2 = type_convert<fp16x2_t>(output_pk_fp4);
        EXPECT_EQ(output_fp16x2[0], ref_fp16_0);
        EXPECT_EQ(output_fp16x2[1], ref_fp16_1);

        // fp32x2 -> pk_fp4 -> fp32x2
        const auto output_fp32x2 = type_convert<fp32x2_t>(output_pk_fp4);
        EXPECT_EQ(output_fp32x2[0], ref_0);
        EXPECT_EQ(output_fp32x2[1], ref_1);
        // TODO: add test for fp16 -> fp4
    };

    const std::vector<std::pair<float, float>> test_data {
        {0, 0}, {0.25, 0}, {0.5, 0.5}, {0.75, 1}, 
        {1, 1}, {1.25, 1}, {1.5, 1.5}, {1.75, 2}, 
        {2, 2}, {2.5, 2}, {3, 3}, {3.5, 4},
        {4, 4}, {5, 4}, {5.0001, 6}, {6, 6}
    };
    assert(test_data.size() % 2 == 0);

    for(auto it = test_data.begin(); it != test_data.end(); ) {
        auto input_0 = it -> first;
        auto ref_0 = (it++) -> second;
        auto input_1 = it -> first;
        auto ref_1 = (it++) -> second;

        test(input_0, input_1, ref_0, ref_1);
        test(-input_0, -input_1, -ref_0, -ref_1);
    }

}
