// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"

using ck_tile::fp32x2_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
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
TEST(PackedFp4, ConvertF32Basic)
{
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(0.0f), pk_fp4_t{0b00000000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-0.0f), pk_fp4_t{0b00001000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-1.0f), pk_fp4_t{0b00001010}.get());
}


#define toF16(x) ck_tile::type_convert<fp16_t>(x)
#define toB16(x) ck_tile::type_convert<bf16_t>(x)
#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF4(x) ck_tile::type_convert<pk_fp4_t>(x)
#define toDST(x) ck_tile::type_convert<DST>(x)
#define toDSTx2(x) ck_tile::type_convert<DSTx2_t>(x)
TEST(PackedFp4, ConvertFP16Nearest)
{
    using ck_tile::type_convert;
    auto test = [](auto input_0, auto input_1, auto ref_0, auto ref_1) {

        using SRC = decltype(input_0);
        using DST = decltype(ref_0);
        using SRCx2_t = ck_tile::ext_vector_t<SRC, 2>;
        using DSTx2_t = ck_tile::ext_vector_t<DST, 2>;

        // ex: fp32 x 2 -> fp32x2_t
        const auto inputx2 = SRCx2_t{input_0, input_1};
        EXPECT_EQ(inputx2[0], input_0);
        EXPECT_EQ(inputx2[1], input_1);

        // ex: fp32x2 -> pk_fp4 -> unpack(0) -> bf16
        const auto output_pk_fp4 = toPF4(inputx2);
        const auto output_0 = toDST(toPF4(output_pk_fp4.unpack(number<0>{})));
        const auto output_1 = toDST(toPF4(output_pk_fp4.unpack(number<1>{})));
        EXPECT_EQ(output_0, ref_0);
 //   << "input:" << toF32(input_0) << ", output:" << toF32(output_0) << ", answer:" << toF32(ref_0) << "\n";
        EXPECT_EQ(output_1, ref_1);

        // ex: fp32x2 -> pk_fp4 -> fp16x2
        const auto outputx2 = toDSTx2(toPF4(inputx2));
        EXPECT_EQ(outputx2[0], ref_0);
        EXPECT_EQ(outputx2[1], ref_1);
    };

    const std::vector<std::pair<float, float>> test_data {
        {0, 0}, {0.25, 0}, {0.5, 0.5}, {0.75, 1}, 
        {1, 1}, {1.25, 1}, {1.5, 1.5}, {1.75, 2}, 
        {2, 2}, {2.5, 2}, {3, 3}, {3.5, 4},
        {4, 4}, {5, 4}, {5.0625, 6}, {6, 6}
    };
    assert(test_data.size() % 2 == 0);

    for(auto it = test_data.begin(); it != test_data.end(); ) {
        auto input_0 = it -> first;
        auto ref_0 = (it++) -> second;
        auto input_1 = it -> first;
        auto ref_1 = (it++) -> second;

        // convert without pk_fp4
        EXPECT_EQ(input_0, toF32(toF16(input_0)));
        EXPECT_EQ(input_0, toF32(toB16(input_0)));
        EXPECT_EQ(-input_0, toF32(toF16(-input_0)));
        EXPECT_EQ(-input_0, toF32(toB16(-input_0)));
        EXPECT_EQ(input_1, toF32(toF16(input_1)));
        EXPECT_EQ(input_1, toF32(toB16(input_1)));
        EXPECT_EQ(-input_1, toF32(toF16(-input_1)));
        EXPECT_EQ(-input_1, toF32(toB16(-input_1))); 
        EXPECT_EQ(ref_0, toF32(toF16(ref_0)));
        EXPECT_EQ(ref_0, toF32(toB16(ref_0)));
        EXPECT_EQ(-ref_0, toF32(toF16(-ref_0)));
        EXPECT_EQ(-ref_0, toF32(toB16(-ref_0)));
        EXPECT_EQ(ref_1, toF32(toF16(ref_1)));
        EXPECT_EQ(ref_1, toF32(toB16(ref_1)));
        EXPECT_EQ(-ref_1, toF32(toF16(-ref_1)));
        EXPECT_EQ(-ref_1, toF32(toB16(-ref_1)));

        // fp32x2 -> pk_fp4 -> fp32x2
        test(input_0, input_1, ref_0, ref_1);
        test(-input_0, -input_1, -ref_0, -ref_1);

        // fp16x2 -> pk_fp4 -> fp16x2
        test(toF16(input_0), toF16(input_1), toF16(ref_0), toF16(ref_1));
        test(toF16(-input_0), toF16(-input_1), toF16(-ref_0), toF16(-ref_1));

        // bf16x2 -> pk_fp4 -> bf16x2
        test(toB16(input_0), toB16(input_1), toB16(ref_0), toB16(ref_1));
        test(toB16(-input_0), toB16(-input_1), toB16(-ref_0), toB16(-ref_1));

        // fp32x2 -> pk_fp4 -> fp16x2
        test(input_0, input_1, toF16(ref_0), toF16(ref_1));
        test(-input_0, -input_1, toF16(-ref_0), toF16(-ref_1));

        // fp32x2 -> pk_fp4 -> bf16x2
        test(input_0, input_1, toB16(ref_0), toB16(ref_1));
        test(-input_0, -input_1, toB16(-ref_0), toB16(-ref_1));

        // fp16x2 -> pk_fp4 -> fp32x2
        test(toF16(input_0), toF16(input_1), ref_0, ref_1);
        test(toF16(-input_0), toF16(-input_1), -ref_0, -ref_1);

        // bf16x2 -> pk_fp4 -> fp32x2
        test(toB16(input_0), toB16(input_1), ref_0, ref_1);
        test(toB16(-input_0), toB16(-input_1), -ref_0, -ref_1);
    }

}
