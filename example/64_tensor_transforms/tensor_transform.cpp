// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/sequence.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

static constexpr auto I0 = ck::Number<0>{};
static constexpr auto I1 = ck::Number<1>{};

using DataType = int;

template <typename Desc>
void Print(const Desc& desc)
{
    for(ck::index_t h = 0; h < desc.GetLength(I0); h++)
    {
        for(ck::index_t w = 0; w < desc.GetLength(I1); w++)
        {
            std::cout << desc.CalculateOffset(ck::make_tuple(h, w)) << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    // Basic descriptor 0, 1, 2, ... 30, 31
    // (dims:4,8 strides:1,1)
    const auto desc_4x8_s1x1 = ck::make_naive_tensor_descriptor_packed(ck::make_tuple(4, 8));
    std::cout << "dims:4,8 strides:1,1" << std::endl;
    Print(desc_4x8_s1x1);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31
    // dims:4,(4,2) strides:2,(8,1)
    const auto desc_4x4x2_s2x8x1 =
        ck::make_naive_tensor_descriptor(ck::make_tuple(4, 4, 2), ck::make_tuple(2, 8, 1));
    // Transform to 2d
    const auto desc_4x4x2_s2x8x1_merged = ck::transform_tensor_descriptor(
        desc_4x4x2_s2x8x1,
        ck::make_tuple(ck::make_pass_through_transform(4),
                       ck::make_merge_transform(ck::make_tuple(4, 2))),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1, 2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

    std::cout << "dims:4,(4,2) strides:2,(8,1)" << std::endl;
    Print(desc_4x4x2_s2x8x1_merged);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31
    // dims:(2,2),(4,2) strides:(4,1),(8,2)
    const auto desc_2x2x4x2_s4x1x8x2 =
        ck::make_naive_tensor_descriptor(ck::make_tuple(2, 2, 4, 2), ck::make_tuple(4, 1, 8, 2));
    // Transform to 2d
    const auto desc_2x2x4x2_s4x1x8x2_double_merged = ck::transform_tensor_descriptor(
        desc_2x2x4x2_s4x1x8x2,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(2, 2)),
                       ck::make_merge_transform(ck::make_tuple(4, 2))),
        ck::make_tuple(ck::Sequence<0, 1>{}, ck::Sequence<2, 3>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    std::cout << "dims:(2,2),(4,2) strides:(4,1),(8,2)" << std::endl;
    Print(desc_2x2x4x2_s4x1x8x2_double_merged);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31
    // dims:((2,2),4),2 strides:((4,1),8),2
    // Transform to 2d
    const auto desc_2x2x4x2_s4x1x8x2_merged = ck::transform_tensor_descriptor(
        desc_2x2x4x2_s4x1x8x2,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(2, 2)),
                       ck::make_pass_through_transform(4),
                       ck::make_pass_through_transform(2)),
        ck::make_tuple(ck::Sequence<0, 1>{}, ck::Sequence<2>{}, ck::Sequence<3>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}));
    const auto desc_2x2x4x2_s4x1x8x2_nested_merged = ck::transform_tensor_descriptor(
        desc_2x2x4x2_s4x1x8x2_merged,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(4, 4)),
                       ck::make_pass_through_transform(2)),
        ck::make_tuple(ck::Sequence<0, 1>{}, ck::Sequence<2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    std::cout << "dims:((2,2),4),2 strides:((4,1),8),2" << std::endl;
    Print(desc_2x2x4x2_s4x1x8x2_nested_merged);

    return 0;
}
