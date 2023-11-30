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
static constexpr auto I2 = ck::Number<2>{};

using DataType = int;

template <typename Desc>
void Print1d(const Desc& desc)
{
    std::cout << "Print1d" << std::endl;
    for(ck::index_t w = 0; w < desc.GetLength(I0); w++)
    {
        std::cout << desc.CalculateOffset(ck::make_tuple(w)) << " ";
    }
    std::cout << std::endl;
}

template <typename Desc>
void Print2d(const Desc& desc)
{
    std::cout << "Print2d" << std::endl;
    for(ck::index_t h = 0; h < desc.GetLength(I0); h++)
    {
        for(ck::index_t w = 0; w < desc.GetLength(I1); w++)
        {
            std::cout << desc.CalculateOffset(ck::make_tuple(h, w)) << " ";
        }
        std::cout << std::endl;
    }
}

template <typename Desc>
void Print3dCustom(const Desc& desc)
{
    std::cout << "Print3dCustom" << std::endl;
    for(ck::index_t d = 0; d < desc.GetLength(I0); d++)
    {
        for(ck::index_t h = 0; h < desc.GetLength(I1); h++)
        {
            for(ck::index_t w = 0; w < desc.GetLength(I2); w++)
            {
                std::cout << desc.CalculateOffset(ck::make_tuple(d, h, w)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main()
{
    // Tensor descriptor traverse in row-major (need to reverse dims)
    std::cout << "Note: Tensor descriptor traverse in row-major" << std::endl;
    // Basic descriptor 0, 1, 2, ... 30, 31
    // (dims:4,8 strides:1,4)
    const auto desc_4x8_s1x4 =
        ck::make_naive_tensor_descriptor(ck::make_tuple(ck::Number<4>{}, ck::Number<8>{}),
                                         ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}));
    std::cout << "dims:4,8 strides:1,4" << std::endl;
    Print2d(desc_4x8_s1x4);

    using Cord1x1Type                = ck::Tuple<ck::Number<1>, ck::Number<1>>;
    constexpr ck::index_t offset_1x1 = desc_4x8_s1x4.CalculateOffset(Cord1x1Type{});
    std::cout << "Constexpr calculated [1, 1] offset:" << offset_1x1 << std::endl;

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:4,(2,4) strides:2,(1,8)
    const auto desc_4x2x4_s2x1x8 =
        ck::make_naive_tensor_descriptor(ck::make_tuple(4, 2, 4), ck::make_tuple(2, 1, 8));
    // Transform to 2d (column-major, need to to reverse dims)
    const auto desc_4x2x4_s2x1x8_merged = ck::transform_tensor_descriptor(
        desc_4x2x4_s2x1x8,
        ck::make_tuple(ck::make_pass_through_transform(4),
                       ck::make_merge_transform(ck::make_tuple(4, 2))),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<2, 1>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

    std::cout << "dims:4,(2,4) strides:2,(1,8)" << std::endl;
    Print2d(desc_4x2x4_s2x1x8_merged);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:(2,2),(2,4) strides:((1,4),(2,8)
    const auto desc_2x2x2x4_s1x4x2x8 =
        ck::make_naive_tensor_descriptor(ck::make_tuple(2, 2, 2, 4), ck::make_tuple(1, 4, 2, 8));
    // Transform to 2d
    const auto desc_2x2x2x4_s1x4x2x8_double_merged_2d = ck::transform_tensor_descriptor(
        desc_2x2x2x4_s1x4x2x8,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(2, 2)),
                       ck::make_merge_transform(ck::make_tuple(4, 2))),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<3, 2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    // Transform to 3d
    const auto desc_2x2x2x4_s1x4x2x8_double_merged_3d = ck::transform_tensor_descriptor(
        desc_2x2x2x4_s1x4x2x8,
        ck::make_tuple(ck::make_pass_through_transform(2),
                       ck::make_pass_through_transform(2),
                       ck::make_merge_transform(ck::make_tuple(4, 2))),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<3, 2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}));

    std::cout << "dims:(2,2),(2,4) strides:(1,4),(2,8)" << std::endl;
    Print2d(desc_2x2x2x4_s1x4x2x8_double_merged_2d);
    Print3dCustom(desc_2x2x2x4_s1x4x2x8_double_merged_3d);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:((2,2),2),4 strides:((1,4),2),8
    // Transform to 2d
    const auto desc_2x2x2x4_s1x4x2x8_nested =
        ck::make_naive_tensor_descriptor(ck::make_tuple(2, 2, 2, 4), ck::make_tuple(1, 4, 2, 8));
    const auto desc_2x2x2x4_s1x4x2x8_nested_merged_3d = ck::transform_tensor_descriptor(
        desc_2x2x2x4_s1x4x2x8_nested,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(2, 2)),
                       ck::make_pass_through_transform(2),
                       ck::make_pass_through_transform(4)),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<2>{}, ck::Sequence<3>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}));
    const auto desc_2x2x2x4_s1x4x2x8_nested_merged_1d = ck::transform_tensor_descriptor(
        desc_2x2x2x4_s1x4x2x8_nested,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(4, 2, 2, 2))),
        ck::make_tuple(ck::Sequence<3, 2, 1, 0>{}),
        ck::make_tuple(ck::Sequence<0>{}));
    const auto desc_2x2x2x4_s1x4x2x8_nested_merged_2d = ck::transform_tensor_descriptor(
        desc_2x2x2x4_s1x4x2x8_nested_merged_3d,
        ck::make_tuple(ck::make_merge_transform(ck::make_tuple(2, 4)),
                       ck::make_pass_through_transform(4)),
        ck::make_tuple(ck::Sequence<1, 0>{}, ck::Sequence<2>{}),
        ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

    std::cout << "dims:((2,2),2),4 strides:((1,4),2),8" << std::endl;
    Print1d(desc_2x2x2x4_s1x4x2x8_nested_merged_1d);
    Print2d(desc_2x2x2x4_s1x4x2x8_nested_merged_2d);
    Print3dCustom(desc_2x2x2x4_s1x4x2x8_nested_merged_3d);

    return 0;
}
