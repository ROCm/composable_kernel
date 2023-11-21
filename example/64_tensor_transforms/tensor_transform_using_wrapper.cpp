// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/sequence.hpp"

#include "tensor_transform_wrapper.hpp"

using DataType = int;

template <typename Layout>
void Print(const Layout& layout)
{
    for(ck::index_t h = 0; h < ck::tensor_transform_wrapper::size<0>(layout); h++)
    {
        for(ck::index_t w = 0; w < ck::tensor_transform_wrapper::size<1>(layout); w++)
        {
            std::cout << layout(ck::make_tuple(h, w)) << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    // Basic descriptor 0, 1, 2, ... 30, 31 (runtime descriptor)
    // (dims:4,8 strides:1,1)
    const auto shape_4x8       = ck::make_tuple(4, 8);
    const auto layout_4x8_s1x1 = ck::tensor_transform_wrapper::make_layout(shape_4x8);
    std::cout << "dims:4,8 strides:1,1" << std::endl;
    Print(layout_4x8_s1x1);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:4,(4,2) strides:2,(8,1)
    const auto shape_4x4x2 =
        ck::make_tuple(ck::Number<4>{}, ck::make_tuple(ck::Number<4>{}, ck::Number<2>{}));
    const auto strides_s2x8x1 =
        ck::make_tuple(ck::Number<2>{}, ck::make_tuple(ck::Number<8>{}, ck::Number<1>{}));
    const auto layout_4x4x2_s2x8x1 =
        ck::tensor_transform_wrapper::make_layout(shape_4x4x2, strides_s2x8x1);

    std::cout << "dims:4,(4,2) strides:2,(8,1)" << std::endl;
    Print(layout_4x4x2_s2x8x1);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:(2,2),(4,2) strides:((4,1),(8,2)
    const auto shape_2x2x4x2    = ck::make_tuple(ck::make_tuple(ck::Number<2>{}, ck::Number<2>{}),
                                              ck::make_tuple(ck::Number<4>{}, ck::Number<2>{}));
    const auto strides_s4x1x8x2 = ck::make_tuple(ck::make_tuple(ck::Number<4>{}, ck::Number<1>{}),
                                                 ck::make_tuple(ck::Number<8>{}, ck::Number<2>{}));
    static const auto layout_2x2x4x2_s4x1x8x2 =
        ck::tensor_transform_wrapper::make_layout(shape_2x2x4x2, strides_s4x1x8x2);

    std::cout << "dims:(2,2),(4,2) strides:(4,1),(8,2)" << std::endl;
    Print(layout_2x2x4x2_s4x1x8x2);

    // Basic descriptor 0, 1, 8, 9, 16, 17, ... 30, 31 (compile-time descriptor)
    // dims:((2,2),4),2 strides:((4,1),8),2
    // Transform to 2d
    const auto shape_2x2x4x2_nested = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<2>{}, ck::Number<2>{}), ck::Number<4>{}),
        ck::Number<2>{});
    const auto strides_s4x1x8x2_nested = ck::make_tuple(
        ck::make_tuple(ck::make_tuple(ck::Number<4>{}, ck::Number<1>{}), ck::Number<8>{}),
        ck::Number<2>{});
    static const auto layout_2x2x4x2_s4x1x8x2_nested =
        ck::tensor_transform_wrapper::make_layout(shape_2x2x4x2_nested, strides_s4x1x8x2_nested);

    std::cout << "dims:((2,2),4),2 strides:((4,1),8),2" << std::endl;
    Print(layout_2x2x4x2_s4x1x8x2_nested);

    return 0;
}
