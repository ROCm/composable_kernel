// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "maxpool2d_bwd_common.hpp"

using InDataType      = float;
using OutDataType     = float;
using IndexDataType   = int32_t;
using ComputeDataType = float;
using DInDataType     = float;
using DOutDataType    = float;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using OutLayout = ck::tensor_layout::convolution::NHWC;

static constexpr bool PropagateNan = false;

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    // Pool shape
    constexpr ck::index_t N               = 1;
    constexpr ck::index_t C               = 1;
    constexpr ck::index_t Y               = 2;
    constexpr ck::index_t X               = 2;
    constexpr ck::index_t Hi              = 31;
    constexpr ck::index_t Wi              = 31;
    constexpr ck::index_t window_stride_h = 2;
    constexpr ck::index_t window_stride_w = 2;
    constexpr ck::index_t in_left_pad_h   = 0;
    constexpr ck::index_t in_left_pad_w   = 0;
    constexpr ck::index_t in_right_pad_h  = 1;
    constexpr ck::index_t in_right_pad_w  = 1;

    constexpr bool WindowOverlap                  = Y > window_stride_h || X > window_stride_w;
    constexpr ck::InMemoryDataOperationEnum MemOp = WindowOverlap
                                                        ? ck::InMemoryDataOperationEnum::AtomicAdd
                                                        : ck::InMemoryDataOperationEnum::Set;
    std::cout << "WindowOverlap = " << WindowOverlap << std::endl;

    bool pass = maxpool_bwd_test<InDataType,
                                 OutDataType,
                                 IndexDataType,
                                 ComputeDataType,
                                 DInDataType,
                                 DOutDataType,
                                 InLayout,
                                 OutLayout,
                                 ck::ReduceTensorOp::MAX,
                                 PropagateNan,
                                 MemOp>(do_verification,
                                                time_kernel,
                                                N,
                                                C,
                                                Y,
                                                X,
                                                Hi,
                                                Wi,
                                                window_stride_h,
                                                window_stride_w,
                                                in_left_pad_h,
                                                in_left_pad_w,
                                                in_right_pad_h,
                                                in_right_pad_w);

    return (pass ? 0 : 1);
}
