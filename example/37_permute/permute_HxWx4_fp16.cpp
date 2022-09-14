// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType = F64;
using BDataType = F64;

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermute
// ######|    InData|   OutData| Elementwise| NumDim| Block|  NPer|  HPer|  WPer|   InBlock|      InBlockTransfer|           InBlockTransfer|       Src|       Dst|             Src|             Dst|
// ######|      Type|      Type|   Operation|       |  Size| Block| Block| Block| LdsExtraW| ThreadClusterLengths| ThreadClusterArrangeOrder| VectorDim| VectorDim| ScalarPerVector| ScalarPerVector|
// ######|          |          |            |       |      |      |      |      |          |                     |                          |          |          |                |                |
// ######|          |          |            |       |      |      |      |      |          |                     |                          |          |          |                |                |
         < ADataType, BDataType, PassThrough,      3,   256,     1,    32,    32,         5,         S<1, 32,  8>,                S<0, 1, 2>,         2,         1,               4,               1>;
// clang-format on

#define NUM_ELEMS_IN_BUNDLE 4
static_assert(std::is_same_v<detail::get_bundled_t<F64, NUM_ELEMS_IN_BUNDLE>, F16>);

#include "run_permute_bundle_example.inc"

int main() { return !run_permute_bundle_example({1, 80, 16000}, {0, 2, 1}); }
