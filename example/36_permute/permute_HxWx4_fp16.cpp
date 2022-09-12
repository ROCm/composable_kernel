// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType = F64;
using BDataType = F64;

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermute
// ######|    InData|   OutData| Elementwise| NumDim| Block|  HPer|  WPer|   InBlock|      InBlockTransfer|           InBlockTransfer|       Src|       Dst|             Src|             Dst|
// ######|      Type|      Type|   Operation|       |  Size| Block| Block| LdsExtraW| ThreadClusterLengths| ThreadClusterArrangeOrder| VectorDim| VectorDim| ScalarPerVector| ScalarPerVector|
// ######|          |          |            |       |      |      |      |          |                     |                          |          |          |                |                |
// ######|          |          |            |       |      |      |      |          |                     |                          |          |          |                |                |
         < ADataType, BDataType, PassThrough,      3,   256,   128,   128,         0,         S<1, 16, 16>,                S<0, 1, 2>,         2,         1,               1,               1>;
// clang-format on

#define NUM_ELEMS_IN_BUNDLE 4
#include "run_permute_example.inc"

int main(int argc, char* argv[]) { return !run_permute_example(argc, argv, {1, 6, 5}, {0, 2, 1}); }
