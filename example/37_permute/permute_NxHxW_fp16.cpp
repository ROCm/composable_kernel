// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType = F16;
using BDataType = F16;

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermute
// ######|    InData|   OutData| Elementwise| NumDim| Block|  NPer|  HPer|  WPer|   InBlock|      InBlockTransfer|           InBlockTransfer|       Src|       Dst|             Src|             Dst|
// ######|      Type|      Type|   Operation|       |  Size| Block| Block| Block| LdsExtraW| ThreadClusterLengths| ThreadClusterArrangeOrder| VectorDim| VectorDim| ScalarPerVector| ScalarPerVector|
// ######|          |          |            |       |      |      |      |      |          |                     |                          |          |          |                |                |
// ######|          |          |            |       |      |      |      |      |          |                     |                          |          |          |                |                |
         < ADataType, BDataType, PassThrough,      3,   128,     4,    16,     8,         6,          S<2, 16, 4>,                S<0, 1, 2>,         2,         1,               2,               1>;
// clang-format on

#include "run_permute_element_example.inc"

int main() { return !run_permute_element_example({121, 768, 80}, {0, 2, 1}); }
