// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType = F16;
using BDataType = F16;

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermute
// ######|    InData|   OutData| Elementwise| NumDim|  NPer|  HPer|  WPer|MPerThread|  InScalar| OutScalar|
// ######|      Type|      Type|   Operation|       | Block| Block| Block|          | PerVector| PerVector|
// ######|          |          |            |       |      |      |      |          |          |          |
// ######|          |          |            |       |      |      |      |          |          |          |
         < ADataType, BDataType, PassThrough,      4,   128,   128,   128,         8,         8,         1>;
// clang-format on

#include "run_permute_example.inc"

int main(int argc, char* argv[]) { return !run_permute_example(argc, argv); }
