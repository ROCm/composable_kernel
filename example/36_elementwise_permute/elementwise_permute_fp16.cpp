// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using ADataType = F16;
using BDataType = F16;

using DevicePermuteInstance = ck::tensor_operation::device::
    DevicePermute<ck::Tuple<ADataType>, ck::Tuple<BDataType>, PassThrough, 4, 8, S<8>, S<1>>;

#include "run_elementwise_permute_example.inc"

int main(int argc, char* argv[]) { return !run_permute_example(argc, argv); }
