// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using F16 = ck::half_t;

using ADataType = F16;
using BDataType = F16;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ADataType>,
                                                    ck::Tuple<BDataType>,
                                                    PassThrough,
                                                    4,
                                                    8,
                                                    ck::Sequence<8>,
                                                    ck::Sequence<1>>;

#include "run_elementwise_permute_example.inc"

int main(int argc, char* argv[]) { return !run_elementwise_permute_example(argc, argv); }
