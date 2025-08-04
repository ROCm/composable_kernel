// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using ck::bhalf_t;
using ck::type_convert;

template <typename Tin, typename Tout>
__global__ void PassThroughPack2_kernel(Tout* output, const Tin* input)
{
    ck::tensor_operation::element_wise::PassThroughPack2{}(*output, *input);
}

template <typename Tin, typename Tout>
void test_pass_through_pack2(Tin input)
{
    Tin input_host = input;
    Tin* input_device;
    hip_check_error(hipMalloc(&input_device, sizeof(Tin)));
    hip_check_error(hipMemcpy(input_device, &input_host, sizeof(Tin), hipMemcpyHostToDevice));

    Tout output_host;
    Tout* output_device;
    hip_check_error(hipMalloc(&output_device, sizeof(Tout)));

    PassThroughPack2_kernel<<<1, 1>>>(output_device, input_device);
    hip_check_error(hipGetLastError());

    hip_check_error(hipMemcpy(&output_host, output_device, sizeof(Tout), hipMemcpyDeviceToHost));

    const float expected_output1 = type_convert<float>(input_host[0]);
    const float expected_output2 = type_convert<float>(input_host[1]);

    const float actual_output1 = type_convert<float>(output_host[0]);
    const float actual_output2 = type_convert<float>(output_host[1]);

    EXPECT_EQ(actual_output1, expected_output1);
    EXPECT_EQ(actual_output2, expected_output2);

    hip_check_error(hipFree(input_device));
    hip_check_error(hipFree(output_device));
}

TEST(PassThrough, Pack_float2_to_bhalf2)
{
    test_pass_through_pack2<ck::float2_t, ck::bhalf2_t>(ck::float2_t{1.0f, 2.0f});
    test_pass_through_pack2<ck::float2_t, ck::bhalf2_t>(ck::float2_t{-0.125f, 7.0f});
}
