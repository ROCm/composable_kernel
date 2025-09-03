// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include "gtest/gtest.h"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using namespace ck;

static constexpr auto I0 = Number<0>{};
static constexpr auto I1 = Number<1>{};
static constexpr auto I2 = Number<2>{};
static constexpr auto I3 = Number<3>{};

TEST(vector_type, TestConstruction)
{
    using float_vector_1 = typename vector_type_maker<float, 1>::type;
    using float_vector_2 = typename vector_type_maker<float, 2>::type;
    using float_vector_4 = typename vector_type_maker<float, 4>::type;

    using bhalf_vector_1 = typename vector_type_maker<ck::bhalf_t, 1>::type;
    using bhalf_vector_2 = typename vector_type_maker<ck::bhalf_t, 2>::type;

    // Verify sizes
    EXPECT_EQ(sizeof(float_vector_1), sizeof(float) * 1);
    EXPECT_EQ(sizeof(float_vector_2), sizeof(float) * 2);
    EXPECT_EQ(sizeof(float_vector_4), sizeof(float) * 4);
    EXPECT_EQ(sizeof(bhalf_vector_1), sizeof(ck::bhalf_t) * 1);
    EXPECT_EQ(sizeof(bhalf_vector_2), sizeof(ck::bhalf_t) * 2);
}

TEST(vector_type, TestModifyingContents)
{
    typename vector_type_maker<ck::bhalf_t, 2>::type dst_vector;

    const float val_0 = 1.5f;
    const float val_1 = -2.25f;

    const ck::bhalf_t bhalf_0 = ck::type_convert<ck::bhalf_t>(val_0);
    const ck::bhalf_t bhalf_1 = ck::type_convert<ck::bhalf_t>(val_1);

    dst_vector.template AsType<ck::bhalf_t>()(I0) = bhalf_0;
    dst_vector.template AsType<ck::bhalf_t>()(I1) = bhalf_1;

    const ck::bhalf_t stored_0 = dst_vector.template AsType<ck::bhalf_t>()[I0];
    const ck::bhalf_t stored_1 = dst_vector.template AsType<ck::bhalf_t>()[I1];

    const float result_0 = ck::type_convert<float>(stored_0);
    const float result_1 = ck::type_convert<float>(stored_1);

    EXPECT_NEAR(val_0, result_0, 1e-3f);
    EXPECT_NEAR(val_1, result_1, 1e-3f);
}

TEST(vector_type, TestPointerToData)
{
    typename vector_type_maker<float, 2>::type float_vector;

    const float val_1 = 3.14f;
    const float val_2 = -1.618f;

    const float val_3 = 2.718f;
    const float val_4 = 0.577f;

    // Initialize the vector with some values
    float_vector.template AsType<float>()(I0) = val_1;
    float_vector.template AsType<float>()(I1) = val_2;

    // Get read-only pointer to data
    const float* data_ptr = &(float_vector.template AsType<float>()[I0]);

    // Verify the contents through pointer access
    EXPECT_NEAR(data_ptr[0], val_1, 1e-3f);
    EXPECT_NEAR(data_ptr[1], val_2, 1e-3f);

    // Get mutable pointer to data
    float* mutable_data_ptr = &(float_vector.template AsType<float>()(I0));
    mutable_data_ptr[0]     = val_3;
    mutable_data_ptr[1]     = val_4;

    // Verify the contents after modification
    EXPECT_NEAR(float_vector.template AsType<float>()[I0], val_3, 1e-3f);
    EXPECT_NEAR(float_vector.template AsType<float>()[I1], val_4, 1e-3f);
}

TEST(vector_type, TestVectorizedAccess_float)
{
    typename vector_type_maker<float, 4>::type float_vector;

    const float val_1     = 3.14f;
    const float val_2     = -1.618f;
    const float val_3     = 2.718f;
    const float val_4     = 0.577f;
    const float2_t val_12 = {val_1, val_2};
    const float2_t val_34 = {val_3, val_4};

    // Via reference
    float2_t& mutable_value = float_vector.template AsType<float2_t>()(I0);
    mutable_value           = val_12;

    // Directly via AsType
    float_vector.template AsType<float2_t>()(I1) = val_34;

    // Verify the contents after modification
    EXPECT_NEAR(float_vector.template AsType<float>()[I0], val_1, 1e-3f);
    EXPECT_NEAR(float_vector.template AsType<float>()[I1], val_2, 1e-3f);
    EXPECT_NEAR(float_vector.template AsType<float>()[I2], val_3, 1e-3f);
    EXPECT_NEAR(float_vector.template AsType<float>()[I3], val_4, 1e-3f);
}

__global__ void testVectorizedAccess_kernel(float* input, float* output)
{
    typename vector_type_maker<ck::bhalf_t, 4>::type vector;

    const ck::bhalf_t bf_val_1 = ck::type_convert<ck::bhalf_t>(input[0]);
    const ck::bhalf_t bf_val_2 = ck::type_convert<ck::bhalf_t>(input[1]);
    const ck::bhalf_t bf_val_3 = ck::type_convert<ck::bhalf_t>(input[2]);
    const ck::bhalf_t bf_val_4 = ck::type_convert<ck::bhalf_t>(input[3]);
    const ck::bhalf2_t val_12  = {bf_val_1, bf_val_2};
    const ck::bhalf2_t val_34  = {bf_val_3, bf_val_4};

    // Via reference
    ck::bhalf2_t& mutable_value = vector.template AsType<ck::bhalf2_t>()(I0);
    mutable_value               = val_12;

    // Directly via AsType
    vector.template AsType<ck::bhalf2_t>()(I1) = val_34;

    // Store the results back to output
    output[0] = ck::type_convert<float>(vector.template AsType<ck::bhalf_t>()[I0]);
    output[1] = ck::type_convert<float>(vector.template AsType<ck::bhalf_t>()[I1]);
    output[2] = ck::type_convert<float>(vector.template AsType<ck::bhalf_t>()[I2]);
    output[3] = ck::type_convert<float>(vector.template AsType<ck::bhalf_t>()[I3]);
}

TEST(vector_type, TestVectorizedAccess_bf16_device)
{
    std::vector<float> input_values = {3.14f, -1.618f, 2.718f, 0.577f};
    const float val_1               = input_values[0];
    const float val_2               = input_values[1];
    const float val_3               = input_values[2];
    const float val_4               = input_values[3];

    // Allocate device memory
    float* d_input;
    float* d_output;
    hip_check_error(hipMalloc(&d_input, input_values.size() * sizeof(float)));
    hip_check_error(hipMalloc(&d_output, input_values.size() * sizeof(float)));
    hip_check_error(hipMemcpy(
        d_input, input_values.data(), input_values.size() * sizeof(float), hipMemcpyHostToDevice));

    // Launch the kernel
    testVectorizedAccess_kernel<<<1, 1>>>(d_input, d_output);
    hip_check_error(hipDeviceSynchronize());

    // Copy the output back to host
    std::vector<float> output_values(input_values.size());
    hip_check_error(hipMemcpy(output_values.data(),
                              d_output,
                              output_values.size() * sizeof(float),
                              hipMemcpyDeviceToHost));

    // Free device memory
    hip_check_error(hipFree(d_input));
    hip_check_error(hipFree(d_output));

    // Verify the contents after modification
    const float tol = 2.5e-3f;
    EXPECT_NEAR(output_values[0], val_1, tol);
    EXPECT_NEAR(output_values[1], val_2, tol);
    EXPECT_NEAR(output_values[2], val_3, tol);
    EXPECT_NEAR(output_values[3], val_4, tol);
}
