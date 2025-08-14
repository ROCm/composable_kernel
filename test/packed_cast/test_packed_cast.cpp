// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include "gtest/gtest.h"

#include "ck/utility/data_type.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using namespace ck;

static constexpr auto I0 = Number<0>{};
static constexpr auto I1 = Number<1>{};

__global__ void packed_cast(float x1, float x2, ck::bhalf2_t* output)
{
    typename vector_type_maker<ck::bhalf_t, 2>::type dst_buffer;
    
    dst_buffer.template AsType<ck::bhalf2_t>()(I0) = bf16x2_convert_rne<ck::bhalf2_t, float>(x1, x2);
    
    // Store the packed value to the output pointers
    (*output)[0] = dst_buffer.template AsType<ck::bhalf_t>()[I0];
    (*output)[1] = dst_buffer.template AsType<ck::bhalf_t>()[I1];
}

void run_test(const float x1, const float x2) 
{
    const auto& get_tolerance = [](const float test_val) -> float
    {
        const float abs_tol    = std::pow(2, -7);
        constexpr float rel_tol = 1e-3f; 
        if (std::abs(test_val) > 128.0f) 
        {
            return std::abs(test_val) * rel_tol;  // Relative error
        } 
        else 
        {
            return abs_tol;  // Absolute error for small values
        }
    };

    ck::bhalf2_t* value_after_cast_dev;
    ck::bhalf2_t value_after_cast_host;
    hip_check_error(hipMalloc(&value_after_cast_dev, sizeof(ck::bhalf2_t)));

    packed_cast<<<1, 1>>>(x1, x2, value_after_cast_dev);
    hip_check_error(hipGetLastError());
    hip_check_error(hipMemcpy(&value_after_cast_host,
                            value_after_cast_dev,
                            sizeof(ck::bhalf2_t),
                            hipMemcpyDeviceToHost));

    hip_check_error(hipFree(value_after_cast_dev));

    // Convert back to floats
    const float x1_actual = type_convert<float>(value_after_cast_host[0]);
    const float x2_actual = type_convert<float>(value_after_cast_host[1]);
    std::cout << "x1: " << x1_actual << ", x2: " << x2_actual << std::endl;
    ASSERT_NEAR(x1_actual, x1, get_tolerance(x1));
    ASSERT_NEAR(x2_actual, x2, get_tolerance(x2));
};

TEST(packed_cast, vectorized_float2_to_bhalf2)
{
    // Test packed cast from bhalf2 to float2
    // Use values that are representable in bhalf2 as well as values that are not
    constexpr int num_vals = 15;

    std::vector<float> exact_in_both {
        0.0f,
        1.0f,  
        2.0f,
        8.0f,
        32.0f,
        128.0f,
        0.5f, 
        0.25f,
        0.125f,
        0.0625f,
        1.5f, 
        2.5f, 
        3.0f, 
        7.0f,
        15.0f
    };

    std::vector<float> exact_fp32_not_bf16 {
        // Small fractional values requiring more than 7 mantissa bits
        0.1f,           // 0.1 needs more precision
        0.3f,           // 0.3 = 3/10
        0.7f,           // 0.7 = 7/10
        0.9f,           // 0.9 = 9/10
        
        // Values with fine granularity
        1.1f,           // 1.1 = 11/10
        1.01f,          // 1.01 = 101/100
        1.001f,         // Even finer
        2.1f,           // 2.1 = 21/10
        
        // Values requiring >7 mantissa bits
        1.0078125f,     // Needs 8+ mantissa bits
        1.00390625f,    // Needs 9+ mantissa bits
        
        // Small values near zero
        1e-6f,          // Very small
        1e-5f,
        1e-4f,
        
        // Values just outside BF16 range precision
        65504.5f,       // Close to BF16 max but needs more precision
        0.00006103515625f, // 2^-14, at edge of BF16 precision
    };

    for(int i = 0; i < num_vals; i++)
    {
        for (int j = 0; j < num_vals; j++)
        {
            const float exact_in_both_value = exact_in_both[i];
            const float exact_fp32_not_bf16_value = exact_fp32_not_bf16[j];

            run_test(exact_in_both_value, exact_fp32_not_bf16_value);
            run_test(exact_in_both_value, -exact_fp32_not_bf16_value);
            run_test(-exact_in_both_value, exact_fp32_not_bf16_value);
            run_test(-exact_in_both_value, -exact_fp32_not_bf16_value);
            run_test(exact_fp32_not_bf16_value, exact_in_both_value);
            run_test(exact_fp32_not_bf16_value, -exact_in_both_value);
            run_test(-exact_fp32_not_bf16_value, exact_in_both_value);
            run_test(-exact_fp32_not_bf16_value, -exact_in_both_value);
        }
    }
}
