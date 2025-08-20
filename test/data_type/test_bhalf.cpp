// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/stream_config.hpp"

using ck::bhalf_t;
using ck::type_convert;

__global__ void cast_roundtrip(const float2 input, float2* output)
{
    const ck::bhalf2_t bhalf2_val = ck::bf16x2_convert_rne<ck::bhalf2_t, float>(input.x, input.y);
    const float fval1 = type_convert<float>(bhalf2_val[0]);
    const float fval2 = type_convert<float>(bhalf2_val[1]);
    output->x = fval1;
    output->y = fval2;
}

__global__ void packed_cast(const float x1, const float x2, ck::bhalf2_t* output)
{
    *output = ck::bf16x2_convert_rne<ck::bhalf2_t, float>(x1, x2);
}

__global__ void cast(const float input, float* output)
{
    const bhalf_t bhalf_val = type_convert<bhalf_t>(input);
    *output                 = type_convert<float>(bhalf_val);
}

__global__ void packed_cast_in_place(const float x1, const float x2, ck::bhalf2_t* output)
{
    union 
    {
        float src;
        ck::bhalf2_t dst;
    } converter;
    
    float x = x1;
    ck::static_cast_float_to_bhalf_packed_v2(x, x2);

    converter.src = x;
    *output = converter.dst;
}

enum struct CastMode : int
{
    Standard = 0,
    Packed = 1,
    PackedInPlace = 2
};

template <CastMode PackedCast, int NumElements>
__global__ void test_performance_kernel(float* input, ck::bhalf_t* output)
{
    ck::bhalf_t buffer_bf16[NumElements];
    float buffer_float[NumElements];

    // Initialize input data
    for(int i = 0; i < NumElements; i++)
    {
        buffer_float[i] = input[i];
    }

    // Do enough work to offset kernel launch overhead and memory transfers.
    for(int i = 0; i < NumElements; i++)
    {
        for (int j = 0; j < NumElements; j++)
        {
            int index = (i + j) % NumElements;
            index = index < NumElements - 1 ? index : NumElements - 2;
            if constexpr (PackedCast == CastMode::Packed)
            {
                ck::bhalf2_t* buffer_range = reinterpret_cast<ck::bhalf2_t*>(&buffer_bf16[index]);
                *buffer_range = ck::bf16x2_convert_rne<ck::bhalf2_t, float>(buffer_float[i], buffer_float[j]);
            }
            else
            {
                buffer_bf16[index] = ck::bf16_convert_rtn_base(buffer_float[i]);
                buffer_bf16[index + 1] = ck::bf16_convert_rtn_base(buffer_float[j]);
            }
        }
    }

    // Copy results back to output
    for(int i = 0; i < NumElements; i++)
    {
        output[i] = buffer_bf16[i];
    }
}

template <CastMode PackedCast, int NumElements>
__global__ void test_in_place_performance_kernel(float* input, ck::bhalf_t* output)
{
    ck::bhalf_t buffer_bf16[NumElements];
    float buffer_float[NumElements];

    // Initialize input data
    for(int i = 0; i < NumElements; i++)
    {
        buffer_float[i] = input[i];
    }

    if constexpr (PackedCast == CastMode::PackedInPlace)
    {
        union 
        {   
            float src;
            ck::bhalf2_t dst;
        } workspace;

        for(int i = 0; i < NumElements; i++)
        {
            for (int j = 0; j < NumElements; j++)
            {
                int index = (i + j) % NumElements;
                index = index < NumElements - 1 ? index : NumElements - 2;
                workspace.src = buffer_float[i];
                ck::static_cast_float_to_bhalf_packed_v2(workspace.src, buffer_float[j]);
                ck::bhalf2_t* buffer_range = reinterpret_cast<ck::bhalf2_t*>(&buffer_bf16[index]);
                *buffer_range = workspace.dst;
            }
        }
    }
    else 
    {
        for(int i = 0; i < NumElements; i++)
        {
            for (int j = 0; j < NumElements; j++)
            {
                int index = (i + j) % NumElements;
                index = index < NumElements - 1 ? index : NumElements - 2;
                buffer_bf16[index] = ck::bf16_convert_rtn_base(buffer_float[i]);
                buffer_bf16[index + 1] = ck::bf16_convert_rtn_base(buffer_float[j]);
            }
        }
    }

    // Copy results back to output
    for(int i = 0; i < NumElements; i++)
    {
        output[i] = buffer_bf16[i];
    }
}

template <int NumElements>
void run_performance_test()
{
    float* input_dev;
    ck::bhalf_t* output_dev;
    std::vector<ck::bhalf_t> output_host(NumElements);

    hip_check_error(hipMalloc(&input_dev, sizeof(float) * NumElements));
    hip_check_error(hipMalloc(&output_dev, sizeof(ck::bhalf_t) * NumElements));

    // Initialize input data on the device
    std::vector<float> input_host(NumElements);
    for (int i = 0; i < NumElements; i++)
    {
        input_host[i] = 3.14f * static_cast<float>(i) - 1.7f;
    }

    hip_check_error(hipMemcpy(input_dev, input_host.data(), sizeof(float) * NumElements, hipMemcpyHostToDevice));

    StreamConfig stream_config;
    stream_config.time_kernel_ = true;

    auto baseline_kernel = test_performance_kernel<CastMode::Standard, NumElements>;
    auto packed_kernel = test_performance_kernel<CastMode::Packed, NumElements>;

    constexpr dim3 grid_size(1);
    constexpr dim3 block_size(1);
    constexpr size_t shared_mem_size = 0;

    const float baseline_time = launch_and_time_kernel(stream_config, baseline_kernel, grid_size, block_size, shared_mem_size, input_dev, output_dev);
    hip_check_error(hipMemcpy(output_host.data(), output_dev, sizeof(ck::bhalf_t) * NumElements, hipMemcpyDeviceToHost));

    const float packed_time = launch_and_time_kernel(stream_config, packed_kernel, grid_size, block_size, shared_mem_size, input_dev, output_dev);
    hip_check_error(hipMemcpy(output_host.data(), output_dev, sizeof(ck::bhalf_t) * NumElements, hipMemcpyDeviceToHost));

    // Cleanup
    hip_check_error(hipFree(input_dev));
    hip_check_error(hipFree(output_dev));

    std::cout << "Packed cast time ( " << NumElements << " elements): " << packed_time << " ms" << std::endl;
    std::cout << "Baseline cast time ( " << NumElements << " elements): " << baseline_time << " ms" << std::endl;

    // Check if packed cast is faster than baseline
    ASSERT_LT(packed_time, baseline_time);
}

template <int NumElements>
void run_in_place_performance_test()
{
    float* input_dev;
    ck::bhalf_t* output_dev;
    std::vector<ck::bhalf_t> output_host(NumElements);

    hip_check_error(hipMalloc(&input_dev, sizeof(float) * NumElements));
    hip_check_error(hipMalloc(&output_dev, sizeof(ck::bhalf_t) * NumElements));

    // Initialize input data on the device
    std::vector<float> input_host(NumElements);
    for (int i = 0; i < NumElements; i++)
    {
        input_host[i] = 3.14f * static_cast<float>(i) - 1.7f;
    }

    hip_check_error(hipMemcpy(input_dev, input_host.data(), sizeof(float) * NumElements, hipMemcpyHostToDevice));

    StreamConfig stream_config;
    stream_config.time_kernel_ = true;

    auto baseline_kernel = test_in_place_performance_kernel<CastMode::Standard, NumElements>;
    auto packed_kernel = test_in_place_performance_kernel<CastMode::PackedInPlace, NumElements>;

    constexpr dim3 grid_size(1);
    constexpr dim3 block_size(1);
    constexpr size_t shared_mem_size = 0;

    const float baseline_time = launch_and_time_kernel(stream_config, baseline_kernel, grid_size, block_size, shared_mem_size, input_dev, output_dev);
    hip_check_error(hipMemcpy(output_host.data(), output_dev, sizeof(ck::bhalf_t) * NumElements, hipMemcpyDeviceToHost));

    const float packed_time = launch_and_time_kernel(stream_config, packed_kernel, grid_size, block_size, shared_mem_size, input_dev, output_dev);
    hip_check_error(hipMemcpy(output_host.data(), output_dev, sizeof(ck::bhalf_t) * NumElements, hipMemcpyDeviceToHost));

    // Cleanup
    hip_check_error(hipFree(input_dev));
    hip_check_error(hipFree(output_dev));

    std::cout << "Packed cast time ( " << NumElements << " elements): " << packed_time << " ms" << std::endl;
    std::cout << "Baseline cast time ( " << NumElements << " elements): " << baseline_time << " ms" << std::endl;

    // Check if packed cast is faster than baseline
    ASSERT_LT(packed_time, baseline_time);
}

TEST(BHALF_T, Nan)
{
    const uint16_t binary_bhalf_nan = 0x7FC0;
    const bhalf_t bhalf_nan         = ck::bit_cast<bhalf_t>(binary_bhalf_nan);
    EXPECT_EQ(bhalf_nan, type_convert<bhalf_t>(ck::NumericLimits<float>::QuietNaN()));
}

TEST(BHALF_T, Inf)
{
    const uint16_t binary_bhalf_inf = 0x7F80;
    const bhalf_t bhalf_inf         = ck::bit_cast<bhalf_t>(binary_bhalf_inf);
    EXPECT_EQ(bhalf_inf, type_convert<bhalf_t>(ck::NumericLimits<float>::Infinity()));
}

TEST(BHALF_T, MantisaOverflow)
{
    const float abs_tol   = std::pow(2, -7);
    const uint32_t val    = 0x81FFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_NEAR(float_val, type_convert<float>(type_convert<bhalf_t>(float_val)), abs_tol);
}

TEST(BHALF_T, ExpOverflow)
{
    const uint32_t val    = 0xFF800000;
    const float float_val = ck::bit_cast<float>(val);
    ASSERT_EQ(type_convert<float>(type_convert<bhalf_t>(float_val)), float_val);
}

TEST(BHALF_T, MantisaExpOverflow)
{
    const uint32_t val    = 0xFFFFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_TRUE(std::isnan(float_val));
    ASSERT_TRUE(std::isnan(type_convert<float>(type_convert<bhalf_t>(float_val))));
}

TEST(BHALF_T, Performance)
{
    if (ck::get_device_name() == "gfx950")
    {
        run_performance_test<32>();
        run_performance_test<64>();
        run_performance_test<128>();
    }
    else
    {
        GTEST_SKIP() << "Packed cast performance test requires gfx950.";
    }
}

TEST(BHALF_T, PackedCastCorrectness)
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

    ck::bhalf2_t* value_after_cast_dev;
    ck::bhalf2_t value_after_cast_host;
    hip_check_error(hipMalloc(&value_after_cast_dev, sizeof(ck::bhalf2_t)));

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

    const auto& test = [&](const float x1, const float x2) 
        {
            packed_cast<<<1, 1>>>(x1, x2, value_after_cast_dev);
            hip_check_error(hipGetLastError());
            hip_check_error(hipMemcpy(&value_after_cast_host,
                                    value_after_cast_dev,
                                    sizeof(ck::bhalf2_t),
                                    hipMemcpyDeviceToHost));

            // Convert back to floats
            const float x1_actual = type_convert<float>(value_after_cast_host[0]);
            const float x2_actual = type_convert<float>(value_after_cast_host[1]);
            ASSERT_NEAR(x1_actual, x1, get_tolerance(x1));
            ASSERT_NEAR(x2_actual, x2, get_tolerance(x2));
        };

    for(int i = 0; i < num_vals; i++)
    {
        for (int j = 0; j < num_vals; j++)
        {
            const float exact_in_both_value = exact_in_both[i];
            const float exact_fp32_not_bf16_value = exact_fp32_not_bf16[j];

            test(exact_in_both_value, exact_fp32_not_bf16_value);
            test(exact_in_both_value, -exact_fp32_not_bf16_value);
            test(-exact_in_both_value, exact_fp32_not_bf16_value);
            test(-exact_in_both_value, -exact_fp32_not_bf16_value);
            test(exact_fp32_not_bf16_value, exact_in_both_value);
            test(exact_fp32_not_bf16_value, -exact_in_both_value);
            test(-exact_fp32_not_bf16_value, exact_in_both_value);
            test(-exact_fp32_not_bf16_value, -exact_in_both_value);
        }
    }
}

TEST(BHALF_T, PackedCastRoundtrip)
{
    constexpr int num_vals = 11;
    const float abs_tol    = std::pow(2, -7);
    std::vector<float> float_vals {0.5, 0.875, 1.5, 1, 2, 4, 8, 16, 32, 64, 128};

    float2* float_val_after_cast_dev;
    float2 float_val_after_cast_host;
    hip_check_error(hipMalloc(&float_val_after_cast_dev, sizeof(float2)));

    // Positive
    for(int i = 0; i < num_vals; i++)
    {
        for (int j = 0; j < num_vals; j++)
        {
            cast_roundtrip<<<1, 1>>>(float2{float_vals[i], float_vals[j]}, float_val_after_cast_dev);
            hip_check_error(hipGetLastError());
            hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                    float_val_after_cast_dev,
                                    sizeof(float2),
                                    hipMemcpyDeviceToHost));

            ASSERT_NEAR(float_val_after_cast_host.x, float_vals[i], abs_tol);
            ASSERT_NEAR(float_val_after_cast_host.y, float_vals[j], abs_tol);
        }
    }

    // Negative
    for(int i = 0; i < num_vals; i++)
    {
        for (int j = 0; j < num_vals; j++)
        {
            cast_roundtrip<<<1, 1>>>(float2{-float_vals[i], -float_vals[j]}, float_val_after_cast_dev);
            hip_check_error(hipGetLastError());
            hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                    float_val_after_cast_dev,
                                    sizeof(float2),
                                    hipMemcpyDeviceToHost));

            ASSERT_NEAR(float_val_after_cast_host.x, -float_vals[i], abs_tol);
            ASSERT_NEAR(float_val_after_cast_host.y, -float_vals[j], abs_tol);
        }
    }

    hip_check_error(hipFree(float_val_after_cast_dev));
}

TEST(BHALF_T, CastOnDevice)
{
    constexpr int num_vals     = 11;
    const float abs_tol        = std::pow(2, -7);
    float float_vals[num_vals] = {0.5, 0.875, 1.5, 1, 2, 4, 8, 16, 32, 64, 128};

    float* float_val_after_cast_dev;
    float float_val_after_cast_host;
    hip_check_error(hipMalloc(&float_val_after_cast_dev, sizeof(float)));

    // Positive
    for(int idx = 0; idx < num_vals; idx++)
    {
        cast<<<1, 1>>>(float_vals[idx], float_val_after_cast_dev);

        hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                  float_val_after_cast_dev,
                                  sizeof(float),
                                  hipMemcpyDeviceToHost));

        ASSERT_NEAR(float_val_after_cast_host, float_vals[idx], abs_tol);
    }
    // Negative
    for(int idx = 0; idx < num_vals; idx++)
    {
        cast<<<1, 1>>>(-float_vals[idx], float_val_after_cast_dev);

        hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                  float_val_after_cast_dev,
                                  sizeof(float),
                                  hipMemcpyDeviceToHost));

        ASSERT_NEAR(float_val_after_cast_host, -float_vals[idx], abs_tol);
    }
}

TEST(BHALF_T, PackedCast_in_place)
{
    const float v1 = 3.14f;
    const float v2 = -1.618f;
    ck::bhalf2_t* bhalf2_val_d;
    hip_check_error(hipMalloc(&bhalf2_val_d, sizeof(ck::bhalf2_t)));

    packed_cast_in_place<<<1, 1>>>(v1, v2, bhalf2_val_d);
    hip_check_error(hipGetLastError());

    ck::bhalf2_t bhalf2_val_h;
    hip_check_error(hipMemcpy(&bhalf2_val_h, bhalf2_val_d, sizeof(ck::bhalf2_t), hipMemcpyDeviceToHost));

    // Convert back to floats
    const float fval1 = type_convert<float>(bhalf2_val_h[0]);
    const float fval2 = type_convert<float>(bhalf2_val_h[1]);

    const float abs_tol = std::pow(2, -7);
    ASSERT_NEAR(fval1, v1, abs_tol);
    ASSERT_NEAR(fval2, v2, abs_tol);

    hip_check_error(hipFree(bhalf2_val_d));
}

TEST(BHALF_T, PackedCast_in_place_performance)
{
    if (ck::get_device_name() == "gfx950")
    {
        run_in_place_performance_test<32>();
        run_in_place_performance_test<64>();
        run_in_place_performance_test<128>();
    }
    else
    {
        GTEST_SKIP() << "Packed cast performance test requires gfx950.";
    }
}
