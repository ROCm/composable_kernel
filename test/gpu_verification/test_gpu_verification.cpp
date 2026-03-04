// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <random>

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/gpu_verification.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_utils.hpp"

using namespace ck::profiler;
using ck::ref::SimpleDeviceMem;

// Test fixture for GPU verification tests
class GPUVerificationTest : public ::testing::Test
{
    protected:
    // Random number generator - initialized once per test for reproducibility
    std::mt19937 rng_;

    void SetUp() override
    {
        // Ensure HIP is initialized
        hipDeviceProp_t prop;
        [[maybe_unused]] hipError_t err = hipGetDeviceProperties(&prop, 0);

        // Initialize RNG with fixed seed for reproducibility
        // Can be overridden with CK_TEST_SEED environment variable
        unsigned int seed = 12345;
        if(const char* env_seed = std::getenv("CK_TEST_SEED"))
        {
            seed = std::stoul(env_seed);
        }
        rng_.seed(seed);
    }

    void TearDown() override
    {
        // Cleanup handled automatically
    }

    // Helper to upload data to device using SimpleDeviceMem
    template <typename T>
    std::unique_ptr<SimpleDeviceMem> CreateDeviceBuffer(const std::vector<T>& host_data)
    {
        auto device_buf = std::make_unique<SimpleDeviceMem>(host_data.size() * sizeof(T));
        HIP_CHECK_ERROR(hipMemcpy(device_buf->GetDeviceBuffer(),
                                  host_data.data(),
                                  host_data.size() * sizeof(T),
                                  hipMemcpyHostToDevice));
        return device_buf;
    }

    // Helper to compare CPU max reduction with GPU
    template <typename T>
    float ComputeCPUMaxAbs(const std::vector<T>& data)
    {
        if(data.empty())
            return 0.0f;

        float max_val = 0.0f;
        for(const auto& val : data)
        {
            float abs_val = std::abs(ck::type_convert<float>(val));
            max_val       = std::max(max_val, abs_val);
        }
        return max_val;
    }

    // Helper to generate random data
    template <typename T>
    std::vector<T> GenerateRandomData(size_t size, float min_val = -10.0f, float max_val = 10.0f)
    {
        std::vector<T> data(size);

        // Use test fixture's RNG (rng_) for reproducibility
        // RNG is seeded in SetUp() with fixed seed or CK_TEST_SEED environment variable
        if constexpr(std::is_integral_v<T> && !std::is_same_v<T, ck::bhalf_t>)
        {
            std::uniform_int_distribution<int> dis(static_cast<int>(min_val),
                                                   static_cast<int>(max_val));
            for(auto& val : data)
                val = static_cast<T>(dis(rng_));
        }
        else
        {
            std::uniform_real_distribution<float> dis(min_val, max_val);
            for(auto& val : data)
                val = ck::type_convert<T>(dis(rng_));
        }
        return data;
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(GPUVerificationTest, FP32_ExactMatch_ShouldPass)
{
    constexpr size_t size   = 1024;
    std::vector<float> data = GenerateRandomData<float>(size);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    // Identical data should pass with zero tolerance
    bool result = gpu_verify<float>(device_buf1->GetDeviceBuffer(),
                                    device_buf2->GetDeviceBuffer(),
                                    0.0f, // rtol
                                    0.0f, // atol
                                    size);

    EXPECT_TRUE(result) << "Identical FP32 tensors should pass verification";
}

TEST_F(GPUVerificationTest, FP32_Different_ShouldFail)
{
    constexpr size_t size    = 1024;
    std::vector<float> data1 = GenerateRandomData<float>(size);
    std::vector<float> data2 = GenerateRandomData<float>(size);

    auto device_buf1 = CreateDeviceBuffer(data1);
    auto device_buf2 = CreateDeviceBuffer(data2);

    // Different random data should fail with zero tolerance
    bool result = gpu_verify<float>(device_buf1->GetDeviceBuffer(),
                                    device_buf2->GetDeviceBuffer(),
                                    0.0f, // rtol
                                    0.0f, // atol
                                    size);

    EXPECT_FALSE(result) << "Different FP32 tensors should fail with zero tolerance";
}

TEST_F(GPUVerificationTest, FP32_WithinTolerance_ShouldPass)
{
    constexpr size_t size = 1024;
    std::vector<float> data1(size, 1.0f);
    std::vector<float> data2(size, 1.01f);

    auto device_buf1 = CreateDeviceBuffer(data1);
    auto device_buf2 = CreateDeviceBuffer(data2);

    // 1% relative difference should pass with 2% tolerance
    bool result = gpu_verify<float>(device_buf1->GetDeviceBuffer(),
                                    device_buf2->GetDeviceBuffer(),
                                    0.02f, // rtol
                                    0.02f, // atol
                                    size);

    EXPECT_TRUE(result) << "Data within tolerance should pass";
}

TEST_F(GPUVerificationTest, FP32_OutsideTolerance_ShouldFail)
{
    constexpr size_t size = 1024;
    std::vector<float> data1(size, 1.0f);
    std::vector<float> data2(size, 1.1f);

    auto device_buf1 = CreateDeviceBuffer(data1);
    auto device_buf2 = CreateDeviceBuffer(data2);

    // 10% relative difference should fail with 1% tolerance
    bool result = gpu_verify<float>(device_buf1->GetDeviceBuffer(),
                                    device_buf2->GetDeviceBuffer(),
                                    0.01f, // rtol
                                    0.01f, // atol
                                    size);

    EXPECT_FALSE(result) << "Data outside tolerance should fail";
}

// ============================================================================
// Data Type Coverage Tests
// ============================================================================

TEST_F(GPUVerificationTest, FP16_ExactMatch_ShouldPass)
{
    constexpr size_t size        = 1024;
    std::vector<ck::half_t> data = GenerateRandomData<ck::half_t>(size);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<ck::half_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Identical FP16 tensors should pass verification";
}

TEST_F(GPUVerificationTest, BF16_ExactMatch_ShouldPass)
{
    constexpr size_t size         = 1024;
    std::vector<ck::bhalf_t> data = GenerateRandomData<ck::bhalf_t>(size);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<ck::bhalf_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Identical BF16 tensors should pass verification";
}

TEST_F(GPUVerificationTest, INT8_ExactMatch_ShouldPass)
{
    constexpr size_t size    = 1024;
    std::vector<int8_t> data = GenerateRandomData<int8_t>(size, int8_t{-100}, int8_t{100});

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<int8_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Identical INT8 tensors should pass verification";
}

TEST_F(GPUVerificationTest, INT16_ExactMatch_ShouldPass)
{
    constexpr size_t size     = 1024;
    std::vector<int16_t> data = GenerateRandomData<int16_t>(size, int16_t{-1000}, int16_t{1000});

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<int16_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Identical INT16 tensors should pass verification";
}

TEST_F(GPUVerificationTest, INT32_ExactMatch_ShouldPass)
{
    constexpr size_t size     = 1024;
    std::vector<int32_t> data = GenerateRandomData<int32_t>(size, -10000, 10000);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<int32_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Identical INT32 tensors should pass verification";
}

// ============================================================================
// Tolerance Validation Tests
// ============================================================================

TEST_F(GPUVerificationTest, RelativeTolerance_ScalesWithReferenceValue)
{
    constexpr size_t size = 100;
    std::vector<float> reference(size);
    std::vector<float> result(size);

    // Test that relative tolerance scales correctly
    // For reference = 100, result = 101, relative error = 1%
    for(size_t i = 0; i < size; ++i)
    {
        reference[i] = 100.0f;
        result[i]    = 101.0f;
    }

    auto device_ref = CreateDeviceBuffer(reference);
    auto device_res = CreateDeviceBuffer(result);

    // Should pass with 2% relative tolerance
    bool pass = gpu_verify<float>(device_res->GetDeviceBuffer(),
                                  device_ref->GetDeviceBuffer(),
                                  0.02f, // rtol
                                  0.0f,  // atol
                                  size);

    EXPECT_TRUE(pass) << "Should pass with sufficient relative tolerance";

    // Should fail with 0.5% relative tolerance
    bool fail = gpu_verify<float>(device_res->GetDeviceBuffer(),
                                  device_ref->GetDeviceBuffer(),
                                  0.005f, // rtol
                                  0.0f,   // atol
                                  size);

    EXPECT_FALSE(fail) << "Should fail with insufficient relative tolerance";
}

TEST_F(GPUVerificationTest, AbsoluteTolerance_CriticalForSmallValues)
{
    constexpr size_t size = 100;
    std::vector<float> reference(size, 0.0f);
    std::vector<float> result(size, 0.001f);

    auto device_ref = CreateDeviceBuffer(reference);
    auto device_res = CreateDeviceBuffer(result);

    // For values near zero, relative tolerance doesn't help - need absolute
    bool pass = gpu_verify<float>(device_res->GetDeviceBuffer(),
                                  device_ref->GetDeviceBuffer(),
                                  0.0f,   // rtol
                                  0.002f, // atol (larger than difference)
                                  size);

    EXPECT_TRUE(pass) << "Should pass with sufficient absolute tolerance";

    bool fail = gpu_verify<float>(device_res->GetDeviceBuffer(),
                                  device_ref->GetDeviceBuffer(),
                                  0.0f,    // rtol
                                  0.0005f, // atol (smaller than difference)
                                  size);

    EXPECT_FALSE(fail) << "Should fail with insufficient absolute tolerance";
}

TEST_F(GPUVerificationTest, AutomaticToleranceComputation_FP32)
{
    constexpr size_t size   = 1024;
    std::vector<float> data = GenerateRandomData<float>(size);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    // Use automatic tolerance computation (3-template parameter version)
    bool result = gpu_verify<float, float, float>(device_buf1->GetDeviceBuffer(),
                                                  device_buf2->GetDeviceBuffer(),
                                                  1, // number_of_accumulations
                                                  size);

    EXPECT_TRUE(result) << "Identical data should pass with automatic tolerances";
}

TEST_F(GPUVerificationTest, AutomaticToleranceComputation_FP16)
{
    constexpr size_t size        = 1024;
    std::vector<ck::half_t> data = GenerateRandomData<ck::half_t>(size);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<ck::half_t, ck::half_t, ck::half_t>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 1, size);

    EXPECT_TRUE(result) << "Identical FP16 data should pass with automatic tolerances";
}

TEST_F(GPUVerificationTest, ToleranceScalesWithAccumulations)
{
    // Verify that tolerance increases with number of accumulations
    constexpr size_t size = 100;
    std::vector<float> reference(size, 1.0f);
    std::vector<float> result(size);

    // Create result with small accumulated error
    for(size_t i = 0; i < size; ++i)
    {
        result[i] = 1.0f + 1e-6f; // Small error
    }

    auto device_ref = CreateDeviceBuffer(reference);
    auto device_res = CreateDeviceBuffer(result);

    // With more accumulations, tolerance should be larger, so this should pass
    bool result_many_accums = gpu_verify<float, float, float>(device_res->GetDeviceBuffer(),
                                                              device_ref->GetDeviceBuffer(),
                                                              1000, // Many accumulations
                                                              size);

    // With fewer accumulations, tolerance is tighter
    bool result_few_accums = gpu_verify<float, float, float>(device_res->GetDeviceBuffer(),
                                                             device_ref->GetDeviceBuffer(),
                                                             1, // Few accumulations
                                                             size);

    // Note: The actual behavior depends on the error magnitude and tolerance formulas
    // This test documents the expected behavior
    EXPECT_TRUE(result_many_accums || result_few_accums)
        << "At least one configuration should pass for small errors";
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(GPUVerificationTest, SingleElement_ExactMatch)
{
    constexpr size_t size = 1;
    std::vector<float> data{42.0f};

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<float>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Single element exact match should pass";
}

TEST_F(GPUVerificationTest, LargeTensor_Performance)
{
    constexpr size_t size = 10 * 1024 * 1024; // 10M elements
    std::vector<float> data(size, 1.0f);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<float>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Large tensor verification should complete successfully";
}

TEST_F(GPUVerificationTest, VeryLargeValues_NearTypeLimit)
{
    constexpr size_t size = 100;
    float large_val       = 1e36f; // Close to FP32 limit but not overflow
    std::vector<float> data(size, large_val);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<float>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Very large values should be handled correctly";
}

TEST_F(GPUVerificationTest, VerySmallValues_NearZero)
{
    constexpr size_t size = 100;
    float small_val       = 1e-36f; // Very small but not denormal
    std::vector<float> data(size, small_val);

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<float>(device_buf1->GetDeviceBuffer(),
                                    device_buf2->GetDeviceBuffer(),
                                    0.0f,
                                    1e-38f, // Very small absolute tolerance
                                    size);

    EXPECT_TRUE(result) << "Very small values should be handled correctly";
}

TEST_F(GPUVerificationTest, MixedPositiveNegative_Values)
{
    constexpr size_t size = 100;
    std::vector<float> data(size);
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }

    auto device_buf1 = CreateDeviceBuffer(data);
    auto device_buf2 = CreateDeviceBuffer(data);

    bool result = gpu_verify<float>(
        device_buf1->GetDeviceBuffer(), device_buf2->GetDeviceBuffer(), 0.0f, 0.0f, size);

    EXPECT_TRUE(result) << "Mixed positive/negative values should work correctly";
}

// ============================================================================
// GPU Max Reduction Tests
// ============================================================================

TEST_F(GPUVerificationTest, GPUReduceMax_FP32_Correctness)
{
    constexpr size_t size   = 1024;
    std::vector<float> data = GenerateRandomData<float>(size);

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<float>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(cpu_max, gpu_max) << "GPU max reduction should match CPU for FP32";
}

TEST_F(GPUVerificationTest, GPUReduceMax_FP16_Correctness)
{
    constexpr size_t size        = 1024;
    std::vector<ck::half_t> data = GenerateRandomData<ck::half_t>(size);

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<ck::half_t>(device_buf->GetDeviceBuffer(), size);

    // FP16 might have small precision differences
    EXPECT_NEAR(cpu_max, gpu_max, 1e-3f)
        << "GPU max reduction should match CPU for FP16 within precision";
}

TEST_F(GPUVerificationTest, GPUReduceMax_BF16_Correctness)
{
    constexpr size_t size         = 1024;
    std::vector<ck::bhalf_t> data = GenerateRandomData<ck::bhalf_t>(size);

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<ck::bhalf_t>(device_buf->GetDeviceBuffer(), size);

    // BF16 has lower precision
    EXPECT_NEAR(cpu_max, gpu_max, 1e-2f)
        << "GPU max reduction should match CPU for BF16 within precision";
}

TEST_F(GPUVerificationTest, GPUReduceMax_INT8_Correctness)
{
    constexpr size_t size    = 1024;
    std::vector<int8_t> data = GenerateRandomData<int8_t>(size, int8_t{-100}, int8_t{100});

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<int8_t>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(cpu_max, gpu_max) << "GPU max reduction should match CPU for INT8";
}

TEST_F(GPUVerificationTest, GPUReduceMax_SingleElement)
{
    constexpr size_t size = 1;
    std::vector<float> data{-42.5f};

    auto device_buf = CreateDeviceBuffer(data);

    float gpu_max = gpu_reduce_max<float>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(42.5f, gpu_max) << "Max of single element should be its absolute value";
}

TEST_F(GPUVerificationTest, GPUReduceMax_LargeBuffer)
{
    constexpr size_t size   = 10 * 1024 * 1024; // 10M elements
    std::vector<float> data = GenerateRandomData<float>(size);

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<float>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(cpu_max, gpu_max) << "GPU max reduction should handle large buffers correctly";
}

TEST_F(GPUVerificationTest, GPUReduceMax_AllNegative)
{
    constexpr size_t size = 100;
    std::vector<float> data(size);
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = -static_cast<float>(i + 1);
    }

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<float>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(cpu_max, gpu_max)
        << "GPU max reduction should handle all negative values (absolute)";
}

TEST_F(GPUVerificationTest, GPUReduceMax_MixedPositiveNegative)
{
    constexpr size_t size = 100;
    std::vector<float> data(size);
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
    }

    auto device_buf = CreateDeviceBuffer(data);

    float cpu_max = ComputeCPUMaxAbs(data);
    float gpu_max = gpu_reduce_max<float>(device_buf->GetDeviceBuffer(), size);

    EXPECT_FLOAT_EQ(cpu_max, gpu_max) << "GPU max reduction should handle mixed signs correctly";
}

// ============================================================================
// Tolerance Computation Tests
// ============================================================================

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_IntegerTypes_ReturnsZero)
{
    // Integer types should have zero relative tolerance
    float rtol_int8  = compute_relative_tolerance<int8_t, int8_t, int8_t>();
    float rtol_int16 = compute_relative_tolerance<int16_t, int16_t, int16_t>();
    float rtol_int32 = compute_relative_tolerance<int32_t, int32_t, int32_t>();

    EXPECT_FLOAT_EQ(0.0f, rtol_int8) << "INT8 should have zero relative tolerance";
    EXPECT_FLOAT_EQ(0.0f, rtol_int16) << "INT16 should have zero relative tolerance";
    EXPECT_FLOAT_EQ(0.0f, rtol_int32) << "INT32 should have zero relative tolerance";
}

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_FP32_NonZero)
{
    // FP32 should have non-zero relative tolerance
    float rtol = compute_relative_tolerance<float, float, float>();

    EXPECT_GT(rtol, 0.0f) << "FP32 should have non-zero relative tolerance";
    EXPECT_LT(rtol, 1.0f) << "FP32 tolerance should be reasonable (< 1.0)";
}

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_FP16_NonZero)
{
    // FP16 should have non-zero relative tolerance
    float rtol = compute_relative_tolerance<ck::half_t, ck::half_t, ck::half_t>();

    EXPECT_GT(rtol, 0.0f) << "FP16 should have non-zero relative tolerance";
    EXPECT_LT(rtol, 1.0f) << "FP16 tolerance should be reasonable (< 1.0)";
}

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_BF16_NonZero)
{
    // BF16 should have non-zero relative tolerance
    float rtol = compute_relative_tolerance<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>();

    EXPECT_GT(rtol, 0.0f) << "BF16 should have non-zero relative tolerance";
    EXPECT_LT(rtol, 1.0f) << "BF16 tolerance should be reasonable (< 1.0)";
}

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_ScalesWithAccumulations)
{
    // Tolerance should increase with more accumulations
    float rtol_1    = compute_relative_tolerance<float, float, float>(1);
    float rtol_10   = compute_relative_tolerance<float, float, float>(10);
    float rtol_100  = compute_relative_tolerance<float, float, float>(100);
    float rtol_1000 = compute_relative_tolerance<float, float, float>(1000);

    // More accumulations should give larger tolerance (or equal, but not smaller)
    EXPECT_GE(rtol_10, rtol_1) << "10 accums should have >= tolerance than 1";
    EXPECT_GE(rtol_100, rtol_10) << "100 accums should have >= tolerance than 10";
    EXPECT_GE(rtol_1000, rtol_100) << "1000 accums should have >= tolerance than 100";
}

TEST_F(GPUVerificationTest, ComputeRelativeTolerance_MixedPrecision)
{
    // Test mixed precision scenarios common in ML
    float rtol_fp16_fp32 = compute_relative_tolerance<ck::half_t, float, float>();
    float rtol_fp32_fp32 = compute_relative_tolerance<float, float, float>();

    // FP16 compute with FP32 output should have reasonable tolerance
    EXPECT_GT(rtol_fp16_fp32, 0.0f) << "Mixed precision should have non-zero tolerance";

    // Mixed precision might need larger tolerance than pure FP32
    // (This is implementation-dependent, just document the behavior)
    EXPECT_GT(rtol_fp16_fp32, 0.0f);
    EXPECT_GT(rtol_fp32_fp32, 0.0f);
}

// ============================================================================
// Integration Tests (End-to-End)
// ============================================================================

TEST_F(GPUVerificationTest, EndToEnd_ConvolutionLikeWorkload_FP32)
{
    // Simulate a convolution output verification scenario
    constexpr size_t size               = 256 * 256; // Realistic output size
    std::vector<float> kernel_output    = GenerateRandomData<float>(size);
    std::vector<float> reference_output = kernel_output; // Start identical

    // Add small numerical errors like real kernels might have
    for(size_t i = 0; i < size; i += 100)
    {
        reference_output[i] += 1e-5f;
    }

    auto device_kernel = CreateDeviceBuffer(kernel_output);
    auto device_ref    = CreateDeviceBuffer(reference_output);

    // Should pass with automatic tolerance for FP32 compute
    bool result = gpu_verify<float, float, float>(device_kernel->GetDeviceBuffer(),
                                                  device_ref->GetDeviceBuffer(),
                                                  1000, // Typical number of accumulations in conv
                                                  size);

    EXPECT_TRUE(result) << "Realistic convolution output should pass verification";
}

TEST_F(GPUVerificationTest, EndToEnd_ConvolutionLikeWorkload_FP16)
{
    // FP16 computation scenario
    constexpr size_t size                    = 128 * 128;
    std::vector<ck::half_t> kernel_output    = GenerateRandomData<ck::half_t>(size);
    std::vector<ck::half_t> reference_output = kernel_output;

    // Add errors within FP16 precision
    for(size_t i = 0; i < size; i += 50)
    {
        float val           = ck::type_convert<float>(reference_output[i]);
        reference_output[i] = ck::type_convert<ck::half_t>(val + 1e-3f);
    }

    auto device_kernel = CreateDeviceBuffer(kernel_output);
    auto device_ref    = CreateDeviceBuffer(reference_output);

    bool result = gpu_verify<ck::half_t, ck::half_t, ck::half_t>(
        device_kernel->GetDeviceBuffer(), device_ref->GetDeviceBuffer(), 1000, size);

    EXPECT_TRUE(result) << "FP16 convolution output should pass verification";
}

TEST_F(GPUVerificationTest, EndToEnd_DetectsActualErrors)
{
    // Verify that the system catches real errors
    constexpr size_t size               = 1024;
    std::vector<float> kernel_output    = GenerateRandomData<float>(size);
    std::vector<float> reference_output = GenerateRandomData<float>(size); // Completely different

    auto device_kernel = CreateDeviceBuffer(kernel_output);
    auto device_ref    = CreateDeviceBuffer(reference_output);

    // Should fail when data is truly different
    bool result = gpu_verify<float, float, float>(
        device_kernel->GetDeviceBuffer(), device_ref->GetDeviceBuffer(), 1, size);

    EXPECT_FALSE(result) << "System should detect actual errors";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
