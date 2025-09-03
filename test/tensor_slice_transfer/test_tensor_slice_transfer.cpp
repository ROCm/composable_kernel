// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/static_buffer.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/device_prop.hpp"

using namespace ck;

static constexpr auto I0 = Number<0>{};
static constexpr auto I1 = Number<1>{};

constexpr index_t NumThreads = 64;

__global__ void packed_cast_kernel(float x1, float x2, ck::bhalf2_t* output)
{
    typename vector_type_maker<ck::bhalf_t, 2>::type dst_buffer;
    
    dst_buffer.template AsType<ck::bhalf2_t>()(I0) = bf16x2_convert_rne<ck::bhalf2_t, float>(x1, x2);
    
    // Store the packed value to the output pointers
    (*output)[0] = dst_buffer.template AsType<ck::bhalf_t>()[I0];
    (*output)[1] = dst_buffer.template AsType<ck::bhalf_t>()[I1];
}

// Execute threadwise slice transfer from VGPR to LDS.
// We want to measure the performance of the transfer operation, and each thread can do 
// perform the same transfer operation.
template<index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4,
    typename SrcData, typename DstData, bool UsePackedCast, bool TransferOutputToGlobalMemory = true, int NRepeats = 1>
__global__ void testVGPRToLDSTransfer_kernel(
    SrcData* input_data,
    DstData* output_data,
    index_t num_elements)
{
    // Thread buffer (VGPR)
    constexpr auto c_thread_desc = make_naive_tensor_descriptor(
        make_tuple(Number<TestCShuffleMXdlPerWavePerShuffle>{},
                   Number<TestCShuffleNXdlPerWavePerShuffle>{},
                   Number<1>{},
                   Number<1>{},
                   Number<TestM2>{},
                   Number<1>{},
                   Number<TestM4>{},
                   Number<1>{}),
        make_tuple(Number<TestCShuffleNXdlPerWavePerShuffle * 1 * 1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * 1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * TestM4 * 1>{},
                   Number<TestM4 * 1>{},
                   Number<1>{},
                   Number<1>{}));

    // LDS buffer, this can be the same for each thread since we are interested in the threadwise transfer performance/correctness.
    constexpr auto lds_desc = make_naive_tensor_descriptor(
        make_tuple(Number<TestCShuffleMXdlPerWavePerShuffle>{},
                   Number<TestCShuffleNXdlPerWavePerShuffle>{},
                   Number<1>{},
                   Number<1>{},
                   Number<TestM2>{},
                   Number<1>{},
                   Number<TestM4>{},
                   Number<1>{}),
        make_tuple(Number<TestCShuffleNXdlPerWavePerShuffle * 1 * 1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * 1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * TestM2 * 1 * TestM4 * 1>{},
                   Number<TestM2 * 1 * TestM4 * 1>{},
                   Number<1 * TestM4 * 1>{},
                   Number<TestM4 * 1>{},
                   Number<1>{},
                   Number<1>{}));

    // We run the whole transfer in one go.
    constexpr auto src_slice_origin_index = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);

    // Create thread buffer and populate with input data
    constexpr auto buffer_size = TestCShuffleMXdlPerWavePerShuffle * 
                                TestCShuffleNXdlPerWavePerShuffle * 
                                TestM2 * TestM4;

    // Allocate shared memory for LDS
    __shared__ DstData lds_data[buffer_size * NumThreads];
    
    //StaticBuffer<AddressSpaceEnum::Vgpr, SrcData, buffer_size, true> src_thread_buf;
    static constexpr index_t vector_size = 2;
    static constexpr index_t num_vectors = buffer_size / vector_size;
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, SrcData, num_vectors, vector_size, true> src_thread_buf;
    
    // Initialize thread buffer with test data.
    // Each thread will handle a slice of the input data.
    const index_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t offset_in_global_memory = thread_id * buffer_size;
    if (thread_id < num_elements) 
    {
        static_for<0, buffer_size, 1>{}([&](auto i) 
        {
            src_thread_buf(i) = input_data[offset_in_global_memory + i.value];
        });
    }

    // The packed cast requires that the element-wise op is a pass-through operation.
    using ElementOp = tensor_operation::element_wise::PassThrough;
    ElementOp element_op{};

    // This could be compile time constant since we run the transfer in one go.
    // However, the API allows this to be dynamic, so we keep it as such.
    const auto dst_slice_origin_index = make_multi_index(0, 0, 0, 0, 0, 0, 0, 0);

    using TransferType = std::conditional_t<
        UsePackedCast,
            ThreadwiseTensorSliceTransfer_v1r3_packed_cast<
                SrcData,
                DstData,
                decltype(c_thread_desc),
                decltype(lds_desc),
                ElementOp,
                Sequence<TestCShuffleMXdlPerWavePerShuffle,
                        TestCShuffleNXdlPerWavePerShuffle,
                        1, 1, TestM2, 1, TestM4, 1>,
                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                7, // DstVectorDim
                1, // DstScalarPerVector
                InMemoryDataOperationEnum::Set,
                1, // DstScalarStrideInVector
                true // DstResetCoordinateAfterRun
            >,
            ThreadwiseTensorSliceTransfer_v1r3<
                SrcData,
                DstData,
                decltype(c_thread_desc),
                decltype(lds_desc),
                ElementOp,
                Sequence<TestCShuffleMXdlPerWavePerShuffle,
                        TestCShuffleNXdlPerWavePerShuffle,
                        1, 1, TestM2, 1, TestM4, 1>,
                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                7, // DstVectorDim
                1, // DstScalarPerVector
                InMemoryDataOperationEnum::Set,
                1, // DstScalarStrideInVector
                true // DstResetCoordinateAfterRun
            >
        >;

    auto thread_transfer = TransferType{lds_desc, dst_slice_origin_index, element_op};

    // Perform the transfer from VGPRs to LDS.
    // To mesure the performance, we repeat the transfer NRepeats times.
    if (thread_id < num_elements) 
    {
        static_for<0, NRepeats, 1>{}([&](auto i) 
        {
            // Create a view to the LDS slice of this thread.
            const auto offset_in_lds = ((i + thread_id) % NumThreads) * buffer_size;
            auto lds_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                &lds_data[offset_in_lds], buffer_size);

            thread_transfer.Run(c_thread_desc,
                            src_slice_origin_index,
                            src_thread_buf,
                            lds_desc,
                            lds_buf);    
        });
    }

    if constexpr (TransferOutputToGlobalMemory)
    {
        // Ensure all threads have written to LDS.
        // This is important if we process the threadwise slice in smaller parts.
        // Currently, we run the whole transfer in one go, so this is not strictly necessary.
        __syncthreads();

        // Copy results back to global memory
        const auto offset_in_lds = thread_id * buffer_size;
        auto lds_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            &lds_data[offset_in_lds], buffer_size);

        static_for<0, buffer_size, 1>{}([&](auto i) {
            output_data[thread_id * buffer_size + i.value] = lds_buf[Number<i.value>{}];
        });
    }

}

void run_single_packed_cast_test(const float x1, const float x2, std::function<void(float, float, ck::bhalf2_t*)> launch_kernel)
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

    launch_kernel(x1, x2, value_after_cast_dev);
    hip_check_error(hipGetLastError());
    hip_check_error(hipMemcpy(&value_after_cast_host,
                            value_after_cast_dev,
                            sizeof(ck::bhalf2_t),
                            hipMemcpyDeviceToHost));

    hip_check_error(hipFree(value_after_cast_dev));

    // Convert back to floats
    const float x1_actual = type_convert<float>(value_after_cast_host[0]);
    const float x2_actual = type_convert<float>(value_after_cast_host[1]);
    ASSERT_NEAR(x1_actual, x1, get_tolerance(x1));
    ASSERT_NEAR(x2_actual, x2, get_tolerance(x2));
};

void run_packed_cast_test(std::function<void(float, float, ck::bhalf2_t*)> launch_kernel)
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

            run_single_packed_cast_test(exact_in_both_value, exact_fp32_not_bf16_value, launch_kernel);
            run_single_packed_cast_test(exact_in_both_value, -exact_fp32_not_bf16_value, launch_kernel);
            run_single_packed_cast_test(-exact_in_both_value, exact_fp32_not_bf16_value, launch_kernel);
            run_single_packed_cast_test(-exact_in_both_value, -exact_fp32_not_bf16_value, launch_kernel);
            run_single_packed_cast_test(exact_fp32_not_bf16_value, exact_in_both_value, launch_kernel);
            run_single_packed_cast_test(exact_fp32_not_bf16_value, -exact_in_both_value, launch_kernel);
            run_single_packed_cast_test(-exact_fp32_not_bf16_value, exact_in_both_value, launch_kernel);
            run_single_packed_cast_test(-exact_fp32_not_bf16_value, -exact_in_both_value, launch_kernel);
        }
    }
}

// Test class for VGPR to LDS transfer
class VGPRToLDSTransferTest : public ::testing::Test 
{
public:
    template <index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4>
    void run_perf_test(float required_speedup)
    {
        constexpr int NRepeats = 5000;
        const int num_iters = 250;
        const int num_warmup_iters = 10;

        const auto packed_cast_time = run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, true, NRepeats>(num_iters, num_warmup_iters);
        const auto baseline_time = run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, false, NRepeats>(num_iters, num_warmup_iters);

        const auto default_value = std::numeric_limits<float>::signaling_NaN();
        std::cout << "Performance test results for case: "
                << "MXdlPerWavePerShuffle=" << TestCShuffleMXdlPerWavePerShuffle
                << ", NXdlPerWavePerShuffle=" << TestCShuffleNXdlPerWavePerShuffle
                << ", M2=" << TestM2
                << ", M4=" << TestM4
                << std::endl;
        std::cout << "Baseline average execution time = " 
                << (baseline_time.has_value() ? *baseline_time : default_value) << " ms" << std::endl;
        std::cout << "Packed cast average execution time = " 
                << (packed_cast_time.has_value() ? *packed_cast_time : default_value) << " ms" << std::endl;

        if (baseline_time && packed_cast_time) {
            const float speedup = (*baseline_time - *packed_cast_time)  / *baseline_time;
            std::cout << "Speedup = " << speedup * 100.0f << "%" << std::endl;
            EXPECT_GT(speedup, required_speedup) << "Packed cast should be at least " << 100.0f * required_speedup << "% faster than baseline";
        }
        else {
            GTEST_FAIL() << "Failed to get average execution time for one or both runs.";
        }
    }

    template <index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4, bool UsePackedCast, bool UseGpu>
    void run()
    {
        std::ignore = run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, UsePackedCast, UseGpu>(false, 0, 0);
    };

    template <index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4, bool UsePackedCast, int NRepeats>
    std::optional<float> run(index_t num_iters, index_t num_warmup_iters)
    {
        return run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, UsePackedCast, true>(true, num_iters, num_warmup_iters);
    };
private:
    template <index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4, bool UsePackedCast, bool UseGpu, int NRepeats=1>
    std::optional<float> run(bool time_kernel, index_t num_iters, index_t num_warmup_iters)
    {
        if constexpr (UseGpu) 
        {
            return run_device<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, UsePackedCast, NRepeats>(time_kernel, num_iters, num_warmup_iters);
        }
        else 
        {
            run_host<UsePackedCast>();
            return std::nullopt;
        }
    };

    template <bool UsePackedCast>
    void run_host()
    {
        // Fail the test as it it is not yet implemented.
        GTEST_FAIL() << "Host transfer test not implemented yet.";
    };

    template <index_t TestCShuffleMXdlPerWavePerShuffle, index_t TestCShuffleNXdlPerWavePerShuffle, index_t TestM2, index_t TestM4, bool UsePackedCast, int NRepeats = 1>
    std::optional<float> run_device(bool time_kernel, index_t num_iters, index_t num_warmup_iters)
    {
        constexpr index_t elements_per_thread = TestCShuffleMXdlPerWavePerShuffle * 
                                            TestCShuffleNXdlPerWavePerShuffle * 
                                            TestM2 * TestM4;
        constexpr index_t total_elements = NumThreads * elements_per_thread;

        // Host data
        std::vector<float> h_input(total_elements);
        std::vector<ck::bhalf_t> h_output(total_elements);
        std::vector<ck::bhalf_t> h_reference(total_elements);

        // Initialize input data
        for (index_t i = 0; i < total_elements; ++i) {
            h_input[i] = static_cast<float>(i);
            h_reference[i] = ck::bf16_convert_rtn_base(h_input[i]);
        }

        // Device data
        float* d_input;
        ck::bhalf_t* d_output;
        
        HIP_CHECK_ERROR(hipMalloc(&d_input, total_elements * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&d_output, total_elements * sizeof(ck::bhalf_t)));
        HIP_CHECK_ERROR(hipMemcpy(d_input, h_input.data(), total_elements * sizeof(float), hipMemcpyHostToDevice));

        std::optional<float> kernel_average_execution_time = std::nullopt;

        // Launch kernel
        dim3 grid(1), block(NumThreads);
        if (time_kernel)
        {
            StreamConfig stream_config;
            stream_config.time_kernel_ = true;
            stream_config.cold_niters_ = num_warmup_iters;
            stream_config.nrepeat_ = num_iters;
            auto kernel = testVGPRToLDSTransfer_kernel<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, float, ck::bhalf_t, UsePackedCast, false, NRepeats>;
            kernel_average_execution_time = launch_and_time_kernel(stream_config, kernel, grid, block, 0, d_input, d_output, total_elements);
        }
        else 
        {
            testVGPRToLDSTransfer_kernel<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, float, ck::bhalf_t, UsePackedCast><<<grid, block>>>(d_input, d_output, total_elements);
            HIP_CHECK_ERROR(hipDeviceSynchronize());

            // Copy results back
            HIP_CHECK_ERROR(hipMemcpy(h_output.data(), d_output, total_elements * sizeof(ck::bhalf_t), hipMemcpyDeviceToHost));

            // Verify results
            index_t errors = 0;
            float max_error = 0.0f;
            const float abs_tolerance = 1e-3f; // Allow small tolerance for bf16 conversion
            for (index_t i = 0; i < total_elements; ++i) {
                float output_val = ck::type_convert<float>(h_output[i]);
                float ref_val = ck::type_convert<float>(h_reference[i]);
                float val_error = std::abs(output_val - ref_val);
                
                if (val_error > abs_tolerance) 
                {
                    errors++;
                    max_error = std::max(max_error, val_error);
                    // Print first 10 errors
                    if (errors <= 10) 
                    { 
                        std::cout << "Error at " << i << ": got " << output_val 
                                << ", expected " << ref_val << ", error " << val_error << std::endl;
                    }
                }
            }

            std::cout << "Total errors: " << errors << "/" << total_elements 
                    << " (" << (100.0f * errors / total_elements) << "%)" << std::endl;
            std::cout << "Max error: " << max_error << std::endl;

            // Allow up to 1% errors due to precision differences
            const float fraction_of_allowed_errors = 0.01f;
            EXPECT_LT(static_cast<float>(errors) / total_elements, fraction_of_allowed_errors);
        }

        HIP_CHECK_ERROR(hipFree(d_input));
        HIP_CHECK_ERROR(hipFree(d_output));

        return kernel_average_execution_time;
    }
protected:
    void SetUp() override {
        hipError_t hip_status = hipGetDeviceCount(&device_count_);
        if (hip_status != hipSuccess || device_count_ == 0) {
            GTEST_SKIP() << "No HIP devices available, skipping GPU tests";
        }
    }

    int device_count_ = 0;
};

TEST(PackedCast, vectorized_float2_to_bhalf2)
{
    run_packed_cast_test([](float x1, float x2, ck::bhalf2_t* output) {
        packed_cast_kernel<<<1, 1>>>(x1, x2, output);
    });
}

TEST_F(VGPRToLDSTransferTest, FloatToBhalf_device_NoPack) 
{
    constexpr bool UsePackedCast = false;
    constexpr bool UseGpu = true;
    constexpr index_t TestM2 = 4;
    constexpr index_t TestM4 = 2;
    constexpr index_t TestCShuffleMXdlPerWavePerShuffle = 1;
    constexpr index_t TestCShuffleNXdlPerWavePerShuffle = 1;
    run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, UsePackedCast, UseGpu>();
}

TEST_F(VGPRToLDSTransferTest, FloatToBhalf_device_PackedCast) 
{
    constexpr bool UsePackedCast = true;
    constexpr bool UseGpu = true;
    constexpr index_t TestM2 = 4;
    constexpr index_t TestM4 = 2;
    constexpr index_t TestCShuffleMXdlPerWavePerShuffle = 1;
    constexpr index_t TestCShuffleNXdlPerWavePerShuffle = 1;
    run<TestCShuffleMXdlPerWavePerShuffle, TestCShuffleNXdlPerWavePerShuffle, TestM2, TestM4, UsePackedCast, UseGpu>();
}

// This test might occasionally although the tolerances are quite lenient.
TEST_F(VGPRToLDSTransferTest, DISABLED_FloatToBhalf_device_test_peformance) 
{
    if (ck::get_device_name() == "gfx950")
    {
        // Relevant cases for convolution gridwise GEMMs.
        //            MXdlPerWavePerShuffle  NXdlPerWavePerShuffle   M2              M4
        run_perf_test<1,                     1,                      4,              4>(0.1);       // 10% speedup required
        run_perf_test<1,                     1,                      1,              4>(0.01);      // 1% speedup required
        run_perf_test<1,                     1,                      4,              1>(0.01);      // 1% speedup required
    }
    else 
    {
        GTEST_SKIP() << "Performance test skipped on non-gfx950 devices.";
    }
}
