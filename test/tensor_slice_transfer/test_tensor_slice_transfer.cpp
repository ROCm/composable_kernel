// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using namespace ck;

// Test configuration constants
constexpr index_t TestM2 = 4;
constexpr index_t TestM4 = 2;
constexpr index_t TestCShuffleMXdlPerWavePerShuffle = 4;
constexpr index_t TestCShuffleNXdlPerWavePerShuffle = 8;
constexpr auto I0 = Number<0>{};

// Mock GPU kernel for testing data transfer
template<typename SrcData, typename DstData, bool UsePackedCast, bool TransferOutputToGlobalMemory = true>
__global__ void testVGPRToLDSTransfer_kernel(
    SrcData* input_data,
    DstData* output_data,
    index_t num_elements)
{
    // Simulate thread buffer (VGPR)
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

    // Simulate LDS buffer descriptor
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

    constexpr auto src_slice_origin_index = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);

    // Create thread buffer and populate with input data
    constexpr auto buffer_size = TestCShuffleMXdlPerWavePerShuffle * 
                                TestCShuffleNXdlPerWavePerShuffle * 
                                TestM2 * TestM4;
    
    StaticBuffer<AddressSpaceEnum::Vgpr, SrcData, buffer_size, true> src_thread_buf;
    
    // Initialize thread buffer with test data.
    // Each thread will handle a slice of the input data.
    const index_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < num_elements) {
        static_for<0, buffer_size, 1>{}([&](auto i) {
            src_thread_buf(i) = input_data[thread_id * buffer_size + i.value];
        });
    }

    // Allocate shared memory for LDS
    __shared__ DstData lds_data[buffer_size * 256]; // Assume 256 threads max
    
    auto lds_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
        &lds_data[thread_id * buffer_size], buffer_size);


    // Create threadwise transfer
    using ElementOp = tensor_operation::element_wise::PassThrough;
    ElementOp element_op{};

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
            Sequence<0, 1, 2, 3, 4, 5, 7, 6>, // Note: 7, 6 are swapped to enable vectorized transfer.
            7, // DstVectorDim
            2, // DstScalarPerVector
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

    // Perform the transfer
    if (thread_id < num_elements) {
        thread_transfer.Run(c_thread_desc,
                           src_slice_origin_index,
                           src_thread_buf,
                           lds_desc,
                           lds_buf);

        __syncthreads();

        if constexpr (TransferOutputToGlobalMemory)
        {
            // Copy results back to global memory
            static_for<0, buffer_size, 1>{}([&](auto i) {
                output_data[thread_id * buffer_size + i.value] = lds_buf[Number<i.value>{}];
            });
        }
    }
}

ck::bhalf_t convert(float x)
{
    if(x != x)
    {
        return uint16_t(0x7FC0);
    }

    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    const uint32_t first_bf16_mantisa_bit = ((u.int32 >> 16) & 1);
    constexpr uint32_t rounding_bias      = uint32_t((1 << 15) - 1);

    return uint16_t((u.int32 + first_bf16_mantisa_bit + rounding_bias) >> 16);
}

// Test class for VGPR to LDS transfer
class VGPRToLDSTransferTest : public ::testing::Test 
{
public:
    template <bool UsePackedCast, bool UseGpu>
    void run()
    {
        std::ignore = run<UsePackedCast, UseGpu>(false, 0, 0);
    };

    template <bool UsePackedCast>
    std::optional<float> run(index_t num_iters, index_t num_warmup_iters)
    {
        return run<UsePackedCast, true>(true, num_iters, num_warmup_iters);
    };
private:
    template <bool UsePackedCast, bool UseGpu>
    std::optional<float> run(bool time_kernel, index_t num_iters, index_t num_warmup_iters)
    {
        if constexpr  (UseGpu) 
        {
            return run_device<UsePackedCast>(time_kernel, num_iters, num_warmup_iters);
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

    template <bool UsePackedCast>
    std::optional<float> run_device(bool time_kernel, index_t num_iters, index_t num_warmup_iters)
    {
        constexpr index_t num_threads = 64;
        constexpr index_t elements_per_thread = TestCShuffleMXdlPerWavePerShuffle * 
                                            TestCShuffleNXdlPerWavePerShuffle * 
                                            TestM2 * TestM4;
        constexpr index_t total_elements = num_threads * elements_per_thread;

        // Host data
        std::vector<float> h_input(total_elements);
        std::vector<ck::bhalf_t> h_output(total_elements);
        std::vector<ck::bhalf_t> h_reference(total_elements);

        // Initialize input data
        for (index_t i = 0; i < total_elements; ++i) {
            h_input[i] = static_cast<float>(i) - 5.0f;
            h_reference[i] = convert(h_input[i]);
        }

        // Device data
        float* d_input;
        ck::bhalf_t* d_output;
        
        HIP_CHECK_ERROR(hipMalloc(&d_input, total_elements * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&d_output, total_elements * sizeof(ck::bhalf_t)));
        HIP_CHECK_ERROR(hipMemcpy(d_input, h_input.data(), total_elements * sizeof(float), hipMemcpyHostToDevice));

        std::optional<float> kernel_average_execution_time = std::nullopt;

        // Launch kernel
        dim3 grid(1), block(num_threads);
        if (time_kernel)
        {
            hipEvent_t start, stop;
            HIP_CHECK_ERROR(hipEventCreate(&start));
            HIP_CHECK_ERROR(hipEventCreate(&stop));

            // Warmup iterations
            for (index_t i = 0; i < num_warmup_iters; ++i) 
            {
                testVGPRToLDSTransfer_kernel<float, ck::bhalf_t, UsePackedCast, false><<<grid, block>>>(d_input, d_output, num_threads);
            }
            HIP_CHECK_ERROR(hipDeviceSynchronize());

            // Timing iterations
            HIP_CHECK_ERROR(hipEventRecord(start));
            for (index_t i = 0; i < num_iters; ++i) 
            {
                testVGPRToLDSTransfer_kernel<float, ck::bhalf_t, UsePackedCast, false><<<grid, block>>>(d_input, d_output, num_threads);
            }
            HIP_CHECK_ERROR(hipEventRecord(stop));
            HIP_CHECK_ERROR(hipEventSynchronize(stop));

            float milliseconds = 0.0f;
            HIP_CHECK_ERROR(hipEventElapsedTime(&milliseconds, start, stop));
            kernel_average_execution_time = milliseconds / num_iters;

            HIP_CHECK_ERROR(hipEventDestroy(start));
            HIP_CHECK_ERROR(hipEventDestroy(stop));
        }
        else 
        {
            testVGPRToLDSTransfer_kernel<float, ck::bhalf_t, UsePackedCast><<<grid, block>>>(d_input, d_output, num_threads);
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

TEST_F(VGPRToLDSTransferTest, FloatToBhalf_device_NoPack) 
{
    constexpr bool UsePackedCast = false;
    constexpr bool UseGpu = true;
    run<UsePackedCast, UseGpu>();
}

TEST_F(VGPRToLDSTransferTest, FloatToBhalf_device_PackedCast) 
{
    constexpr bool UsePackedCast = true;
    constexpr bool UseGpu = true;
    run<UsePackedCast, UseGpu>();
}

TEST_F(VGPRToLDSTransferTest, FloatToBhalf_device_test_peformance) 
{
    const int num_iters = 250;
    const int num_warmup_iters = 10;
    const auto packed_cast_time = run<true>(num_iters, num_warmup_iters);
    const auto baseline_time = run<false>(num_iters, num_warmup_iters);

    const auto default_value = std::numeric_limits<float>::signaling_NaN();
    std::cout << "Baseline average execution time: " 
              << (baseline_time.has_value() ? *baseline_time : default_value) << " ms" << std::endl;
    std::cout << "Packed cast average execution time: " 
              << (packed_cast_time.has_value() ? *packed_cast_time : default_value) << " ms" << std::endl;

    if (baseline_time && packed_cast_time) {
        EXPECT_LT(*packed_cast_time, *baseline_time);
    }
    else {
        GTEST_FAIL() << "Failed to get average execution time for one or both runs.";
    }
}
