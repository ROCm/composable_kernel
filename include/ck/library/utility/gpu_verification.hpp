// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/type.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/check_err.hpp"

namespace ck {
namespace profiler {

// Result struct for GPU verification with detailed error reporting
// Provides backward compatibility via operator bool()
struct GpuVerifyResult
{
    unsigned long long error_count; // Number of elements that exceeded tolerance
    float max_error;                // Maximum error value observed
    std::size_t total;              // Total number of elements compared
    bool all_zero;                  // True if device result is all zeros (likely kernel issue)

    // Implicit conversion to bool for backward compatibility
    // Allows: if (gpu_verify(...)) { ... }
    operator bool() const { return error_count == 0; }

    // Calculate error percentage
    float error_percentage() const
    {
        if(total == 0)
            return 0.0f;
        return static_cast<float>(error_count) / static_cast<float>(total) * 100.0f;
    }

    // Print error summary to stderr (matches check_err format)
    void print_error_summary() const
    {
        if(error_count > 0)
        {
            if(all_zero)
            {
                std::cerr << "WARNING: Device result is all zeros - kernel may not have executed "
                             "properly!"
                          << std::endl;
            }
            std::cerr << "max err: " << max_error;
            std::cerr << ", number of errors: " << error_count;
            std::cerr << ", " << std::setprecision(2) << std::fixed << error_percentage()
                      << "% wrong values" << std::endl;
        }
    }
};

// Compute relative tolerance for GPU verification
// Matches the logic of ck::utils::get_relative_threshold but handles all types
template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
inline float compute_relative_tolerance(const int number_of_accumulations = 1)
{
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;
    using I8   = int8_t;
    using I16  = int16_t;
    using I32  = int32_t;

    // For integer types, tolerance is 0
    if constexpr(std::is_same_v<ComputeDataType, I8> || std::is_same_v<ComputeDataType, I16> ||
                 std::is_same_v<ComputeDataType, I32> || std::is_same_v<ComputeDataType, int>)
    {
        return 0.0f;
    }
    // For types supported by get_relative_threshold, use it
    else if constexpr((std::is_same_v<ComputeDataType, F16> ||
                       std::is_same_v<ComputeDataType, BF16> ||
                       std::is_same_v<ComputeDataType, F32>) &&
                      (std::is_same_v<OutDataType, F16> || std::is_same_v<OutDataType, BF16> ||
                       std::is_same_v<OutDataType, F32>) &&
                      (std::is_same_v<AccDataType, F16> || std::is_same_v<AccDataType, BF16> ||
                       std::is_same_v<AccDataType, F32>))
    {
        return static_cast<float>(
            ck::utils::get_relative_threshold<ComputeDataType, OutDataType, AccDataType>(
                number_of_accumulations));
    }
    // For unsupported types (FP8, BF8, etc.), use default tolerances based on output type
    else
    {
        if constexpr(std::is_same_v<OutDataType, F16>)
        {
            return 1e-3f;
        }
        else if constexpr(std::is_same_v<OutDataType, BF16>)
        {
            return 1e-1f;
        }
        else
        {
            // For FP8/BF8 and other types, use conservative tolerance
            return 1e-1f;
        }
    }
}

// Device-side result structure for kernel output
// Packed into a single struct to minimize device memory allocations
struct GpuVerifyDeviceResult
{
    unsigned long long error_count; // Number of errors found
    float max_error;                // Maximum error value
    int all_zero;                   // 1 = device result is all zeros, 0 = has non-zero values
};

// GPU verification kernel - compares device result against reference using relative and absolute
// tolerance. Tracks all errors (no early exit) to provide detailed error reporting.
//
// Uses LDS (shared memory) for block-level reduction to minimize atomic contention.
// This reduces atomic operations from O(errors) to O(blocks), providing massive speedup
// when there are many errors.
//
// Assumption: Block size is 256
template <typename T>
__global__ void gpu_verify_kernel(const T* __restrict__ device_result,
                                  const T* __restrict__ reference_result,
                                  float rtol,
                                  float atol,
                                  long long size,
                                  GpuVerifyDeviceResult* result)
{
    constexpr int block_size = 256;

    // Shared memory for block-level reduction
    __shared__ unsigned long long shared_error_count[block_size];
    __shared__ float shared_max_error[block_size];
    __shared__ int shared_has_error[block_size];
    __shared__ int shared_has_nonzero[block_size];

    // Thread-local accumulators (in registers)
    unsigned long long local_error_count = 0;
    float local_max_error                = 0.0f;
    int local_has_error                  = 0;
    int local_has_nonzero                = 0;

    // Grid-stride loop to handle any tensor size
    long long idx    = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for(long long i = idx; i < size; i += stride)
    {
        // Convert to float for comparison
        float dev_val = type_convert<float>(device_result[i]);
        float ref_val = type_convert<float>(reference_result[i]);

        // Check if device value is non-zero
        if(dev_val != 0.0f)
        {
            local_has_nonzero = 1;
        }

        // Compute absolute difference
        float abs_diff = fabsf(dev_val - ref_val);

        // Check tolerance (matches CPU check_err logic: err > atol + rtol * abs(ref))
        if(abs_diff > atol + rtol * fabsf(ref_val))
        {
            local_has_error = 1;
            local_error_count++;
            local_max_error = fmaxf(local_max_error, abs_diff);
        }
    }

    // Store thread-local results to shared memory
    shared_error_count[threadIdx.x] = local_error_count;
    shared_max_error[threadIdx.x]   = local_max_error;
    shared_has_error[threadIdx.x]   = local_has_error;
    shared_has_nonzero[threadIdx.x] = local_has_nonzero;
    __syncthreads();

    // Block-level reduction: 256 -> 128 -> 64 -> 32
    for(unsigned int s = block_size / 2; s >= 32; s >>= 1)
    {
        if(threadIdx.x < s)
        {
            shared_error_count[threadIdx.x] += shared_error_count[threadIdx.x + s];
            shared_max_error[threadIdx.x] =
                fmaxf(shared_max_error[threadIdx.x], shared_max_error[threadIdx.x + s]);
            shared_has_error[threadIdx.x] |= shared_has_error[threadIdx.x + s];
            shared_has_nonzero[threadIdx.x] |= shared_has_nonzero[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final reduction of remaining 32 elements in thread 0
    if(threadIdx.x == 0)
    {
        for(int i = 1; i < 32; ++i)
        {
            shared_error_count[0] += shared_error_count[i];
            shared_max_error[0] = fmaxf(shared_max_error[0], shared_max_error[i]);
            shared_has_error[0] |= shared_has_error[i];
            shared_has_nonzero[0] |= shared_has_nonzero[i];
        }

        // Single atomic update per block (reduces contention from O(errors) to O(blocks))
        if(shared_has_error[0])
        {
            atomicAdd(&result->error_count, shared_error_count[0]);
            atomicMax(&result->max_error, shared_max_error[0]);
        }
        // Update all_zero flag: if no nonzero values found, mark as all zero
        if(!shared_has_nonzero[0])
        {
            atomicMin(&result->all_zero, 1);
        }
        else
        {
            atomicMin(&result->all_zero, 0);
        }
    }
}

// Host-side wrapper for GPU verification with explicit tolerances
// Returns GpuVerifyResult with detailed error information
template <typename T>
GpuVerifyResult gpu_verify(const void* device_result,
                           const void* reference_result,
                           float rtol,
                           float atol,
                           std::size_t size,
                           hipStream_t stream = nullptr)
{
    // Allocate result buffer on device
    GpuVerifyDeviceResult* result_dev;
    hip_check_error(hipMalloc(&result_dev, sizeof(GpuVerifyDeviceResult)));

    // Initialize result struct
    GpuVerifyDeviceResult result_host;
    result_host.error_count = 0;    // No errors yet
    result_host.max_error   = 0.0f; // No error observed
    result_host.all_zero    = 1;    // Start assuming all zeros (will be cleared if nonzero found)
    hip_check_error(
        hipMemcpy(result_dev, &result_host, sizeof(GpuVerifyDeviceResult), hipMemcpyHostToDevice));

    // Launch kernel with grid-stride loop
    // Use 65535 as max grid size (hardware limit for grid dimension in x)
    // Grid-stride loop handles any tensor size regardless of grid dimensions
    constexpr int block_size = 256;
    int grid_size            = std::min<int>(65535, (size + block_size - 1) / block_size);

    gpu_verify_kernel<T>
        <<<grid_size, block_size, 0, stream>>>(static_cast<const T*>(device_result),
                                               static_cast<const T*>(reference_result),
                                               rtol,
                                               atol,
                                               static_cast<long long>(size),
                                               result_dev);

    hip_check_error(hipGetLastError());

    // Synchronize the stream to ensure kernel completion before reading results
    hip_check_error(hipStreamSynchronize(stream));

    // Get result
    hip_check_error(
        hipMemcpy(&result_host, result_dev, sizeof(GpuVerifyDeviceResult), hipMemcpyDeviceToHost));

    // Free device memory
    hip_check_error(hipFree(result_dev));

    // Build and return result struct
    GpuVerifyResult result;
    result.error_count = result_host.error_count;
    result.max_error   = result_host.max_error;
    result.total       = size;
    result.all_zero    = (result_host.all_zero == 1);

    return result;
}

// Forward declaration of gpu_reduce_max
template <typename T>
float gpu_reduce_max(const void* device_buffer, std::size_t size, hipStream_t stream = nullptr);

// Host-side wrapper for GPU verification with automatic tolerance computation
// Computes max value on GPU, then computes tolerances and verifies
// Returns GpuVerifyResult with detailed error information
template <typename OutDataType,
          typename ComputeDataType = OutDataType,
          typename AccDataType     = ComputeDataType>
GpuVerifyResult gpu_verify(const void* device_result,
                           const void* reference_result,
                           int number_of_accumulations,
                           std::size_t size,
                           hipStream_t stream = nullptr)
{
    // Compute max absolute value on GPU (only 4 bytes transferred!)
    double max_abs_value =
        static_cast<double>(gpu_reduce_max<OutDataType>(reference_result, size, stream));

    // Compute tolerances based on data types and accumulation count
    float rtol = compute_relative_tolerance<ComputeDataType, OutDataType, AccDataType>(
        number_of_accumulations);

    float atol = 0.0f;
    // Only compute absolute tolerance for supported types
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;

    if constexpr((std::is_same_v<ComputeDataType, F16> || std::is_same_v<ComputeDataType, BF16> ||
                  std::is_same_v<ComputeDataType, F32>) &&
                 (std::is_same_v<OutDataType, F16> || std::is_same_v<OutDataType, BF16> ||
                  std::is_same_v<OutDataType, F32>) &&
                 (std::is_same_v<AccDataType, F16> || std::is_same_v<AccDataType, BF16> ||
                  std::is_same_v<AccDataType, F32>))
    {
        atol = static_cast<float>(
            ck::utils::get_absolute_threshold<ComputeDataType, OutDataType, AccDataType>(
                max_abs_value, number_of_accumulations));
    }

    // Call the explicit tolerance version
    return gpu_verify<OutDataType>(device_result, reference_result, rtol, atol, size, stream);
}

// GPU reduction kernel for computing max(abs(data))
// This is an internal kernel called only by gpu_reduce_max() wrapper.
//
// Assumption: Block size is 256
template <typename T>
__global__ void
gpu_reduce_max_kernel(const T* __restrict__ data, long long size, float* __restrict__ max_val)
{
    constexpr int block_size = 256;
    __shared__ float shared_max[block_size];

    long long idx    = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    float local_max = 0.0f;

    for(long long i = idx; i < size; i += stride)
    {
        float val = fabsf(type_convert<float>(data[i]));
        local_max = fmaxf(local_max, val);
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // Block-level reduction: 256 -> 128 -> 64 -> 32
    for(unsigned int s = block_size / 2; s >= 32; s >>= 1)
    {
        if(threadIdx.x < s)
        {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Final reduction of remaining 32 elements in thread 0
    if(threadIdx.x == 0)
    {
        for(int i = 1; i < 32; ++i)
        {
            shared_max[0] = fmaxf(shared_max[0], shared_max[i]);
        }

        // Single atomic update per block
        atomicMax(max_val, shared_max[0]);
    }
}

// Host-side wrapper for GPU max reduction
// Computes max(abs(data)) and returns as float
// Only transfers 4 bytes (the final max value) instead of entire tensor
template <typename T>
float gpu_reduce_max(const void* device_buffer, std::size_t size, hipStream_t stream)
{
    if(size == 0)
    {
        return 0.0f;
    }

    // Allocate device memory for result
    float* max_dev;
    hip_check_error(hipMalloc(&max_dev, sizeof(float)));

    // Initialize to zero
    float init_val = 0.0f;
    hip_check_error(hipMemcpy(max_dev, &init_val, sizeof(float), hipMemcpyHostToDevice));

    // Launch reduction kernel
    // Use 1024 blocks max for reduction to balance occupancy vs. grid-stride iterations
    // For very large tensors (>256M elements), grid-stride loop handles the remainder
    constexpr int block_size = 256;
    int grid_size            = std::min<int>(1024, (size + block_size - 1) / block_size);

    gpu_reduce_max_kernel<T><<<grid_size, block_size, 0, stream>>>(
        static_cast<const T*>(device_buffer), static_cast<long long>(size), max_dev);

    hip_check_error(hipGetLastError());

    // Synchronize if using default stream
    if(stream == nullptr)
    {
        hip_check_error(hipDeviceSynchronize());
    }

    // Copy result to host (only 4 bytes!)
    float max_host;
    hip_check_error(hipMemcpy(&max_host, max_dev, sizeof(float), hipMemcpyDeviceToHost));

    // Free device memory
    hip_check_error(hipFree(max_dev));

    return max_host;
}

} // namespace profiler
} // namespace ck
