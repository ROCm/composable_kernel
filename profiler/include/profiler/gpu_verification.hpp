// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/type.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/check_err.hpp"

namespace ck {
namespace profiler {

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

// GPU verification kernel - compares device result against reference using relative and absolute
// tolerance Returns 1 in passed if all elements match within tolerance, 0 otherwise
template <typename T>
__global__ void gpu_verify_kernel(const T* __restrict__ device_result,
                                  const T* __restrict__ reference_result,
                                  float rtol,
                                  float atol,
                                  long long size,
                                  int* passed)
{
    // Grid-stride loop to handle any tensor size
    long long idx    = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for(long long i = idx; i < size; i += stride)
    {
        // Convert to float for comparison
        float dev_val = type_convert<float>(device_result[i]);
        float ref_val = type_convert<float>(reference_result[i]);

        // Compute absolute difference
        float abs_diff = fabsf(dev_val - ref_val);

        // Check tolerance (matches CPU check_err logic: err > atol + rtol * abs(ref))
        if(abs_diff > atol + rtol * fabsf(ref_val))
        {
            atomicMin(passed, 0); // Mark as failed
            return;               // Early exit on first failure
        }
    }
}

// Host-side wrapper for GPU verification with explicit tolerances
// Returns true if verification passed, false otherwise
template <typename T>
bool gpu_verify(const void* device_result,
                const void* reference_result,
                float rtol,
                float atol,
                std::size_t size,
                hipStream_t stream = nullptr)
{
    // Allocate result buffer on device
    int* passed_dev;
    hip_check_error(hipMalloc(&passed_dev, sizeof(int)));

    // Initialize to passed (1)
    int passed_host = 1;
    hip_check_error(hipMemcpy(passed_dev, &passed_host, sizeof(int), hipMemcpyHostToDevice));

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
                                               passed_dev);

    hip_check_error(hipGetLastError());

    // Synchronize the stream to ensure kernel completion before reading results
    hip_check_error(hipStreamSynchronize(stream));

    // Get result
    hip_check_error(hipMemcpy(&passed_host, passed_dev, sizeof(int), hipMemcpyDeviceToHost));

    // Free device memory
    hip_check_error(hipFree(passed_dev));

    return passed_host == 1;
}

// Forward declaration of gpu_reduce_max
template <typename T>
float gpu_reduce_max(const void* device_buffer, std::size_t size, hipStream_t stream = nullptr);

// Host-side wrapper for GPU verification with automatic tolerance computation
// Computes max value on GPU, then computes tolerances and verifies
// Returns true if verification passed, false otherwise
template <typename OutDataType,
          typename ComputeDataType = OutDataType,
          typename AccDataType     = ComputeDataType>
bool gpu_verify(const void* device_result,
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

//
// Helper function for atomic float max (using compare-and-swap)
__device__ __forceinline__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_int = reinterpret_cast<int*>(address);
    int old             = *address_as_int;
    int assumed;

    do
    {
        assumed = old;
        old =
            atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while(assumed != old);

    return __int_as_float(old);
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
    for(unsigned int s = block_size / 2; s > 32; s >>= 1)
    {
        if(threadIdx.x < s)
        {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Warp-level reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // No sync needed within a warp
    if(threadIdx.x < 32)
    {
        volatile float* smem = shared_max;
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 32]);
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 16]);
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 8]);
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 4]);
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 2]);
        smem[threadIdx.x]    = fmaxf(smem[threadIdx.x], smem[threadIdx.x + 1]);
    }

    // Two-phase reduction pattern minimizes atomic contention:
    // 1. Each block reduces to shared memory (above)
    // 2. Single thread per block updates global max (below)
    // This limits atomic operations to O(grid_size) rather than O(total_threads)
    if(threadIdx.x == 0)
    {
        atomicMaxFloat(max_val, shared_max[0]);
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
