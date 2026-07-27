// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <array>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/check_err.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
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

/// @brief Turn an iterator type into an iterator that can be dereferenced.
///
/// In gpu_verify and gpu_reduce_max, it is valid to pass a void pointer and
/// have the function automatically derive the "concrete" pointer type to
/// be used in the kernel. This function does that: depending on whether
/// the `Iterator` is a void pointer or not, it returns either the iterator
/// (assuming that it is already concrete), or returns the pointer casted
/// to the concrete type.
///
/// @tparam T The value type of the pointer, when dereferenced.
/// @tparam Iterator The abstract iterator, can be void* or an actual pointer.
///
/// @param it The iterator to make concrete.
template <typename T, typename Iterator>
__device__ Iterator make_concrete_iterator(Iterator it)
{
    return it;
}

template <typename T>
__device__ const T* make_concrete_iterator(const void* it)
{
    return reinterpret_cast<const T*>(it);
}

template <typename T>
__device__ const T* make_concrete_iterator(void* it)
{
    return reinterpret_cast<const T*>(it);
}

/// @brief Utility to launch persistent kernels.
///
/// This function launches a GPU kernel with a grid size derived from the kernel's
/// occupancy and the total number of multiprocessors on the GPU.
///
/// @tparam Kernel The type of the kernel function.
/// @tparam Args The types of the kernel arguments.
///
/// @param kernel An instance of the kernel function. This should be a __global__ function.
/// @param block_size The kernel's (1D) block size.
/// @param stream The stream to launch the kernel on.
/// @param args The kernel launch arguments.
template <typename Kernel, typename... Args>
void launch_persistent_kernel(Kernel kernel,
                              int block_size,
                              hipStream_t stream,
                              const Args&... args)
{
    int occupancy;
    hip_check_error(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    int device;
    hip_check_error(hipGetDevice(&device));

    int multiprocessors;
    hip_check_error(
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device));

    kernel<<<occupancy * multiprocessors, block_size, 0, stream>>>(args...);

    hip_check_error(hipGetLastError());
}

/// @brief Simple block reduce kernel.
///
/// This function reduces all `value`s across a block according to `reduce`. This function
/// is a relatively simple implementation as its primary purpose is to be correct and
/// readable: No special cases are done for warp reductions, and the function allocates
/// its own shared memory. The result is broadcasted to all threads.
///
/// @tparam BlockSize The number of threads in a block.
/// @tparam T The value type to reduce over.
/// @tparam F The reduction functor type.
///
/// @param value This thread's value to reduce over.
/// @param reduce The reduction functor, used to combine two values. Should be associative.
template <int BlockSize, typename T, typename F>
__device__ T block_reduce(const T& value, F reduce)
{
    __shared__ T workspace[BlockSize];

    workspace[threadIdx.x] = value;
    __syncthreads();

    for(unsigned int s = BlockSize / 2; s >= 1; s >>= 1)
    {
        if(threadIdx.x < s)
            workspace[threadIdx.x] = reduce(workspace[threadIdx.x], workspace[threadIdx.x + s]);
        __syncthreads();
    }

    return workspace[0];
}

// Device-side result structure for kernel output
// Packed into a single struct to minimize device memory allocations
struct GpuVerifyDeviceResult
{
    unsigned long long error_count; // Number of errors found
    float max_error;                // Maximum error value
    int all_zero;                   // 1 = device result is all zeros, 0 = has non-zero values

    /// @brief Return the neutral element of a GpuVerifyDeviceResult
    ///
    /// This function returns the "neutral element", the element which does nothing
    /// when reduced with another with `reduce_results`. Good to be used as an
    /// initial value.
    __host__ __device__ static GpuVerifyDeviceResult identity()
    {
        GpuVerifyDeviceResult result;
        result.error_count = 0;    // No errors yet
        result.max_error   = 0.0f; // No error observed
        result.all_zero    = 1;    // Start assuming all zeros (will be cleared if nonzero found)
        return result;
    }
};

/// @brief Combine two device verify results.
///
/// This function returns the "combined" version of two GpuVerifyDeviceResult values, which
/// adds the total amount of errors, sets the correct max error, and records whether
/// any of the values had any zeros.
__device__ GpuVerifyDeviceResult reduce_results(const GpuVerifyDeviceResult& a,
                                                const GpuVerifyDeviceResult& b)
{
    GpuVerifyDeviceResult result;
    result.error_count = a.error_count + b.error_count;
    result.max_error   = std::max(a.max_error, b.max_error);
    result.all_zero    = a.all_zero & b.all_zero;
    return result;
}

/// @brief Compare individual tensor elements.
///
/// This function is what actually does the comparison between two tensor
/// elements. The function returns a tuple of three elements.
/// - The absolute maximum difference.
/// - If the second value is set to false, it indicates either that the elements are not
///   equal according to the thresholds `rtol` and `atol`, or that either value is not
///   finite (NaN/Infinity). If set to true, the values are considered equal.
/// - If the third value is set to true, it indicates that both elements are bitwise
///   equal to zero.
template <typename T>
__device__ std::tuple<float, bool, bool>
compare_elements(const T& actual, const T& expected, const float rtol, const float atol)
{
    static_assert(!std::is_same_v<T, double>, "TODO: implement compare_elements() for double");

    const auto o = type_convert<float>(actual);
    const auto r = type_convert<float>(expected);
    const auto e = std::abs(o - r);

    const auto inequal = e > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r);

    using Bytes        = std::array<std::byte, sizeof(T)>;
    const auto o_bytes = *reinterpret_cast<const Bytes*>(&actual);
    const auto r_bytes = *reinterpret_cast<const Bytes*>(&expected);
    bool all_zero      = true;
    for(const auto x : o_bytes)
    {
        if(x != std::byte{0})
            all_zero = false;
    }

    for(const auto x : r_bytes)
    {
        if(x != std::byte{0})
            all_zero = false;
    }

    return std::make_tuple(e, inequal, all_zero);
}

// GPU verification kernel - compares device result against reference using relative and absolute
// tolerance. Tracks all errors (no early exit) to provide detailed error reporting.
//
// Uses LDS (shared memory) for block-level reduction to minimize atomic contention.
// This reduces atomic operations from O(errors) to O(blocks), providing massive speedup
// when there are many errors.
template <int BlockSize, typename T, typename IteratorA, typename IteratorB>
__global__ __launch_bounds__(BlockSize) //
    void gpu_verify_kernel(IteratorA device_result_it,
                           IteratorB reference_result_it,
                           float rtol,
                           float atol,
                           long long size,
                           GpuVerifyDeviceResult* result)
{
    auto device_result    = make_concrete_iterator<T>(device_result_it);
    auto reference_result = make_concrete_iterator<T>(reference_result_it);

    auto local_result = GpuVerifyDeviceResult::identity();

    // Grid-stride loop to handle any tensor size
    long long idx    = blockIdx.x * BlockSize + threadIdx.x;
    long long stride = BlockSize * gridDim.x;

    for(long long i = idx; i < size; i += stride)
    {
        const auto [abs_diff, inequal, bitwise_zero] =
            compare_elements(device_result[i], reference_result[i], rtol, atol);

        local_result = reduce_results(local_result,
                                      GpuVerifyDeviceResult{
                                          static_cast<uint64_t>(inequal), // error_count
                                          abs_diff,                       // max_error
                                          bitwise_zero                    // all_zero
                                      });
    }

    const auto block_result = block_reduce<BlockSize>(local_result, reduce_results);

    // Final reduction of remaining 32 elements in thread 0
    if(threadIdx.x == 0)
    {
        // Single atomic update per block (reduces contention from O(errors) to O(blocks))
        if(block_result.error_count > 0)
        {
            atomicAdd(&result->error_count, block_result.error_count);
            atomicMax(&result->max_error, block_result.max_error);
        }

        if(!block_result.all_zero)
        {
            // A nonzero was found, so set the global value to false.
            // Note: this is a benign race condition; technically a race condition but
            // all blocks write the same value, so its fine.
            result->all_zero = 0;
        }
    }
}

// Host-side wrapper for GPU verification with explicit tolerances
// Returns GpuVerifyResult with detailed error information
template <typename T, typename IteratorA, typename IteratorB>
GpuVerifyResult gpu_verify(IteratorA device_result,
                           IteratorB reference_result,
                           float rtol,
                           float atol,
                           std::size_t size,
                           hipStream_t stream = nullptr)
{
    // Allocate result buffer on device
    GpuVerifyDeviceResult* result_dev;
    hip_check_error(hipMalloc(&result_dev, sizeof(GpuVerifyDeviceResult)));

    // Initialize result struct
    auto result_host = GpuVerifyDeviceResult::identity();
    hip_check_error(
        hipMemcpy(result_dev, &result_host, sizeof(GpuVerifyDeviceResult), hipMemcpyHostToDevice));

    // Launch persistent kernel.
    // automatically derive the optimal grid size from the kernel's occupancy and the
    // number of multiprocessors.
    constexpr int block_size = 256;
    const auto kernel        = gpu_verify_kernel<block_size, T, IteratorA, IteratorB>;

    launch_persistent_kernel(kernel,
                             block_size,
                             stream,
                             device_result,
                             reference_result,
                             rtol,
                             atol,
                             static_cast<long long>(size),
                             result_dev);

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
    result.all_zero    = result_host.all_zero == 1;

    return result;
}

// Forward declaration of gpu_reduce_max
template <typename T, typename Iterator>
float gpu_reduce_max(Iterator device_buffer, std::size_t size, hipStream_t stream = nullptr);

// Host-side wrapper for GPU verification with automatic tolerance computation
// Computes max value on GPU, then computes tolerances and verifies
// Returns GpuVerifyResult with detailed error information
template <typename OutDataType,
          typename ComputeDataType = OutDataType,
          typename AccDataType     = ComputeDataType,
          typename IteratorA,
          typename IteratorB>
GpuVerifyResult gpu_verify(IteratorA device_result,
                           IteratorB reference_result,
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

    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
        std::cout << "verify: accumulations=" << number_of_accumulations << " rtol = " << rtol
                  << " atol=" << atol << std::endl;
    }

    // Call the explicit tolerance version
    return gpu_verify<OutDataType>(device_result, reference_result, rtol, atol, size, stream);
}

// GPU reduction kernel for computing max(abs(data))
// This is an internal kernel called only by gpu_reduce_max() wrapper.
template <int BlockSize, typename T, typename Iterator>
__global__ __launch_bounds__((BlockSize)) //
    void gpu_reduce_max_kernel(Iterator it, long long size, float* __restrict__ max_val)
{
    auto data = make_concrete_iterator<T>(it);

    long long idx    = blockIdx.x * BlockSize + threadIdx.x;
    long long stride = BlockSize * gridDim.x;

    float local_max = 0.0f;

    for(long long i = idx; i < size; i += stride)
    {
        float val = fabsf(type_convert<float>(data[i]));
        local_max = fmaxf(local_max, val);
    }

    const auto block_max = block_reduce<BlockSize>(
        local_max, [](const auto& a, const auto& b) { return std::max(a, b); });

    if(threadIdx.x == 0)
        atomicMax(max_val, block_max);
}

// Host-side wrapper for GPU max reduction
// Computes max(abs(data)) and returns as float
// Only transfers 4 bytes (the final max value) instead of entire tensor
template <typename T, typename Iterator>
float gpu_reduce_max(Iterator device_buffer, std::size_t size, hipStream_t stream)
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

    // Launch persistent kernel.
    // automatically derive the optimal grid size from the kernel's occupancy and the
    // number of multiprocessors.
    constexpr int block_size = 256;
    const auto kernel        = gpu_reduce_max_kernel<block_size, T, Iterator>;

    launch_persistent_kernel(
        kernel, block_size, stream, device_buffer, static_cast<long long>(size), max_dev);

    // Copy result to host (only 4 bytes!)
    float max_host;
    hip_check_error(hipMemcpy(&max_host, max_dev, sizeof(float), hipMemcpyDeviceToHost));

    // Free device memory
    hip_check_error(hipFree(max_dev));

    return max_host;
}

} // namespace profiler
} // namespace ck
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
