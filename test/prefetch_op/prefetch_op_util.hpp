// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/common_header.hpp"

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#include <hip/hip_runtime.h>

#include "ck/utility/data_cache_prefetch.hpp"

namespace ck {
namespace prefetch_op_util {

template <typename T>
struct KernelArgs
{
    const T* p_a_grid;
    T* dst;
    const T* p_b_grid;
    bool enable_prefetch;
};

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, typename PrefetchOp>
__global__ void kernel_with_prefetch(KernelArgs<T> args)
{
    const T* src         = args.p_a_grid;
    T* dst               = args.dst;
    const T* scalar_data = args.p_b_grid;
    bool enable_prefetch = args.enable_prefetch;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate number of 32B cachelines needed to cover num_scalars elements
    constexpr index_t cachelineSize              = 32;
    constexpr index_t elements_per_cachelineSize = cachelineSize / sizeof(T);
    constexpr unsigned int cachelinesNeeded =
        (NUM_SCALARS + elements_per_cachelineSize - 1) / elements_per_cachelineSize;

    const char* byte_addr = reinterpret_cast<const char*>(scalar_data);

    // Prefetch all scalar data at once
    if(tid < cachelinesNeeded)
    {
        if(enable_prefetch)
        {
            // Prefetch the cacheline
            PrefetchOp{}(byte_addr + tid * cachelineSize);
        }
    }

    T sum = 0;
    if(tid < NUM_THREADS)
    {
        sum = src[tid]; // load from global mem to give time for prefetch to finish or be close to
                        // finish
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < NUM_THREADS)
    {
        // Access prefetched scalar data
        for(uint32_t i = 0; i < NUM_SCALARS; i++)
        {
            sum += scalar_data[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, typename PrefetchOp>
__global__ void kernel_with_prefetch_and_shared_mem(KernelArgs<T> args)
{
    const T* src         = args.p_a_grid;
    T* dst               = args.dst;
    const T* scalar_data = args.p_b_grid;
    bool enable_prefetch = args.enable_prefetch;

    __shared__ T sharedMem[32];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate number of 32B cachelines needed to cover num_scalars elements
    constexpr index_t cachelineSize              = 32;
    constexpr index_t elements_per_cachelineSize = cachelineSize / sizeof(T);
    constexpr unsigned int cachelinesNeeded =
        (NUM_SCALARS + elements_per_cachelineSize - 1) / elements_per_cachelineSize;

    bool use_shared_mem = tid % 2 == 1;

    const void* byte_addr;
    if(use_shared_mem)
    {
        byte_addr = reinterpret_cast<const void*>(sharedMem);
    }
    else
    {
        uintptr_t base   = reinterpret_cast<uintptr_t>(scalar_data);
        uintptr_t offset = base + (tid / 2) * cachelineSize;
        byte_addr        = reinterpret_cast<const void*>(offset);
    }

    // Prefetch all scalar data at once
    if(tid < cachelinesNeeded * 2)
    {
        if(enable_prefetch)
        {
            // Prefetch the cacheline
            PrefetchOp{}(byte_addr);
        }
        else
        {
            (void)byte_addr;
        }
    }

    T sum = 0;
    if(tid < NUM_THREADS)
    {
        sum = src[tid]; // load from global mem to give time for prefetch to finish or be close to
                        // finish
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < NUM_THREADS)
    {
        // Access prefetched scalar data
        for(uint32_t i = 0; i < NUM_SCALARS; i++)
        {
            sum += scalar_data[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename PrefetchKernel, typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool test_prefetch_impl(bool time_kernels,
                        const PrefetchKernel& prefetch_kernel,
                        const std::string& kernel_name)
{
    constexpr index_t block_size   = 256;
    constexpr index_t num_elements = NUM_THREADS;
    constexpr index_t num_scalars  = NUM_SCALARS;

    // TODO: maybe add more prefetch instructions inside kernel to support more values
    assert(NUM_SCALARS / sizeof(T) < (32 * block_size) &&
           "Too many scalars to prefetch with current implementation!");

    constexpr index_t grid_size = (num_elements + block_size - 1) / block_size;

    std::cout << "Testing " << kernel_name << " for type: " << typeid(T).name() << std::endl;
    std::cout << "Elements: " << num_elements << ", Scalars: " << num_scalars << std::endl;

    // Host data
    std::vector<T> h_src(num_elements);
    std::vector<T> h_scalar(num_scalars);
    std::vector<T> h_dst_with_prefetch_chunks(num_elements);
    std::vector<T> h_expected(num_elements);

    // Initialize data
    for(index_t i = 0; i < num_elements; i++)
    {
        h_src[i] = static_cast<T>(i % 100);
    }

    T scalar_sum = 0;
    for(index_t i = 0; i < num_scalars; i++)
    {
        h_scalar[i] = static_cast<T>(i + 1);
        scalar_sum += h_scalar[i];
    }

    // Expected results
    for(index_t i = 0; i < num_elements; i++)
    {
        h_expected[i] = h_src[i] + scalar_sum;
    }

    // Device memory
    DeviceMem d_src(sizeof(T) * num_elements);
    DeviceMem d_scalar(sizeof(T) * num_scalars);
    DeviceMem d_dst_with_prefetch_chunks(sizeof(T) * num_elements);

    d_src.ToDevice(h_src.data());
    d_scalar.ToDevice(h_scalar.data());

    KernelArgs<T> args{static_cast<const T*>(d_src.GetDeviceBuffer()),
                       static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
                       static_cast<const T*>(d_scalar.GetDeviceBuffer()),
                       true};
    if(time_kernels)
    {
        std::array<float, 2> avg_times_us;
        ck::static_for<0, 2, 1>{}([&](auto static_i) {
            constexpr bool prefetch_enabled = static_i == 0;
            std::cout << "PREFETCH " << (prefetch_enabled ? "ENABLED!" : "DISABLED!") << std::endl;

            args.enable_prefetch = prefetch_enabled;

            constexpr int num_warmup     = 1;
            constexpr int num_iterations = 10;
            constexpr int rotating_count = num_iterations;
            auto size_a_buffer           = d_src.GetBufferSize();
            auto size_b_buffer           = d_scalar.GetBufferSize();

            ck::utility::RotatingMemWrapper<KernelArgs<T>> rotating_mem(
                args, rotating_count, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck::utility::flush_icache();
                // rotating mem
                rotating_mem.Next();
            };
            float avg_time_ms = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                StreamConfig{nullptr, true, 0, num_warmup, num_iterations, true, rotating_count},
                run_flush_cache,
                prefetch_kernel,
                dim3(grid_size),
                dim3(block_size),
                0,
                args);

            float avg_time_us       = avg_time_ms * 1000.0f;
            float total_bytes       = (size_a_buffer + size_b_buffer); // read
            float bandwidth_gb_s    = (total_bytes / (avg_time_us * 1e-6)) / 1e9;
            float ops_per_iteration = num_elements * num_scalars; // adds
            float gflops            = (ops_per_iteration / (avg_time_us * 1e-6)) / 1e9;

            std::cout << "  Performance: " << std::endl;
            std::cout << "    Average kernel time: " << avg_time_us << " us" << std::endl;
            std::cout << "    Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
            std::cout << "    Compute throughput: " << gflops << " GFLOPS" << std::endl;

            avg_times_us[static_i] = avg_time_us;
        });

        float speedup = avg_times_us[1] / avg_times_us[0];

        std::cout << "On average kernel with prefetch is " << speedup
                  << " times faster than without prefetch." << std::endl;

        if(speedup < 1.0f)
            std::cout << "WARNING: prefetch kernel is slower!" << std::endl;
    }
    else
    {
        launch_and_time_kernel(StreamConfig{nullptr, false},
                               prefetch_kernel,
                               dim3(grid_size),
                               dim3(block_size),
                               0, // lds_byte
                               args);
    }

    // Copy results back
    d_dst_with_prefetch_chunks.FromDevice(h_dst_with_prefetch_chunks.data());

    // Verify results
    bool pass = ck::utils::check_err(h_dst_with_prefetch_chunks, h_expected);

    std::cout << "  Correctness: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    return pass;
}

} // namespace prefetch_op_util
} // namespace ck
