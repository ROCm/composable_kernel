// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#include "ck_tile/core/arch/inst_prefetch.hpp"

#include <hip/hip_runtime.h>

namespace ck {
namespace s_prefetch_inst_op_util {

template <typename T>
struct KernelArgs
{
    const T* p_a_grid;
    T* dst;
    const T* p_b_grid;
    uint32_t num_iters;
};

// ---------------------------------------------------------------------------
// A simple kernel that exercises INST_PREFETCH / INST_PREFETCH_TARGET macros.
//
// The kernel does: dst[tid] = src[tid] + scalar_sum
//
// Between the prefetch site and the target we place a deliberate computation
// loop so that the prefetched instruction cache lines have time to arrive.
// Correctness does not depend on prefetching — it is pure performance hint.
// We verify correctness to ensure the asm volatile markers do not break
// code generation.
// ---------------------------------------------------------------------------

template <bool prefetch_inst_on, typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
__global__ void kernel_with_inst_prefetch(KernelArgs<T> args)
{
    if constexpr(prefetch_inst_on)
    {
        enable_scalar_prefetch();
        // Prefetch the tail section of this kernel into L1I.
        // We try to load 32 cachelines but gets clamped to smaller number inside if needed, to not
        // go oob
        INST_PREFETCH(INST_TEST_TAIL, 32);
    }

    __builtin_amdgcn_sched_barrier(0);

    const T* src       = args.p_a_grid;
    T* dst             = args.dst;
    uint32_t num_iters = args.num_iters;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;

    if(tid < NUM_THREADS)
    {
        sum += src[tid];
    }

    __builtin_amdgcn_sched_barrier(0);

    // Hot loop — the PLACE target sits after it.
    for(uint32_t iter = 0; iter < num_iters; ++iter)
    {
#pragma unroll NUM_SCALARS
        for(uint32_t i = 0; i < NUM_SCALARS; ++i)
        {
            sum += 1;
        }
    }

    __builtin_amdgcn_sched_barrier(0);

    INST_PREFETCH_TARGET(INST_TEST_TAIL, CK_PLACE_MODE_BLOCK_ENTRY);

// Tail section (the code we prefetched).
#pragma unroll NUM_THREADS
    for(uint32_t i = 0; i < NUM_THREADS; ++i)
    {
        sum += src[0];
    }

    if(tid < NUM_THREADS)
    {
        dst[tid] = sum;
    }
}

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool test_inst_prefetch_impl(bool time_kernels, const std::string& test_name)
{
    constexpr index_t num_elements         = NUM_THREADS;
    constexpr index_t num_scalars          = 1;
    constexpr index_t num_scalar_additions = NUM_SCALARS;
    constexpr index_t block_size           = 256;
    constexpr index_t grid_size            = (NUM_THREADS + block_size - 1) / block_size;
    constexpr uint32_t num_iters =
        0; // always 0 so we jump to the prefetch target right after the barrier.

    std::cout << "Testing " << test_name << " for type: " << typeid(T).name() << std::endl;
    std::cout << "Elements: " << num_elements
              << ", Scalar additions(the one we jump over): " << num_scalar_additions << std::endl;

    // Host data
    std::vector<T> h_src(num_elements);
    std::vector<T> h_scalar(num_scalars);
    std::vector<T> h_dst(NUM_THREADS);
    std::vector<T> h_expected(NUM_THREADS);

    for(index_t i = 0; i < num_elements; i++)
    {
        h_src[i] = static_cast<T>((i % 100) + 1);
    }

    for(index_t i = 0; i < static_cast<index_t>(NUM_THREADS); i++)
    {
        h_expected[i] = h_src[i] + h_src[0] * NUM_THREADS;
    }

    DeviceMem d_src(sizeof(T) * num_elements);
    DeviceMem d_scalar(sizeof(T) * num_scalars);
    DeviceMem d_dst(sizeof(T) * NUM_THREADS);

    d_src.ToDevice(h_src.data());
    d_scalar.ToDevice(h_scalar.data());

    KernelArgs<T> args{static_cast<const T*>(d_src.GetDeviceBuffer()),
                       static_cast<T*>(d_dst.GetDeviceBuffer()),
                       static_cast<const T*>(d_scalar.GetDeviceBuffer()),
                       num_iters};

    auto run_single = [&](auto kernel_fn, const std::string& label) -> std::pair<bool, float> {
        float avg_us = 0;

        if(time_kernels)
        {
            constexpr int num_warmup     = 10;
            constexpr int num_iterations = 50;
            constexpr int rotating_count = num_iterations;
            auto size_a_buffer           = d_src.GetBufferSize();
            auto size_b_buffer           = d_scalar.GetBufferSize();

            ck::utility::RotatingMemWrapper<KernelArgs<T>> rotating_mem(
                args, rotating_count, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                ck::utility::flush_icache();
                rotating_mem.Next();
            };
            float avg_time_ms = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                StreamConfig{nullptr, true, 0, num_warmup, num_iterations, true, rotating_count},
                run_flush_cache,
                kernel_fn,
                dim3(grid_size),
                dim3(block_size),
                0,
                args);

            avg_us = avg_time_ms * 1000.0f;
            std::cout << "  " << label << ": avg " << avg_us << " us" << std::endl;
        }
        else
        {
            launch_and_time_kernel(StreamConfig{nullptr, false},
                                   kernel_fn,
                                   dim3(grid_size),
                                   dim3(block_size),
                                   0,
                                   args);
        }

        d_dst.FromDevice(h_dst.data());
        bool pass = ck::utils::check_err(h_dst, h_expected);
        std::cout << "  " << label << " correctness: " << (pass ? "PASS" : "FAIL") << std::endl;
        return {pass, avg_us};
    };

    bool pass = true;
    auto [pass_pf1, time_pf1] =
        run_single(kernel_with_inst_prefetch<true, T, NUM_THREADS, NUM_SCALARS>, "single_prefetch");
    auto [pass_base, time_base] = run_single(
        kernel_with_inst_prefetch<false, T, NUM_THREADS, NUM_SCALARS>, "no_prefetch (baseline)");

    pass &= pass_base;
    pass &= pass_pf1;

    if(time_kernels && time_base > 0)
    {
        auto pct = [&](float t) { return (t - time_base) / time_base * 100.0f; };
        std::cout << "  --- Performance ---" << std::endl;
        std::cout << "  no_prefetch (baseline):    " << time_base << " us" << std::endl;
        std::cout << "  single_prefetch:           " << time_pf1 << " us (" << pct(time_pf1)
                  << " %)" << std::endl;
    }

    std::cout << std::endl;
    return pass;
}

} // namespace s_prefetch_inst_op_util
} // namespace ck
