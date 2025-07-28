// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <chrono>

namespace ck_tile {

struct gpu_timer
{
    CK_TILE_HOST gpu_timer()
    {
        HIP_CHECK_ERROR(hipEventCreate(&start_evt));
        HIP_CHECK_ERROR(hipEventCreate(&stop_evt));
    }

    CK_TILE_HOST ~gpu_timer() noexcept(false)
    {
        HIP_CHECK_ERROR(hipEventDestroy(start_evt));
        HIP_CHECK_ERROR(hipEventDestroy(stop_evt));
    }

    CK_TILE_HOST void start(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        HIP_CHECK_ERROR(hipEventRecord(start_evt, s));
    }

    CK_TILE_HOST void stop(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipEventRecord(stop_evt, s));
        HIP_CHECK_ERROR(hipEventSynchronize(stop_evt));
    }
    // return in ms
    CK_TILE_HOST float duration() const
    {
        float ms = 0;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }

    private:
    hipEvent_t start_evt, stop_evt;
};

struct gpu_timer_new
{
    CK_TILE_HOST gpu_timer_new(const hipStream_t& s)
    {
        for(auto& e : start_event)
        {
            HIP_CHECK_ERROR(hipEventCreate(&e));
        }
        for(auto& e : stop_event)
        {
            HIP_CHECK_ERROR(hipEventCreate(&e));
        }
        HIP_CHECK_ERROR(hipEventCreate(&event0));
        HIP_CHECK_ERROR(hipEventRecord(event0, s));
    }

    CK_TILE_HOST ~gpu_timer_new() noexcept(false)
    {
        for(auto& e : start_event)
        {
            HIP_CHECK_ERROR(hipEventDestroy(e));
        }
        for(auto& e : stop_event)
        {
            HIP_CHECK_ERROR(hipEventDestroy(e));
        }
        HIP_CHECK_ERROR(hipEventDestroy(event0));
    }

    CK_TILE_HOST void start(const hipStream_t& s, int idx = 0)
    {
        HIP_CHECK_ERROR(hipEventRecord(start_event[idx % 2], s));
    }

    CK_TILE_HOST void stop(const hipStream_t& s, int idx = 0)
    {
        HIP_CHECK_ERROR(hipEventRecord(stop_event[idx % 2], s));
    }
    // return in ms
    CK_TILE_HOST float duration(int idx = 0) const
    {
        float ms;
        HIP_CHECK_ERROR(hipEventSynchronize(stop_event[idx % 2]));
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, start_event[idx % 2], stop_event[idx % 2]));
        return ms;
    }
    CK_TILE_HOST float is_exceed(int idx = 0) const
    {
        float ms;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, event0, stop_event[idx % 2]));
        return ms;
    }

    private:
    std::array<hipEvent_t, 2> start_event;
    std::array<hipEvent_t, 2> stop_event;
    hipEvent_t event0;
};

struct cpu_timer
{
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void start(const hipStream_t& s, [[maybe_unused]] int idx = 0)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        start_tick  = std::chrono::high_resolution_clock::now();
        time_event0 = std::chrono::high_resolution_clock::now();
    }
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void stop(const hipStream_t& s, [[maybe_unused]] int idx = 0)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        stop_tick = std::chrono::high_resolution_clock::now();
    }
    // return in ms
    CK_TILE_HOST float duration([[maybe_unused]] int idx = 0) const
    {
        double sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop_tick - start_tick)
                .count();
        return static_cast<float>(sec * 1e3);
    }
    // return in ms
    CK_TILE_HOST float is_exceed([[maybe_unused]] int idx = 0) const
    {
        double sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop_tick - time_event0)
                .count();
        return static_cast<float>(sec * 1e3);
    }

    private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_tick;
    std::chrono::time_point<std::chrono::high_resolution_clock> time_event0;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_tick;
};

} // namespace ck_tile
