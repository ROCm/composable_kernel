// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/high_res_cpu_clock.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>

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

struct cpu_timer
{
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void start(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        start_tick = high_res_now();
    }
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void stop(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        stop_tick = high_res_now();
    }
    // return in ms
    CK_TILE_HOST float duration() const
    {
        auto us = duration_us(start_tick, stop_tick);
        return static_cast<float>(us) / 1e3;
    }

    private:
    timepoint_t start_tick;
    timepoint_t stop_tick;
};

} // namespace ck_tile
