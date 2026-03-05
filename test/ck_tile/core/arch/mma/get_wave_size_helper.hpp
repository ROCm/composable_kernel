// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>

#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"

namespace {

__global__ void getWaveSizeForSelectedOp(uint32_t* waveSize)
{
    using CompilerTarget = decltype(ck_tile::core::arch::get_compiler_target());

    if(waveSize)
        *waveSize = static_cast<uint32_t>(CompilerTarget::WAVE_SIZE_ID);
}

static __host__ uint32_t getDeviceWaveSize()
{
    uint32_t* d_wave_size;
    HIP_CHECK_ERROR(hipMalloc(&d_wave_size, sizeof(uint32_t)));
    getWaveSizeForSelectedOp<<<1, 64>>>(d_wave_size);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    uint32_t wave_size;
    HIP_CHECK_ERROR(hipMemcpy(&wave_size, d_wave_size, sizeof(uint32_t), hipMemcpyDeviceToHost));
    return wave_size;
}

} // namespace
