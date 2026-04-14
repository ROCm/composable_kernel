// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"

#include "../get_wave_size_helper.hpp"

template <typename AType_     = ck_tile::fp16_t,
          typename BType_     = ck_tile::fp16_t,
          typename CType_     = ck_tile::fp32_t,
          uint32_t WaveTileM_ = 16,
          uint32_t WaveTileN_ = 16,
          uint32_t WaveTileK_ = 32>
struct MmaPipelineTest
{
    using AType                     = AType_;
    using BType                     = BType_;
    using CType                     = CType_;
    static constexpr auto WaveTileM = WaveTileM_;
    static constexpr auto WaveTileN = WaveTileN_;
    static constexpr auto WaveTileK = WaveTileK_;

    void test_pipeline(std::function<bool(ck_tile::core::arch::amdgcn_target_id)> shouldSkip,
                       std::function<void(uint32_t, void*, void*, void*, void*)> kernel,
                       std::function<CType(uint32_t)> getExpected,
                       std::function<AType(size_t)> aInitializer = nullptr)
    {
        using namespace ck_tile;
        using namespace ck_tile::core::arch;

        int devCount;
        hipDevice_t dev;
        HIP_CHECK_ERROR(hipGetDevice(&dev));
        HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

        hipDeviceProp_t devProp;
        HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

        auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
        bool hasDevice     = static_cast<bool>(devCount > 0);
        int deviceWarpSize = devProp.warpSize;

        if(!hasDevice || shouldSkip(currentArchId))
        {
            GTEST_SKIP() << "No HIP device found. Skipping test.";
        }

        // WaveTile size, also the expected fragment size (MmaTile) from the selector.
        // Note: Actual FragK might be slightly different due to hardware implementation, but the
        // test_accum_over_k kernel will loop over the K dimension to ensure that the total K is
        // correct.
        static constexpr uint32_t FragM = WaveTileM;
        static constexpr uint32_t FragN = WaveTileN;
        static constexpr uint32_t FragK = WaveTileK;

        // The number of elements per thread
        uint32_t AElements = FragM * FragK / deviceWarpSize;
        uint32_t BElements = FragN * FragK / deviceWarpSize;
        uint32_t CElements = FragM * FragN / deviceWarpSize;

        uint32_t ASize = AElements * sizeof(AType);
        uint32_t BSize = BElements * sizeof(BType);
        uint32_t CSize = CElements * sizeof(CType);

        // Initialize A (use custom initializer or default all 1's), B to all 1's, C to all 0's
        std::vector<AType> h_a(AElements);
        if(aInitializer)
        {
            for(size_t i = 0; i < AElements; ++i)
                h_a[i] = aInitializer(i);
        }
        else
        {
            std::fill(h_a.begin(), h_a.end(), type_convert<AType>(1));
        }
        std::vector<BType> h_b(BElements, type_convert<BType>(1));
        std::vector<CType> h_c(CElements, type_convert<CType>(0));
        std::vector<CType> h_out(CElements, type_convert<CType>(0));

        AType* d_a;
        BType* d_b;
        CType* d_c;
        CType* d_out;

        HIP_CHECK_ERROR(hipMalloc(&d_a, ASize));
        HIP_CHECK_ERROR(hipMalloc(&d_b, BSize));
        HIP_CHECK_ERROR(hipMalloc(&d_c, CSize));
        HIP_CHECK_ERROR(hipMalloc(&d_out, CSize));

        // Copy inputs to device
        HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), ASize, hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), BSize, hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpy(d_c, h_c.data(), CSize, hipMemcpyHostToDevice));

        const auto wave_size = getDeviceWaveSize();
        kernel(wave_size, d_a, d_b, d_c, d_out);
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

        // Verify output against expected value for all elements
        for(size_t i = 0; i < CElements; ++i)
        {
            EXPECT_NEAR(h_out[i], getExpected(FragK), 1e-3);
        }

        HIP_CHECK_ERROR(hipFree(d_a));
        HIP_CHECK_ERROR(hipFree(d_b));
        HIP_CHECK_ERROR(hipFree(d_c));
        HIP_CHECK_ERROR(hipFree(d_out));
    }
};
