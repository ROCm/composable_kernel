// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <iostream>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include "get_wave_size_helper.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

TEST(SparseMMATrait, SparseMfmaGfx950Specialization)
{
    // Test fp16 → fp32 sparse MFMA for GFX950 (16x16x32)
    using TestSparseMfma16x16 = amdgcn_mma<fp16_t,
                                           fp16_t,
                                           fp32_t,
                                           16u,
                                           16u,
                                           32u,
                                           DefaultSparseMfmaCtrlFlags,
                                           CompilerTargetGfx950,
                                           MmaOpFamily::SPARSE>;

    static_assert(std::is_same_v<typename TestSparseMfma16x16::OpType, MfmaOp> &&
                      TestSparseMfma16x16::OpFamily == MmaOpFamily::SPARSE,
                  "GFX950 sparse 16x16x32 should have SparseMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SPARSE, TestSparseMfma16x16>,
                  "GFX950 sparse 16x16x32 should be detected as Sparse");

    std::cout << "GFX950 sparse MFMA specialization is correct" << std::endl;
}

TEST(SparseMMATrait, MmaOpTraitsIntegration)
{
    // Create a sparse MMA op (16x16x32 fp16 specialization)
    using TestSparseMmma = amdgcn_mma<fp16_t,
                                      fp16_t,
                                      fp32_t,
                                      16u,
                                      16u,
                                      32u,
                                      DefaultSparseMfmaCtrlFlags,
                                      CompilerTargetGfx950,
                                      MmaOpFamily::SPARSE>;

    // Get its traits
    using TestTraits = MmaOpTraits<TestSparseMmma>;

    // Verify trait detection
    static_assert(TestTraits::IsSparse, "Sparse MMA should be detected as sparse");
    static_assert(TestTraits::IsSupported, "Sparse MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Sparse MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Sparse MFMA should not be detected as WMMA");

    std::cout << "MmaOpTraits correctly integrates sparse operations" << std::endl;
}

TEST(SparseMMATrait, DenseVsSparseDistinction)
{
    // Dense MFMA from mfma/mfma_gfx9.hpp
    using DenseMfma = amdgcn_mma<fp16_t,
                                 fp16_t,
                                 fp32_t,
                                 16u,
                                 16u,
                                 32u,
                                 DefaultMfmaCtrlFlags,
                                 CompilerTargetGfx950,
                                 MmaOpFamily::DENSE>;

    // Sparse MFMA on GFX950
    using SparseMfma = amdgcn_mma<fp16_t,
                                  fp16_t,
                                  fp32_t,
                                  16u,
                                  16u,
                                  32u,
                                  DefaultSparseMfmaCtrlFlags,
                                  CompilerTargetGfx950,
                                  MmaOpFamily::SPARSE>;

    // Verify they have different operation types
    static_assert(std::is_same_v<typename DenseMfma::OpType, typename SparseMfma::OpType> &&
                      DenseMfma::OpFamily != SparseMfma::OpFamily,
                  "Dense and Sparse MFMA should have the same OpType tags and different OpFamily");

    // Verify traits correctly identify them
    static_assert(MmaOpTraits<DenseMfma>::IsMfma && MmaOpTraits<DenseMfma>::IsDense &&
                      !MmaOpTraits<DenseMfma>::IsSparse && !MmaOpTraits<DenseMfma>::IsScale &&
                      MmaOpTraits<DenseMfma>::IsSupported,
                  "Dense MFMA should be identified correctly");

    static_assert(MmaOpTraits<SparseMfma>::IsSparse && MmaOpTraits<SparseMfma>::IsMfma &&
                      !MmaOpTraits<SparseMfma>::IsDense && !MmaOpTraits<SparseMfma>::IsScale &&
                      MmaOpTraits<SparseMfma>::IsSupported,
                  "Sparse MFMA should be identified correctly");

    std::cout << "Dense and sparse MMA operations are correctly distinguished" << std::endl;
}

TEST(SparseMMATrait, SparseSelector)
{
    static_for<1, 33, 1>{}([](auto i) {
        using Selected = typename MmaDefaultSelector<fp16_t,
                                                     fp16_t,
                                                     fp32_t,
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(i),
                                                     static_cast<uint32_t>(2 * i),
                                                     CompilerTargetGfx950,
                                                     MmaOpFamily::SPARSE>::SelectedOp;

        static constexpr bool isValid = (i == 16) || (i == 32);
        if constexpr(isValid)
        {
            // Selector should pick a sparse MFMA implementation
            static_assert(MmaOpTraits<Selected>::IsSparse);
            static_assert(MmaOpTraits<Selected>::IsMfma);
            static_assert(MmaOpTraits<Selected>::IsSupported);
            static_assert((std::is_same<typename Selected::OpType, MfmaOp>::value));
        }
        else
        {
            // Selector should pick the unsupported pass through
            static_assert(!MmaOpTraits<Selected>::IsSupported);
        }
    });
}

template <typename AType,
          typename BType,
          typename CType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK>
__global__ void test_sparse_accum_over_k(void* a, void* b, void* c, void* out)
{
    using CompilerTarget = decltype(get_compiler_target());
    using Selector       = MmaDefaultSelector<AType,
                                              BType,
                                              CType,
                                              FragM,
                                              FragN,
                                              FragK,
                                              CompilerTarget,
                                              MmaOpFamily::SPARSE>;

    using MmaOp     = typename Selector::SelectedOp;
    using MmaTraits = MmaOpTraits<MmaOp>;

    using CVecType = typename MmaOp::CVecType;

    static constexpr uint32_t kIters = FragK / MmaTraits::BlockK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<typename MmaOp::CVecType*>(c);

    // Accumulate input AxB over FragK/BlockK iterations
    for(uint32_t i = 0; i < kIters; ++i)
    {
        result = MmaOp::exec(*reinterpret_cast<typename MmaOp::AVecType*>(a),
                             *reinterpret_cast<typename MmaOp::BVecType*>(b),
                             result);
    }

    *reinterpret_cast<typename MmaOp::CVecType*>(out) = result;
}

// Live test on real hardware for sparse selection and execution.
TEST(SparseMMATrait, MmaSelector_Sparse_F16_F16_F32_16x16x32_Real)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);
    int deviceWarpSize = devProp.warpSize;

    bool isSupportedWmma = (currentArchId >= amdgcn_target_id::GFX1200) &&
                           (currentArchId <= amdgcn_target_id::GFX12_GENERIC);
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST) ||
       !(isSupportedWmma || isSupportedMfma))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    using AType = fp16_t;
    using BType = fp16_t;
    using CType = fp32_t;

    // Fragment size, also the expected block size from the selector.
    // Note: Actual blockK might be slightly different due to hardware implementation, but the
    // test_accum_over_k kernel will loop over the K dimension to ensure that the total K is
    // correct.
    static constexpr uint32_t FragM  = 16;
    static constexpr uint32_t FragN  = 16;
    static constexpr uint32_t FragK  = 32;
    static constexpr uint32_t BlockM = FragM;
    static constexpr uint32_t BlockN = FragN;
    static constexpr uint32_t BlockK = FragK;

    // The number of elements per thread
    uint32_t AElements = BlockM * BlockK / deviceWarpSize;
    uint32_t BElements = BlockN * BlockK / deviceWarpSize;
    uint32_t CElements = BlockM * BlockN / deviceWarpSize;

    uint32_t ASize = AElements * sizeof(AType);
    uint32_t BSize = BElements * sizeof(BType);
    uint32_t CSize = CElements * sizeof(CType);

    // Initialize A and B to all 1's, C to all 0's
    std::vector<AType> h_a(AElements, static_cast<AType>(1));
    std::vector<BType> h_b(BElements, static_cast<BType>(1));
    std::vector<CType> h_c(CElements, static_cast<CType>(0));
    std::vector<CType> h_out(CElements, static_cast<CType>(0));

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
    test_sparse_accum_over_k<AType, BType, CType, FragM, FragN, FragK>
        <<<1, wave_size>>>(d_a, d_b, d_c, d_out);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

    // Output should be FragK for all elements, because the inputs are all 1's
    for(size_t i = 0; i < CElements; ++i)
    {
        // In sparse only half of the A values are non-zero, thus the /2.
        CType expected = static_cast<CType>(FragK) / 2;

        EXPECT_NEAR(h_out[i], expected, 1e-3);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}
