// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_config.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;

// using F4        = pk_fp4_t;
using F8        = fp8_t;
using BF8       = bf8_t;
using F16       = fp16_t;
using BF16      = bf16_t;
using F32       = fp32_t;
using I8        = int8_t;
using FP8       = fp8_t;
using BF8       = bf8_t;
using I32       = int32_t;
using Target908 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());
using Target942 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX942>());
using Target950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());
using Target11  = decltype(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());
using Target12  = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

// MMA register layout validation test for amdgcn_mma structs.
//
// Strategy: for every (m, k, n) triple in the tile, the test constructs a pair of input tensors A
// and B that contain exactly one non-zero element each, placed so that their product contributes to
// a single output element C(m, n):
//
//         A  (M x K)                B  (K x N)              C = A * B  (M x N)
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . 1 . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . 1 . . . .           . . . . . 1 . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//         A(m,k) = 1                B(k,n) = 1                C(m,n) = 1
//
// The kernel uses TileDistrEncRegMap to scatter A and B into the correct (lane, vecIdx) positions
// of the MMA fragment registers, executes the intrinsic, then uses TileDistrEncRegMap again to
// gather back into C matrix. The position of "1" in C is checked against the expected (m, n)
// location.

/**
 * @class MmaLayoutTestKernel
 * @brief Device kernel that performs C = AB using a given Mma op
 * @tparam MmaOp Intrinsic (amdgcn_mma) to be tested
 */
template <typename MmaOp> // TODO: C++20 concept for MmaOp
struct MmaLayoutTestKernel
{
    static constexpr int kBlockSize = MmaOp::WaveSize;

    __device__ void operator()(uint32_t* error_flags) const
    {
        using ARegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;
        using BRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>;
        using CRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>;

        if constexpr(MmaOpTraits<MmaOp>::IsSupported)
        {
            using AVecType               = typename MmaOp::AVecType;
            using BVecType               = typename MmaOp::BVecType;
            using CVecType               = typename MmaOp::CVecType;
            constexpr index_t a_vec_size = vector_traits<AVecType>::vector_size;
            constexpr index_t b_vec_size = vector_traits<BVecType>::vector_size;
            constexpr index_t c_vec_size = vector_traits<CVecType>::vector_size;

            const index_t lane = threadIdx.x;

            AVecType a_frag{};
            BVecType b_frag{};
            CVecType c_frag{};
            uint32_t sparse_idx{};
            // The actual scale is computed as pow(2, scale - 127), so:
            // 125 -> 2^-2 and 129 -> 2^2.
            int scale_A = 125;
            int scale_B = 129;
            static_assert(MmaOp::kCompressionRatio <= 2); // Allow only 4:2 compression (or no).

            // get (m, k, n), where "1" should be placed for this block
            const index_t case_idx = blockIdx.x;
            const index_t m        = case_idx / (MmaOp::kK * MmaOp::kN);
            const index_t k        = (case_idx / MmaOp::kN) % MmaOp::kK;
            const index_t n        = case_idx % MmaOp::kN;

            // place a single "1" in A/B fragments using (lane, vecIdx) -> (row, col) mapping
            for(index_t v = 0; v < a_vec_size; ++v)
            {
                auto a_coords = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);

                // When dealing with sparse intrinsics, the A matrix is compressed in the K
                // direction and we just put our "1" in the k / 2 position (rounded down).
                if(a_coords[0] == m && a_coords[1] == (k / MmaOp::kCompressionRatio))
                {
                    a_frag[v] = type_convert<typename MmaOp::ADataType>(1.0f);

                    // Calc an appropriate sparse idx value for a single 1 in position k. We use a
                    // baseline index of 0x88888888. This sends each compressed index i to
                    // uncompressed index i * 2. If k is odd, we should send it to i * 2 + 1
                    // instead. We update only the absolutely necessary pair of bits for this
                    // (idx[v*2:v*2+1]). Note that this simple calculation works for any 4:2 sparse
                    // intrinsic with up to 16 packed k elements per lane.
                    sparse_idx = 0x88888888 | ((k % 2) << (v * 2));
                }
            }

            for(index_t v = 0; v < b_vec_size; ++v)
            {
                auto b_coords = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                if(b_coords[0] == n && b_coords[1] == k)
                {
                    b_frag[v] = type_convert<typename MmaOp::BDataType>(1.0f);
                }
            }

            if constexpr(MmaOpTraits<MmaOp>::IsSparse)
            {
                c_frag = MmaOp::exec(a_frag, b_frag, c_frag, sparse_idx);
            }
            else if constexpr(MmaOpTraits<MmaOp>::IsScale)
            {
                c_frag = MmaOp::exec(a_frag, b_frag, c_frag, scale_A, scale_B);
            }
            else
            {
                c_frag = MmaOp::exec(a_frag, b_frag, c_frag);
            }

            // TODO: this tolerance might not be suitable for all data types and
            // should be revisited if we add more configurations
            const float tolerance = 1.0e-1f;
            index_t err           = 0;

            for(index_t v = 0; v < c_vec_size; ++v)
            {
                auto c_coords = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);

                const float expected = (c_coords[0] == m && c_coords[1] == n) ? 1 : 0;
                const float value    = static_cast<float>(c_frag[v]);
                if(std::fabs(value - expected) > tolerance)
                {
                    err = 1;
                }
            }

            const uint32_t any_err = __any(err);
            if(threadIdx.x == 0)
            {
                error_flags[case_idx] = any_err;
            }
        }
    }
};

/**
 * @brief Test driver: runs the test for a given MMA configuration.
 *
 * The testlaunches (mkn) test cases (one per block) to check all possible positions of the "1" in
 * the A/B tensors.
 *   1. Constructs A and B tensors with a single 1 at A(m,k) and B(k,n).
 *   2. Executes MMA intrinsic to compute C tensor.
 *   3. Checks if C has the 1 in the expected position.
 *
 * @tparam MmaOp Intrinsic (amdgcn_mma) to be tested
 */
template <typename MmaOp> // TODO: C++20 concept for MmaOp
void run_mma_layout_test()
{
    EXPECT_TRUE(MmaOpTraits<MmaOp>::IsSupported) << "Unsupported MmaOp! Bad MmaOp in list!\n";

    // Compile-time check that print is instantiable for the given MmaOp
    (void)static_cast<void (*)(MmaOpTraits<MmaOp> const&)>(print);

    int device_count = 0;
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipGetDeviceCount(&device_count));
    EXPECT_TRUE(device_count > 0) << "No device found!";

    hipDeviceProp_t props{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));
    EXPECT_EQ(props.warpSize, static_cast<int>(MmaOp::WaveSize))
        << "Device wavesize " << props.warpSize << " != Mma wavesize " << MmaOp::WaveSize;

    constexpr uint32_t total_cases = MmaOp::kM * MmaOp::kN * MmaOp::kK;
    ck_tile::DeviceMem d_errors(total_cases * sizeof(uint32_t));
    std::vector<uint32_t> h_errors(total_cases, 0u);

    auto* d_error_ptr = static_cast<uint32_t*>(d_errors.GetDeviceBuffer());

    (void)hipGetLastError();

    using Kernel = MmaLayoutTestKernel<MmaOp>;

    ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0, 0, 1},
        ck_tile::make_kernel(Kernel{}, dim3(total_cases), dim3(MmaOp::WaveSize), 0, d_error_ptr));

    HIP_CHECK_ERROR(hipMemcpyAsync(
        h_errors.data(), d_error_ptr, d_errors.GetBufferSize(), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));

    for(uint32_t case_idx = 0; case_idx < total_cases; ++case_idx)
    {
        const uint32_t m = case_idx / (MmaOp::kK * MmaOp::kN);
        const uint32_t k = (case_idx / MmaOp::kN) % MmaOp::kK;
        const uint32_t n = case_idx % MmaOp::kN;

        EXPECT_EQ(h_errors[case_idx], 0u) << "Mismatch for m=" << m << " k=" << k << " n=" << n;
    }
}

// Lists of intrinsics to test.
// clang-format off
using Gfx9Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u,  DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_16x16x16f16
    amdgcn_mma<F16, F16, F32, 64u, 32u, 4u,   DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16, F16, F32, 32u, 64u, 4u,   DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16, F16, F32, 64u, 4u,  4u,   DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_4x4x4f16
    amdgcn_mma<F16, F16, F32, 4u,  64u, 4u,   DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>  // mfma_f32_4x4x4f16
    >;
using Gfx942Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 32u,  DefaultSparseMfmaCtrlFlags,          Target942, MmaOpFamily::SPARSE> // smfmac_f32_16x16x32_f16
>;
using Gfx950Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 32u,  DefaultMfmaCtrlFlags,                Target950, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_f16
    amdgcn_mma<F8,  F8,  F32, 16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<BF8, BF8, F32, 16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    // amdgcn_mma<F4,  F4,  F32, 16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<F8,  F8,  F32, 32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>, // mfma_scale_f32_32x32x64_f8f6f4
    amdgcn_mma<BF8, BF8, F32, 32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>  // mfma_scale_f32_32x32x64_f8f6f4
    // amdgcn_mma<F4,  F4,  F32, 32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           Target950, MmaOpFamily::SCALE>  // mfma_scale_f32_32x32x64_f8f6f4
>;
using Gfx11Intrinsics = ::testing::Types<
    amdgcn_mma<F16,  F16,  F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_f16_w32,
    amdgcn_mma<BF16, BF16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_bf16_w32,
    amdgcn_mma<I8,   I8,   I32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>  // wmma_i32_16x16x16_iu8_w32
>;
using Gfx12Intrinsics = ::testing::Types<
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_f16_w32_gfx12,
    amdgcn_mma<BF16, BF16, F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_bf16_w32_gfx12,
    amdgcn_mma<F16,  F16,  F16,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f16_16x16x16_f16_w32_gfx12,
    amdgcn_mma<BF16, BF16, BF16, 16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_bf16_16x16x16_bf16_w32_gfx12,
    amdgcn_mma<I8,   I8,   I32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_i32_16x16x16_iu8_w32_gfx12,
    amdgcn_mma<FP8,  FP8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_fp8_fp8_w32_gfx12,
    amdgcn_mma<FP8,  BF8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_fp8_bf8_w32_gfx12,
    amdgcn_mma<BF8,  FP8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_bf8_fp8_w32_gfx12,
    amdgcn_mma<BF8,  BF8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_bf8_bf8_w32_gfx12
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 32u, DefaultSparseMfmaCtrlFlags,       Target12,  MmaOpFamily::SPARSE> // swmmac_f32_16x16x32_f16_w32
>;
// clang-format on

template <typename MmaOp>
class TestMmaLayout : public ::testing::Test
{
};
} // namespace
