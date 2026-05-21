// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/scale/mfma/scale_gfx9.hpp"

#include <cstdio>
#include <vector>
#include <cmath>

namespace {

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;

using I4        = pk_int4_t;
using F4        = pk_fp4_t;
using F6        = pk_fp6x16_t;
using BF6       = pk_bf6x16_t;
using F8        = fp8_t;
using BF8       = bf8_t;
using I32       = int32_t;
using F16       = fp16_t;
using BF16      = bf16_t;
using F32       = fp32_t;
using F64       = fp64_t;
using I8        = int8_t;
using F8        = fp8_t;
using I32       = int32_t;
using Target908 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());
using Target90a = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX90A>());
using Target942 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX942>());
using Target950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());
using Target11  = decltype(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());
using Target12  = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());
#if defined(CK_MMA_TEST_ARCH_GFX950)
using TestTarget = Target950;
#elif defined(CK_MMA_TEST_ARCH_GFX942)
using TestTarget = Target942;
#elif defined(CK_MMA_TEST_ARCH_GFX90A)
using TestTarget = Target90a;
#else
using TestTarget = Target908;
#endif

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

// Helper function to place a single "1" value in any datatype. For packed types, "off"
// indicates the offset at which the "1" must be placed within the pack. The packed datatypes
// don't always have utilities to convert "1" to their type so we use raw bits instead.
template <typename T>
CK_TILE_DEVICE T packed_one(int off = 0)
{
    static_assert(numeric_traits<T>::PackedSize == 1, "You need a new packed_one() overload.");
    if(off != 0)
    {
        printf("\033[31mBad pack offset %d for non-packed type, returning 0!\n\033[0m", off);
        return type_convert<T>(0.0f);
    }
    return type_convert<T>(1.0f);
}

template <>
CK_TILE_DEVICE pk_int4_t packed_one(int off)
{
    if(off >= 2)
    {
        printf("\033[31mBad pack offset %d for pk_int4_t, returning 0!\n\033[0m", off);
        return {};
    }
    return pk_int4_t(off == 0 ? 0b00000001 : 0b00010000); // pk_int4_t has no type convert...
}

template <>
CK_TILE_DEVICE pk_fp4_t packed_one(int off)
{
    if(off >= 2)
    {
        printf("\033[31mBad pack offset %d for pk_fp4_t, returning 0!\n\033[0m", off);
        return {};
    }
    return type_convert<pk_fp4_t>(off == 0 ? fp32x2_t{1.0f, 0.0f} : fp32x2_t{0.0f, 1.0f});
}

template <>
CK_TILE_DEVICE pk_fp6x16_t packed_one(int off)
{
    if(off >= 16)
    {
        printf("\033[31mBad pack offset %d for pk_fp6x16_t, returning 0!\n\033[0m", off);
        return pk_fp6x16_t{0};
    }
    pk_fp6x16_t r(0);
    r.pack(0b001000, off); // pk_fp6x16_t has no type convert...
    return r;
}

template <>
CK_TILE_DEVICE pk_bf6x16_t packed_one(int off)
{
    if(off >= 16)
    {
        printf("\033[31mBad pack offset %d for pk_bf6x16_t, returning 0!\n\033[0m", off);
        return pk_bf6x16_t{0};
    }
    pk_bf6x16_t r(0);
    r.pack(0b001100, off); // pk_bf6x16_t has no type convert...
    return r;
}

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
            static_assert(MmaOp::kCompressionRatio <= 2); // Allow only 4:2 compression (or no).

            // get (m, k, n), where "1" should be placed for this block
            const index_t case_idx = blockIdx.x;
            const index_t m        = case_idx / (MmaOp::kK * MmaOp::kN);
            const index_t k        = (case_idx / MmaOp::kN) % MmaOp::kK;
            const index_t n        = case_idx % MmaOp::kN;

            // place a single "1" in A/B fragments using (lane, vecIdx) -> (row, col) mapping
            for(index_t v = 0; v < a_vec_size; ++v)
            {
                // In case of packed data types or sparse intrinsics, the calculated matrix indices
                // are in terms of the matrix of packed (AND/OR sparse compressed) elements. We
                // check if the compressed element corresponds with our targeted value element, and
                // calculate the offset (remainder) within our packed element to get to the value
                // element.
                auto a_coords         = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
                index_t k_target      = k / (MmaOp::kCompressionRatio * MmaOp::APackedSize);
                index_t k_rem         = k % (MmaOp::kCompressionRatio * MmaOp::APackedSize);
                index_t k_rem_pack    = k_rem / MmaOp::kCompressionRatio;
                uint32_t k_rem_sparse = k_rem % MmaOp::kCompressionRatio;

                if(a_coords[0] == m && a_coords[1] == k_target)
                {
                    a_frag[v] = packed_one<typename MmaOp::ADataType>(k_rem_pack);

                    // Calc an appropriate sparse idx value for a single 1 in position k. We use a
                    // baseline index of 0x88888888. This sends each compressed index i to
                    // uncompressed index i * 2. If k is odd, we should send it to i * 2 + 1
                    // instead. We update only the absolutely necessary pair of bits for this
                    // (idx[v*2:v*2+1]), where v is meant to be the idx of the *value items* in the
                    // lane. Note that this simple calculation works for any 4:2 sparse intrinsic
                    // with up to 16 packed k elements per lane.

                    // Calc vector item idx as if the datatype were not packed (value item).
                    uint32_t vec_value_item_idx = v * MmaOp::APackedSize + k_rem_pack;
                    sparse_idx = 0x88888888 | (k_rem_sparse << (vec_value_item_idx * 2));
                }
            }

            for(index_t v = 0; v < b_vec_size; ++v)
            {
                // Same considerations as for A (but no compression), see comment above.
                auto b_coords      = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                index_t k_target   = k / (MmaOp::BPackedSize);
                index_t k_rem_pack = k % (MmaOp::BPackedSize);

                if(b_coords[0] == n && b_coords[1] == k_target)
                {
                    b_frag[v] = packed_one<typename MmaOp::BDataType>(k_rem_pack);
                }
            }

            if constexpr(MmaOpTraits<MmaOp>::IsSparse)
            {
                // Sparse_idx is calculated (overwriting default 0 value) in *exactly* one lane
                // (assert), but must be broadcast to all (warp-level "or" reduction intrinsic).
                int num_sparse_idx_calced = __popcll(__ballot(sparse_idx != 0));
                if(num_sparse_idx_calced != 1)
                {
                    if(threadIdx.x == 0)
                    {
                        printf("There should be exactly 1 lane producing a sparse idx (got %d)!\n",
                               num_sparse_idx_calced);
                    }
                    __builtin_trap();
                }
                sparse_idx = __reduce_or_sync(__activemask(), sparse_idx);
                c_frag     = MmaOp::exec(a_frag, b_frag, c_frag, sparse_idx);
            }
            else if constexpr(MmaOpTraits<MmaOp>::IsScale)
            {
                // The actual scale is computed as pow(2, scale - 127), so:
                // 125 -> 2^-2 and 129 -> 2^2.
                int scale_A = 125;
                int scale_B = 129;
                c_frag      = MmaOp::exec(a_frag, b_frag, c_frag, scale_A, scale_B);
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
                const float value    = type_convert<float>(c_frag[v]);
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

// available on all gfx9 (gfx908, gfx90a, gfx942, gfx950)
using Gfx9CommonIntrinsics = ::testing::Types<
    amdgcn_mma<F32,  F32,  F32,  32u, 64u, 1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x1f32
    amdgcn_mma<F32,  F32,  F32,  64u, 32u, 1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x1f32
    amdgcn_mma<F32,  F32,  F32,  16u, 64u, 1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x1f32
    amdgcn_mma<F32,  F32,  F32,  64u, 16u, 1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x1f32
    amdgcn_mma<F32,  F32,  F32,  4u,  64u, 1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x1f32
    amdgcn_mma<F32,  F32,  F32,  64u, 4u,  1u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x1f32
    amdgcn_mma<F32,  F32,  F32,  32u, 32u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x2f32
    amdgcn_mma<F32,  F32,  F32,  16u, 16u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x4f32
    amdgcn_mma<F16,  F16,  F32,  32u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16,  F16,  F32,  64u, 32u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16,  F16,  F32,  16u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x4f16
    amdgcn_mma<F16,  F16,  F32,  64u, 16u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x4f16
    amdgcn_mma<F16,  F16,  F32,  4u,  64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x4f16
    amdgcn_mma<F16,  F16,  F32,  64u, 4u,  4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x4f16
    amdgcn_mma<F16,  F16,  F32,  32u, 32u, 8u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x8f16
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x16f16
    amdgcn_mma<I8,   I8,   I32,  32u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_32x32x4i8
    amdgcn_mma<I8,   I8,   I32,  64u, 32u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_32x32x4i8
    amdgcn_mma<I8,   I8,   I32,  16u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_16x16x4i8
    amdgcn_mma<I8,   I8,   I32,  64u, 16u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_16x16x4i8
    amdgcn_mma<I8,   I8,   I32,  4u,  64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_4x4x4i8
    amdgcn_mma<I8,   I8,   I32,  64u, 4u,  4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>  // mfma_i32_4x4x4i8
>;
using Gfx908andGfx90aIntrinsics = ::testing::Types<
    amdgcn_mma<BF16, BF16, F32,  32u, 64u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x2bf16
    amdgcn_mma<BF16, BF16, F32,  64u, 32u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x2bf16
    amdgcn_mma<BF16, BF16, F32,  16u, 64u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x2bf16
    amdgcn_mma<BF16, BF16, F32,  64u, 16u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x2bf16
    amdgcn_mma<BF16, BF16, F32,  4u,  64u, 2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x2bf16
    amdgcn_mma<BF16, BF16, F32,  64u, 4u,  2u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x2bf16
    amdgcn_mma<BF16, BF16, F32,  32u, 32u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x4bf16
    amdgcn_mma<BF16, BF16, F32,  16u, 16u, 8u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x8bf16
    amdgcn_mma<I8,   I8,   I32,  32u, 32u, 8u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_32x32x8i8
    amdgcn_mma<I8,   I8,   I32,  16u, 16u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>  // mfma_i32_16x16x16i8
>;
using Gfx90aAndHigherIntrinsics = ::testing::Types<
    amdgcn_mma<BF16, BF16, F32,  32u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  64u, 32u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  16u, 64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  64u, 16u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  4u,  64u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  64u, 4u,  4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_4x4x4bf16_1k
    amdgcn_mma<BF16, BF16, F32,  32u, 32u, 8u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x8bf16_1k
    amdgcn_mma<BF16, BF16, F32,  16u, 16u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x16bf16_1k
    amdgcn_mma<F64,  F64,  F64,  16u, 16u, 4u,   DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>  // mfma_f64_16x16x4f64
>;
using Gfx942AndHigherIntrinsics = ::testing::Types<
    amdgcn_mma<I8,   I8,   I32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_16x16x32_i8
    amdgcn_mma<I8,   I8,   I32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_32x32x16_i8
    amdgcn_mma<BF8,  BF8,  F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_bf8_bf8
    amdgcn_mma<BF8,  F8,   F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_bf8_fp8
    amdgcn_mma<F8,   BF8,  F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_fp8_bf8
    amdgcn_mma<F8,   F8,   F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_fp8_fp8
    amdgcn_mma<BF8,  BF8,  F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_bf8_bf8
    amdgcn_mma<BF8,  F8,   F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_bf8_fp8
    amdgcn_mma<F8,   BF8,  F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_fp8_bf8
    amdgcn_mma<F8,   F8,   F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_fp8_fp8
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 32u,  DefaultSparseMfmaCtrlFlags,          TestTarget, MmaOpFamily::SPARSE> // smfmac_f32_16x16x32_f16
>;
using Gfx950Intrinsics = ::testing::Types<
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_f16
    amdgcn_mma<BF16, BF16, F32,  16u, 16u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_bf16
    amdgcn_mma<F16,  F16,  F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_f16
    amdgcn_mma<BF16, BF16, F32,  32u, 32u, 16u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_f32_32x32x16_bf16
    amdgcn_mma<I8,   I8,   I32,  16u, 16u, 64u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_16x16x64_i8
    amdgcn_mma<I8,   I8,   I32,  32u, 32u, 32u,  DefaultMfmaCtrlFlags,                TestTarget, MmaOpFamily::DENSE>, // mfma_i32_32x32x32_i8
    amdgcn_mma<F8,   F8,   F32,  16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<BF8,  BF8,  F32,  16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<F4,   F4,   F32,  16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<F6,   F6,   F32,  16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<BF6,  BF6,  F32,  16u, 16u, 128u, DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_16x16x128_f8f6f4
    amdgcn_mma<F8,   F8,   F32,  32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_32x32x64_f8f6f4
    amdgcn_mma<BF8,  BF8,  F32,  32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_32x32x64_f8f6f4
    amdgcn_mma<F4,   F4,   F32,  32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_32x32x64_f8f6f4
    amdgcn_mma<F6,   F6,   F32,  32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>, // mfma_scale_f32_32x32x64_f8f6f4
    amdgcn_mma<BF6,  BF6,  F32,  32u, 32u, 64u,  DefaultScaleMfmaCtrlFlags,           TestTarget, MmaOpFamily::SCALE>  // mfma_scale_f32_32x32x64_f8f6f4
>;
using Gfx11Intrinsics = ::testing::Types<
    amdgcn_mma<F16,  F16,  F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_f16_w32
    amdgcn_mma<BF16, BF16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_bf16_w32
    amdgcn_mma<I8,   I8,   I32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>,  // wmma_i32_16x16x16_iu8_w32
    amdgcn_mma<I4,   I4,   I32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,              Target11,  MmaOpFamily::DENSE>   // wmma_i32_16x16x16_iu4_w32
>;
using Gfx12Intrinsics = ::testing::Types<
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_f16_w32_gfx12
    amdgcn_mma<BF16, BF16, F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_bf16_w32_gfx12
    amdgcn_mma<F16,  F16,  F16,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f16_16x16x16_f16_w32_gfx12
    amdgcn_mma<BF16, BF16, BF16, 16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_bf16_16x16x16_bf16_w32_gfx12
    amdgcn_mma<I8,   I8,   I32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_i32_16x16x16_iu8_w32_gfx12
    amdgcn_mma<F8,   F8,   F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_fp8_fp8_w32_gfx12
    amdgcn_mma<F8,   BF8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_fp8_bf8_w32_gfx12
    amdgcn_mma<BF8,  F8,   F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_bf8_fp8_w32_gfx12
    amdgcn_mma<BF8,  BF8,  F32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_f32_16x16x16_bf8_bf8_w32_gfx12
    amdgcn_mma<I4,   I4,   I32,  16u, 16u, 16u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_i32_16x16x16_iu4_w32_gfx12
    amdgcn_mma<I4,   I4,   I32,  16u, 16u, 32u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::DENSE>,  // wmma_i32_16x16x32_iu4_w32_gfx12
    amdgcn_mma<F16,  F16,  F32,  16u, 16u, 32u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::SPARSE>, // swmmac_f32_16x16x32_f16_w32
    amdgcn_mma<I4,   I4,   I32,  16u, 16u, 32u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::SPARSE>, // swmmac_i32_16x16x32_iu4_w32
    amdgcn_mma<I4,   I4,   I32,  16u, 16u, 64u, DefaultWmmaCtrlFlags,             Target12,  MmaOpFamily::SPARSE>  // swmmac_i32_16x16x64_iu4_w32
>;
// clang-format on

template <typename MmaOp>
class TestMmaLayout : public ::testing::Test
{
};
} // namespace
