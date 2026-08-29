// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/data_type.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {

// WMMA scale instructions for this test
enum class WMMA_SCALE
{
    SCALE_F32_16x16x128 = static_cast<int>(
        MfmaInstr::wmma_scale_f32_16x16x128_f8f6f4_gfx125), // V_WMMA_SCALE_F32_16X16X128_F8F6F4
    SCALE16_F32_16x16x128 = static_cast<int>(
        MfmaInstr::wmma_scale16_f32_16x16x128_f8f6f4_gfx125), // V_WMMA_SCALE16_F32_16X16X128_F8F6F4
};

template <typename VecT>
static constexpr int32_t vectorSize(const VecT&)
{
    return scalar_type<VecT>::vector_size;
}

// Load functions for WMMA scale operations
// These are similar to MFMA load functions but adapted for WMMA layout

template <typename AType, typename AFragT, int32_t BLOCK_M, int32_t BLOCK_K>
__device__ AFragT load_A_row_major(AType const* input_ptr)
{
    // clang-format off

    // Register Mapping for 16x128 for FP8:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K15  |    K16-K31  |
    // Reg 4 - 7         |    K32-K47 |    K48-K63  |
    // Reg 8 - 11        |    K64-K79 |    K80-K95  |
    // Reg 12 - 15       |    K96-K111|    K112-K127|

    // Register Mapping for 16x128 for FP6:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |

    // Register Mapping for 16x128 for FP4:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 32; // WMMA uses wave32

    // FP8 chunk_size = 16, num_chunks = 4, packed_size = 1
    // FP4 chunk_size = 32, num_chunks = 2, packed_size = 2
    // FP6 chunk_size = 32, num_chunks = 2, packed_size = 32

    constexpr index_t num_chunks = is_packed_type_v<AType> ? 2 : 4;

    constexpr bool is_single_rate = ((BLOCK_K / WAVE_SIZE) > 2) ? false : true;
    constexpr uint32_t chunk_size = is_single_rate ? (is_packed_type_v<AType> ? 16u : 8u)
                                                   : (is_packed_type_v<AType> ? 32u : 16u);

    // each chunk is separated by offset (for K)
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_M; // 64 or 32

    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M, (threadIdx.x / BLOCK_M) * chunk_size);
    auto majorStepCoord2D = std::make_pair(0, chunk_offset);

    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    using ARawT = typename scalar_type<AFragT>::type;
    using AScalarChunkT =
        typename vector_type<ARawT, scalar_type<AFragT>::vector_size / (num_chunks)>::type;

    union
    {
        AFragT frag;
        AScalarChunkT chunks[num_chunks];
    } fragA{};

    const AScalarChunkT* fragPtr;

    auto startOffset  = row_major(startCoord2D, BLOCK_K) / packed_size_v<AType>;
    auto kMajorOffset = row_major(majorStepCoord2D, BLOCK_K) / packed_size_v<AType>;

    for(index_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
    {
        fragPtr                 = reinterpret_cast<AScalarChunkT const*>(input_ptr + startOffset +
                                                         chunk_idx * kMajorOffset);
        fragA.chunks[chunk_idx] = *fragPtr;
    }

    return fragA.frag;
}

template <typename BType, typename BFragT, int32_t BLOCK_K, int32_t BLOCK_N>
__device__ BFragT load_B_col_major(BType const* input_ptr)
{
    // clang-format off

    // Register Mapping for 16x128 for FP8:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K15  |    K16-K31  |
    // Reg 4 - 7         |    K32-K47 |    K48-K63  |
    // Reg 8 - 11        |    K64-K79 |    K80-K95  |
    // Reg 12 - 15       |    K96-K111|    K112-K127|

    // Register Mapping for 16x128 for FP6:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |

    // Register Mapping for 16x128 for FP4:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 32;

    // FP8 chunk_size = 16, num_chunks = 4, packed_size = 1
    // FP4 chunk_size = 32, num_chunks = 2, packed_size = 2
    // FP6 chunk_size = 32, num_chunks = 2, packed_size = 32

    constexpr index_t num_chunks = is_packed_type_v<BType> ? 2 : 4;

    // Use is_single_rate to control 16x64 vs 16x128 instruction variants
    constexpr bool is_single_rate = ((BLOCK_K / WAVE_SIZE) > 2) ? false : true;
    constexpr uint32_t chunk_size = is_single_rate ? (is_packed_type_v<BType> ? 16u : 8u)
                                                   : (is_packed_type_v<BType> ? 32u : 16u);

    // each chunk is separated by an offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_N; // 64 or 32

    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N) * chunk_size, threadIdx.x % BLOCK_N);

    auto majorStepCoord2D = std::make_pair(chunk_offset, 0);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    using BRawT = typename scalar_type<BFragT>::type;
    using BScalarChunkT =
        typename vector_type<BRawT, scalar_type<BFragT>::vector_size / num_chunks>::type;

    union
    {
        BFragT frag;
        BScalarChunkT chunks[num_chunks];
    } fragB{};

    const BScalarChunkT* fragPtr;

    auto startOffset  = col_major(startCoord2D, BLOCK_K) / packed_size_v<BType>;
    auto kMajorOffset = col_major(majorStepCoord2D, BLOCK_K) / packed_size_v<BType>;

    for(index_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
    {
        fragPtr                 = reinterpret_cast<BScalarChunkT const*>(input_ptr + startOffset +
                                                         chunk_idx * kMajorOffset);
        fragB.chunks[chunk_idx] = *fragPtr;
    }

    return fragB.frag;
}

template <typename AType,
          typename AFragT,
          typename ScaleType,
          typename ScaleFragT,
          int32_t BLOCK_M,
          int32_t BLOCK_K,
          int32_t BLOCK_X>
__device__ AFragT load_mx_A_row_major(AType const* input_ptr,
                                      ScaleType const* scale_ptr,
                                      ScaleFragT& fragX)
{
    // clang-format off

    // Register Mapping for 16x128 for FP8, scale block size 32:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |   K0-K15   |   K16-K31   |
    // Reg 4 - 7         |   K32-K47  |   K48-K63   |
    // Reg 8 - 11        |   K64-K79  |   K80-K95   |
    // Reg 12 - 15       |   K96-K111 |   K112-K127 |
    // Reg 16            | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP8, scale block size 16:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |   K0-K15   |   K16-K31   |
    // Reg 4 - 7         |   K32-K47  |   K48-K63   |
    // Reg 8 - 11        |   K64-K79  |   K80-K95   |
    // Reg 12 - 15       |   K96-K111 |   K112-K127 |
    // Reg 16 - 17       | Scale[0-7] |  Scale[0-7] |

    // Register Mapping for 16x128 for FP6, scale block size 32:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |
    // Reg 12            | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP6, scale block size 16:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |
    // Reg 12 - 13       | Scale[0-7] |  Scale[0-7] |

    // Register Mapping for 16x128 for FP4, scale block size 32:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |
    // Reg 8             | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP4, scale block size 16:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |
    // Reg 8 - 9         | Scale[0-7] |  Scale[0-7] |

    // clang-format on

    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M, (threadIdx.x / BLOCK_M));

    index_t startOffset = startCoord2D.first * (BLOCK_K / BLOCK_X);

    if(threadIdx.x >= 16)
    {
        auto& scale_vec = fragX.template AsType<ScaleType>();
        static_for<0, scalar_type<ScaleFragT>::vector_size, 1>{}(
            [&](auto i) { scale_vec(Number<i.value>{}) = scale_ptr[startOffset + i.value]; });
    }
    return load_A_row_major<AType, AFragT, BLOCK_M, BLOCK_K>(input_ptr);
}

template <typename BType,
          typename BFragT,
          typename ScaleType,
          typename ScaleFragT,
          int32_t BLOCK_K,
          int32_t BLOCK_N,
          int32_t BLOCK_X>
__device__ BFragT load_mx_B_col_major(BType const* input_ptr,
                                      ScaleType const* scale_ptr,
                                      ScaleFragT& fragX)
{
    // clang-format off

    // Register Mapping for 16x128 for FP8, scale block size 32:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |   K0-K15   |   K16-K31   |
    // Reg 4 - 7         |   K32-K47  |   K48-K63   |
    // Reg 8 - 11        |   K64-K79  |   K80-K95   |
    // Reg 12 - 15       |   K96-K111 |   K112-K127 |
    // Reg 16            | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP8, scale block size 16:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |   K0-K15   |   K16-K31   |
    // Reg 4 - 7         |   K32-K47  |   K48-K63   |
    // Reg 8 - 11        |   K64-K79  |   K80-K95   |
    // Reg 12 - 15       |   K96-K111 |   K112-K127 |
    // Reg 16 - 17       | Scale[0-7] |  Scale[0-7] |

    // Register Mapping for 16x128 for FP6, scale block size 32:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |
    // Reg 12            | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP6, scale block size 16:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 5         |    K0-K31  |    K32-K63  |
    // Reg 6 - 11        |    K64-K95 |    K96-K127 |
    // Reg 12 - 13       | Scale[0-7] |  Scale[0-7] |

    // Register Mapping for 16x128 for FP4, scale block size 32:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |
    // Reg 8             | Scale[0-3] |  Scale[0-3] |

    // Register Mapping for 16x128 for FP4, scale block size 16:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |
    // Reg 8 - 9         | Scale[0-7] |  Scale[0-7] |

    // clang-format on

    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N), threadIdx.x % BLOCK_N);
    auto col_major    = [](auto const& coord, auto ld) { return coord.second * ld; };
    auto startOffset  = col_major(startCoord2D, BLOCK_K / BLOCK_X);

    if(threadIdx.x < 16)
    {
        auto& scale_vec = fragX.template AsType<ScaleType>();
        static_for<0, scalar_type<ScaleFragT>::vector_size, 1>{}(
            [&](auto i) { scale_vec(Number<i.value>{}) = scale_ptr[startOffset + i.value]; });
    }

    return load_B_col_major<BType, BFragT, BLOCK_K, BLOCK_N>(input_ptr);
}

// Store function for WMMA output
template <typename CType, typename CFragT, int32_t BLOCK_M, int32_t BLOCK_N>
struct store_C_row_major;

template <typename CType, typename CFragT>
struct store_C_row_major<CType, CFragT, 16, 16>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t VW  = vectorSize(cFrag);
        static constexpr uint32_t Dim = 16;

        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, threadIdx.x % Dim);
        auto stepCoord2D  = std::make_pair(1u, 0u);

        auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

        auto startOffset = row_major(startCoord2D, 16);
        auto kOffset     = row_major(stepCoord2D, 16);

        for(uint32_t i = 0; i < vectorSize(cFrag); ++i)
        {
            CType* out_addr = output + startOffset + i * kOffset;
            *out_addr       = cFrag[i];
        }
    }
};

// WMMA scale kernel
template <typename AType,
          typename BType,
          typename AScaleType,
          typename BScaleType,
          typename CType,
          typename AccType,
          int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_K,
          int32_t BLOCK_X,
          typename ALayout,
          typename BLayout,
          typename CLayout>
__global__ void matmul(const packed_type_t<AType>* a,
                       const AScaleType* xa,
                       const packed_type_t<BType>* b,
                       const BScaleType* xb,
                       CType* c)
{
    using PackedAType            = packed_type_t<AType>;
    constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType            = packed_type_t<BType>;
    constexpr auto packed_size_b = packed_size_v<PackedBType>;

    constexpr int WAVE_SIZE = 32; // WMMA uses wave32
    assert(threadIdx.x < WAVE_SIZE);

    using AFragT =
        typename vector_type<PackedAType, BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_a>::type;
    using BFragT =
        typename vector_type<PackedBType, BLOCK_K * BLOCK_N / WAVE_SIZE / packed_size_b>::type;
    using CFragT        = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AccumFragT    = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = typename vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AScaleFragT   = typename vector_type<AScaleType, BLOCK_K / BLOCK_X>::type;
    using BScaleFragT   = typename vector_type<BScaleType, BLOCK_K / BLOCK_X>::type;

    // Create frags
    auto fragA   = AFragT{};
    auto fragB   = BFragT{};
    auto fragC   = CFragT{};
    auto fragAcc = AccumFragT{0};
    auto fragXa  = AScaleFragT{};
    auto fragXb  = BScaleFragT{};

    // Load the inputs
    if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    {
        fragA = load_mx_A_row_major<PackedAType,
                                    AFragT,
                                    AScaleType,
                                    AScaleFragT,
                                    BLOCK_M,
                                    BLOCK_K,
                                    BLOCK_X>(a, xa, fragXa);
    }
    else
    {
        static_assert(!is_same_v<ALayout, ALayout>, "ALayout must be RowMajor for matmul kernel");
    }

    if constexpr(is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
    {
        static_assert(!is_same_v<BLayout, BLayout>,
                      "BLayout must be ColumnMajor for matmul kernel");
    }
    else
    {
        fragB = load_mx_B_col_major<PackedBType,
                                    BFragT,
                                    BScaleType,
                                    BScaleFragT,
                                    BLOCK_K,
                                    BLOCK_N,
                                    BLOCK_X>(b, xb, fragXb);
    }

    // Scaled Matrix multiply-accumulate using WMMA scale units
    constexpr auto mfma_type_obj = ck::MfmaSelector<AType,
                                                    BLOCK_M,
                                                    BLOCK_N,
                                                    BType,
                                                    false,
                                                    true,
                                                    AccType,
                                                    BLOCK_X,
                                                    AScaleType,
                                                    BScaleType>::selected_mfma;
    mfma_type_obj
        .template run<BLOCK_M, BLOCK_N, 1, 0, AFragT, AScaleFragT, BFragT, BScaleFragT, AccumFragT>(
            fragA, fragXa, fragB, fragXb, fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        fragC[i] = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
    }

    if constexpr(is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
    {
        store_C_row_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
    else
    {
        static_assert(!is_same_v<CLayout, CLayout>, "CLayout must be RowMajor for matmul kernel");
    }
}

// Unscaled WMMA kernel for new instructions (no scale type)
template <typename AType,
          typename BType,
          typename CType,
          typename AccType,
          int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_K,
          typename ALayout,
          typename BLayout,
          typename CLayout>
__global__ void
matmul_unscaled(const packed_type_t<AType>* a, const packed_type_t<BType>* b, CType* c)
{
    using PackedAType = packed_type_t<AType>;
    using PackedBType = packed_type_t<BType>;

    constexpr int WAVE_SIZE = 32;
    assert(threadIdx.x < WAVE_SIZE);

    using AFragT =
        typename vector_type<PackedAType,
                             BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_v<PackedAType>>::type;
    using BFragT =
        typename vector_type<PackedBType,
                             BLOCK_K * BLOCK_N / WAVE_SIZE / packed_size_v<PackedBType>>::type;
    using CFragT        = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AccumFragT    = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = typename vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;

    auto fragA   = AFragT{};
    auto fragB   = BFragT{};
    auto fragC   = CFragT{};
    auto fragAcc = AccumFragT{0};

    // Load the inputs
    if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    {
        fragA = load_A_row_major<PackedAType, AFragT, BLOCK_M, BLOCK_K>(a);
    }
    else
    {
        static_assert(!is_same_v<ALayout, ALayout>,
                      "ALayout must be RowMajor for matmul_unscaled kernel");
    }

    if constexpr(is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
    {
        fragB = load_B_col_major<PackedBType, BFragT, BLOCK_K, BLOCK_N>(b);
    }
    else
    {
        static_assert(!is_same_v<BLayout, BLayout>,
                      "BLayout must be ColumnMajor for matmul_unscaled kernel");
    }

    // Select the correct MFMA/WMMA instruction using MfmaSelector::selected_mfma (auto-deduced)
    constexpr bool is_single_rate = ((BLOCK_K / WAVE_SIZE) > 2) ? false : true;

    constexpr auto mfma_type_obj = ck::
        MfmaSelector<AType, BLOCK_M, BLOCK_N, BType, is_single_rate, false, AccType>::selected_mfma;
    mfma_type_obj.template run<BLOCK_M, BLOCK_N, AFragT, BFragT, AccumFragT>(fragA, fragB, fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        auto val = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
        fragC[i] = val;
    }

    if constexpr(is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
    {
        static_assert(!is_same_v<CLayout, CLayout>,
                      "ColumnMajor CLayout is not implemented for matmul_unscaled kernel");
    }
    else if constexpr(is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
    {
        store_C_row_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
    }
    else
    {
        static_assert(!is_same_v<CLayout, CLayout>,
                      "CLayout must be RowMajor or ColumnMajor for matmul_unscaled kernel");
    }
}

// Test structure for WMMA scale operations
namespace mx_wmma_test {

template <typename ADataType,
          typename BDataType,
          typename AScaleType,
          typename BScaleType,
          typename CDataType>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<AScaleType>& a_scales,
                 const Tensor<BDataType>& B,
                 const Tensor<BScaleType>& b_scales,
                 Tensor<CDataType>& C)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm<ADataType,
                                                                              BDataType,
                                                                              CDataType,
                                                                              float,
                                                                              AScaleType,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              float,
                                                                              float,
                                                                              BScaleType>;
    auto ref_gemm               = ReferenceGemmInstance{};
    auto ref_invoker            = ref_gemm.MakeInvoker();

    auto ref_argument = ref_gemm.MakeArgument(
        A, a_scales, B, b_scales, C, PassThrough{}, PassThrough{}, PassThrough{});

    ref_invoker.Run(ref_argument);
}

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMMUnscaled(const Tensor<ADataType>& A,
                         const Tensor<BDataType>& B,
                         Tensor<CDataType>& C,
                         AElementwiseOperation a_element_op,
                         BElementwiseOperation b_element_op,
                         CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType,
          typename ADataType,
          typename BDataType,
          typename AScaleType,
          typename BScaleType,
          typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<AScaleType>& a_scales,
                   const Tensor<BDataType>& B,
                   const Tensor<BScaleType>& b_scales,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem a_scales_device_buf(sizeof(AScaleType) * a_scales.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem b_scales_device_buf(sizeof(BScaleType) * b_scales.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    a_scales_device_buf.ToDevice(a_scales.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    b_scales_device_buf.ToDevice(b_scales.mData.data());

    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const AScaleType*>(a_scales_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BScaleType*>(b_scales_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));

    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "HIP kernel launch error: " << hipGetErrorString(err) << std::endl;
        return false;
    }

    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

// RunDeviceGemmUnscaled: Launches the unscaled WMMA kernel (no scale types)
template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGemmUnscaled(KernelType kernel,
                           const Tensor<ADataType>& A,
                           const Tensor<BDataType>& B,
                           Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());

    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));

    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "HIP kernel launch error: " << hipGetErrorString(err) << std::endl;
        return false;
    }

    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceWMMA,
          typename ADataType,
          typename BDataType,
          typename AScaleType,
          typename BScaleType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t BLOCK_M,
          index_t BLOCK_N,
          index_t BLOCK_K,
          index_t BLOCK_X>
struct TestMXWMMA
{
    using PackedAType                   = packed_type_t<ADataType>;
    static constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType                   = packed_type_t<BDataType>;
    static constexpr auto packed_size_b = packed_size_v<PackedBType>;

    struct GemmParams
    {
        ck::index_t M = BLOCK_M;
        ck::index_t N = BLOCK_N;
        ck::index_t K = BLOCK_K;

        ck::index_t StrideA = -1;
        ck::index_t StrideB = -1;
        ck::index_t StrideC = -1;
    };

    auto PrepareGemmTensors(const GemmParams& params, index_t init)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<PackedAType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<AScaleType> a_scales(
            f_host_tensor_descriptor(params.M, params.K / BLOCK_X, params.K / BLOCK_X, ALayout{}));
        Tensor<PackedBType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<BScaleType> b_scales(
            f_host_tensor_descriptor(params.K / BLOCK_X, params.N, params.K / BLOCK_X, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        switch(init)
        {
        case 0:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{0.5f});
            b_n_k.GenerateTensorValue(GeneratorTensor_Sequential<PackedBType, 1>{});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{1.0f});
            break;
        case 1:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{512.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{1.0f / 512});
            break;
        case 2:
            a_m_k.GenerateTensorValue(GeneratorTensor_3<PackedAType>{-2.0, 2.0});
            a_scales.GenerateTensorValue(GeneratorTensor_2<AScaleType>{0, 4});
            b_n_k.GenerateTensorValue(GeneratorTensor_3<PackedBType>{-2.0, 2.0});
            b_scales.GenerateTensorValue(GeneratorTensor_2<BScaleType>{0, 4});
            break;
        case 3:
            // All-ones scales: neutral scaling (scale factor = 1.0), exercises raw arithmetic
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{1.0f});
            break;
        case 4:
            // All-zeros scales: forces zero output regardless of data content
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{0.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{0.0f});
            break;
        case 5:
            // All-ones scales, all ones input: neutral scaling (scale factor = 1.0), exercises raw
            // arithmetic
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{1.0f});
            break;
        case 6:
            // All-zeros scales, all one inputs forces zero output regardless of data content
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{0.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{0.0f});
            break;
        default:
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7});
            a_scales.GenerateTensorValue(GeneratorTensor_3<AScaleType>{0.0625f, 8.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7});
            b_scales.GenerateTensorValue(GeneratorTensor_3<BScaleType>{0.0625f, 8.0f});
            break;
        }

        return std::make_tuple(
            a_m_k, a_scales, b_n_k, b_scales, c_m_n_host_result, c_m_n_device_result);
    }

    template <typename DataType>
    void dump_tensor(Tensor<DataType> mat)
    {
        std::cout << "mat [ " << std::endl;

        auto len = mat.GetLengths();
        for(uint32_t i = 0; i < len[0]; i++)
        {
            std::cout << "    [";
            for(uint32_t j = 0; j < len[1]; j++)
            {
                std::vector<std::size_t> idx({i, j});
                if constexpr(is_same_v<DataType, f4x2_pk_t>)
                {
                    // f4x2_pk_t packs two f4 values - print both
                    auto pack = mat(idx);
                    std::cout << ck::type_convert<float>(f4_t(pack.template unpack<>(Number<0>{})))
                              << "/" // lo/hi separator within a packed element
                              << ck::type_convert<float>(f4_t(pack.template unpack<>(Number<1>{})))
                              << ", ";
                }
                else if constexpr(is_same_v<DataType, f6x16_pk_t> ||
                                  is_same_v<DataType, f6x32_pk_t>)
                {
                    // f6_pk_t packs packed_size f6_t values - print all
                    auto pack = mat(idx);
                    for(index_t k = 0; k < DataType::packed_size; ++k)
                    {
                        std::cout << ck::type_convert<float>(pack.unpack(k));
                        if(k < DataType::packed_size - 1)
                            std::cout << "/";
                    }
                    std::cout << ", ";
                }
                else if constexpr(is_same_v<DataType, bf6x16_pk_t> ||
                                  is_same_v<DataType, bf6x32_pk_t>)
                {
                    // bf6_pk_t packs packed_size bf6_t values - print all
                    auto pack = mat(idx);
                    for(index_t k = 0; k < DataType::packed_size; ++k)
                    {
                        std::cout << ck::type_convert<float>(pack.unpack(k));
                        if(k < DataType::packed_size - 1)
                            std::cout << "/";
                    }
                    std::cout << ", ";
                }
                else
                {
                    std::cout << ck::type_convert<float>(mat(idx)) << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    auto operator()(const DeviceWMMA& wmma_kernel, index_t init)
    {
        // Arrange
        GemmParams params;
        params.M = BLOCK_M;
        params.N = BLOCK_N;
        params.K = BLOCK_K;

        auto f_get_default_stride = [](std::size_t row,
                                       std::size_t col,
                                       ck::index_t stride,
                                       auto layout) {
            if(stride == -1)
            {
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

        params.StrideA = f_get_default_stride(BLOCK_M, BLOCK_K, params.StrideA, ALayout{});
        params.StrideB = f_get_default_stride(BLOCK_K, BLOCK_N, params.StrideB, BLayout{});
        params.StrideC = f_get_default_stride(BLOCK_M, BLOCK_N, params.StrideC, CLayout{});

        auto host_tensors = PrepareGemmTensors(params, init);

        const Tensor<PackedAType>& a       = std::get<0>(host_tensors);
        const Tensor<AScaleType>& a_scales = std::get<1>(host_tensors);
        const Tensor<PackedBType>& b       = std::get<2>(host_tensors);
        const Tensor<BScaleType>& b_scales = std::get<3>(host_tensors);
        Tensor<CDataType>& c_host          = std::get<4>(host_tensors);
        Tensor<CDataType>& c_device        = std::get<5>(host_tensors);

        RunHostGEMM(a, a_scales, b, b_scales, c_host);
        RunDeviceGEMM(wmma_kernel, a, a_scales, b, b_scales, c_device);

        bool res = false;
        if constexpr(std::is_same<CDataType, float>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
        }
        else
        {
            std::cout << "UNSUPPORTED CDataType" << std::endl;
        }

        return res;
    }
};

// Test structure for unscaled WMMA operations (no scale types)
template <typename DeviceWMMA,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t BLOCK_M,
          index_t BLOCK_N,
          index_t BLOCK_K,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct TestMXWMMAUnscaled
{
    using PackedAType                   = packed_type_t<ADataType>;
    static constexpr auto packed_size_a = packed_size_v<PackedAType>;
    using PackedBType                   = packed_type_t<BDataType>;
    static constexpr auto packed_size_b = packed_size_v<PackedBType>;

    struct GemmParams
    {
        ck::index_t M = BLOCK_M;
        ck::index_t N = BLOCK_N;
        ck::index_t K = BLOCK_K;

        ck::index_t StrideA = -1;
        ck::index_t StrideB = -1;
        ck::index_t StrideC = -1;
    };

    auto PrepareGemmTensors(const GemmParams& params, index_t init)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<PackedAType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<PackedBType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        switch(init)
        {
        case 0:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_Sequential<PackedBType, 1>{});
            break;
        case 1:
            a_m_k.GenerateTensorValue(GeneratorTensor_1<PackedAType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            break;
        case 2:
            a_m_k.GenerateTensorValue(GeneratorTensor_3<PackedAType>{-2.0, 2.0});
            b_n_k.GenerateTensorValue(GeneratorTensor_3<PackedBType>{-2.0, 2.0});
            break;
        default:
            a_m_k.GenerateTensorValue(GeneratorTensor_2<PackedAType>{-6, 7});
            b_n_k.GenerateTensorValue(GeneratorTensor_2<PackedBType>{-6, 7});
            break;
        }

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    template <typename DataType>
    void dump_tensor(Tensor<DataType> mat)
    {
        std::cout << "mat [ " << std::endl;

        auto len = mat.GetLengths();
        for(uint32_t i = 0; i < len[0]; i++)
        {
            std::cout << "    [";
            for(uint32_t j = 0; j < len[1]; j++)
            {
                std::vector<std::size_t> idx({i, j});
                std::cout << ck::type_convert<float>(mat(idx)) << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    template <typename DataType>
    void dump_tensor_hex(Tensor<DataType> mat)
    {
        std::cout << "mat (hex) [ " << std::endl;
        auto len = mat.GetLengths();
        for(uint32_t i = 0; i < len[0]; i++)
        {
            std::cout << "    [";
            for(uint32_t j = 0; j < len[1]; j++)
            {
                std::vector<std::size_t> idx({i, j});
                union
                {
                    float f;
                    uint32_t u;
                } uval;
                uval.f = ck::type_convert<float>(mat(idx));
                std::cout << "0x" << std::hex << uval.u << std::dec << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    auto operator()(const DeviceWMMA& wmma_kernel, index_t init)
    {
        // Arrange
        GemmParams params;
        params.M = BLOCK_M;
        params.N = BLOCK_N;
        params.K = BLOCK_K;

        auto f_get_default_stride = [](std::size_t row,
                                       std::size_t col,
                                       ck::index_t stride,
                                       auto layout) {
            if(stride == -1)
            {
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

        params.StrideA = f_get_default_stride(BLOCK_M, BLOCK_K, params.StrideA, ALayout{});
        params.StrideB = f_get_default_stride(BLOCK_K, BLOCK_N, params.StrideB, BLayout{});
        params.StrideC = f_get_default_stride(BLOCK_M, BLOCK_N, params.StrideC, CLayout{});

        auto host_tensors = PrepareGemmTensors(params, init);

        const Tensor<PackedAType>& a = std::get<0>(host_tensors);
        const Tensor<PackedBType>& b = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host    = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device  = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        RunHostGEMMUnscaled<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);
        RunDeviceGemmUnscaled(wmma_kernel, a, b, c_device);

        bool res = false;
        if constexpr(std::is_same<CDataType, float>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
        }
        else if(std::is_same<CDataType, ck::half_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
        }
        else
        {
            std::cout << "UNSUPPORTED CDataType" << std::endl;
        }
        return res;
    }
};

} // namespace mx_wmma_test
} // namespace ck
