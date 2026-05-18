// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

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
    SCALE_F32_32x16x128 = static_cast<int>(
        MfmaInstr::wmma_scale_f32_32x16x128_f4_gfx125), // V_WMMA_SCALE_F32_32X16X128_F4
    SCALE16_F32_32x16x128 = static_cast<int>(
        MfmaInstr::wmma_scale16_f32_32x16x128_f4_gfx125) // V_WMMA_SCALE16_F32_32X16X128_F4
};

template <int32_t BLOCK_M,
          int32_t BLOCK_N,
          int32_t BLOCK_X,
          typename ScaleTypeA,
          typename ScaleTypeB>
struct wmma_scale_type_selector;

template <typename ScaleTypeA, typename ScaleTypeB>
struct wmma_scale_type_selector<16, 16, 32, ScaleTypeA, ScaleTypeB>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::wmma_scale_f32_16x16x128_f8f6f4_gfx125>{};
        op.template run<16, 16, 1, 1, AFragT, AScaleFragT, BFragT, BScaleFragT, AccumFragT>(
            fragA, scale_a, fragB, scale_b, fragAcc);
    }
};

template <typename ScaleTypeA, typename ScaleTypeB>
struct wmma_scale_type_selector<32, 16, 32, ScaleTypeA, ScaleTypeB>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::wmma_scale_f32_32x16x128_f4_gfx125>{};
        op.template run<32, 16, 1, AFragT, AScaleFragT, BFragT, BScaleFragT, AccumFragT>(
            fragA, scale_a, fragB, scale_b, fragAcc);
    }
};

template <typename ScaleTypeA, typename ScaleTypeB>
struct wmma_scale_type_selector<16, 16, 16, ScaleTypeA, ScaleTypeB>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::wmma_scale16_f32_16x16x128_f8f6f4_gfx125>{};
        op.template run<16, 16, 1, 1, AFragT, AScaleFragT, BFragT, BScaleFragT, AccumFragT>(
            fragA, scale_a, fragB, scale_b, fragAcc);
    }
};

template <typename ScaleTypeA, typename ScaleTypeB>
struct wmma_scale_type_selector<32, 16, 16, ScaleTypeA, ScaleTypeB>
{
    template <typename AFragT,
              typename AScaleFragT,
              typename BFragT,
              typename BScaleFragT,
              typename AccumFragT>
    __device__ static void run(AFragT const& fragA,
                               AScaleFragT const& scale_a,
                               BFragT const& fragB,
                               BScaleFragT const& scale_b,
                               AccumFragT& fragAcc)
    {
        auto op = mfma_type<MfmaInstr::wmma_scale16_f32_32x16x128_f4_gfx125>{};
        op.template run<32, 16, 1, AFragT, AScaleFragT, BFragT, BScaleFragT, AccumFragT>(
            fragA, scale_a, fragB, scale_b, fragAcc);
    }
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

    // Register Mapping for 16x128 for FP4:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  | v[0]
    // Reg 0 [8:15]      |     K2K3   |     K34K35  | v[1]
    // Reg 0 [16:23]     |     K4K5   |     K36K37  | v[2]
    // Reg 0 [24:31]     |     K6K7   |     K38K39  | v[3]
    // Reg 1 [0:7]       |     K8K9   |     K40K41  | v[4]
    // Reg 1 [8:15]      |     K10K11 |     K42K43  | v[5]
    // Reg 1 [16:23]     |     K12K13 |     K44K45  | v[6]
    // Reg 1 [24:31]     |     K14K15 |     K46K47  | v[7]
    // Reg 2 [0:7]       |     K16K17 |     K48K49  | v[8]
    // Reg 2 [8:15]      |     K18K19 |     K50K51  | v[9]
    // Reg 2 [16:23]     |     K20K21 |     K52K53  | v[10]
    // Reg 2 [24:31]     |     K22K23 |     K54K55  | v[11]
    // Reg 3 [0:7]       |     K24K25 |     K56K57  | v[12]
    // Reg 3 [8:15]      |     K26K27 |     K58K59  | v[13]
    // Reg 3 [16:23]     |     K28K29 |     K60K61  | v[14]
    // Reg 3 [24:31]     |     K30K31 |     K62K63  | v[15]
    // Reg 4 [0:7]       |     K64K65 |     K96K97  | v[16]
    // Reg 4 [8:15]      |     K66K67 |     K98K99  | v[17]
    // Reg 4 [16:23]     |     K68K69 |    K100K101 | v[18]
    // Reg 4 [24:31]     |     K70K71 |    K102K103 | v[19]
    // Reg 5 [0:7]       |     K72K73 |    K104K105 | v[20]
    // Reg 5 [8:15]      |     K74K75 |    K106K107 | v[21]
    // Reg 5 [16:23]     |     K76K77 |    K108K109 | v[22]
    // Reg 5 [24:31]     |     K78K79 |    K110K111 | v[23]
    // Reg 6 [0:7]       |     K80K81 |    K112K113 | v[24]
    // Reg 6 [8:15]      |     K82K83 |    K114K115 | v[25]
    // Reg 6 [16:23]     |     K84K85 |    K116K117 | v[26]
    // Reg 6 [24:31]     |     K86K87 |    K118K119 | v[27]
    // Reg 7 [0:7]       |     K88K89 |    K120K121 | v[28]
    // Reg 7 [8:15]      |     K90K91 |    K122K123 | v[29]
    // Reg 7 [16:23]     |     K92K93 |    K124K125 | v[30]
    // Reg 7 [24:31]     |     K94K95 |    K126K127 | v[31]

    // Register Mapping for 32x128 for FP4:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  | v[0]
    // Reg 0 [8:15]      |     K2K3   |     K34K35  | v[1]
    // Reg 0 [16:23]     |     K4K5   |     K36K37  | v[2]
    // Reg 0 [24:31]     |     K6K7   |     K38K39  | v[3]
    // Reg 1 [0:7]       |     K8K9   |     K40K41  | v[4]
    // Reg 1 [8:15]      |     K10K11 |     K42K43  | v[5]
    // Reg 1 [16:23]     |     K12K13 |     K44K45  | v[6]
    // Reg 1 [24:31]     |     K14K15 |     K46K47  | v[7]
    // Reg 2 [0:7]       |     K16K17 |     K48K49  | v[8]
    // Reg 2 [8:15]      |     K18K19 |     K50K51  | v[9]
    // Reg 2 [16:23]     |     K20K21 |     K52K53  | v[10]
    // Reg 2 [24:31]     |     K22K23 |     K54K55  | v[11]
    // Reg 3 [0:7]       |     K24K25 |     K56K57  | v[12]
    // Reg 3 [8:15]      |     K26K27 |     K58K59  | v[13]
    // Reg 3 [16:23]     |     K28K29 |     K60K61  | v[14]
    // Reg 3 [24:31]     |     K30K31 |     K62K63  | v[15]
    // Reg 4 [0:7]       |     K64K65 |     K96K97  | v[16]
    // Reg 4 [8:15]      |     K66K67 |     K98K99  | v[17]
    // Reg 4 [16:23]     |     K68K69 |    K100K101 | v[18]
    // Reg 4 [24:31]     |     K70K71 |    K102K103 | v[19]
    // Reg 5 [0:7]       |     K72K73 |    K104K105 | v[20]
    // Reg 5 [8:15]      |     K74K75 |    K106K107 | v[21]
    // Reg 5 [16:23]     |     K76K77 |    K108K109 | v[22]
    // Reg 5 [24:31]     |     K78K79 |    K110K111 | v[23]
    // Reg 6 [0:7]       |     K80K81 |    K112K113 | v[24]
    // Reg 6 [8:15]      |     K82K83 |    K114K115 | v[25]
    // Reg 6 [16:23]     |     K84K85 |    K116K117 | v[26]
    // Reg 6 [24:31]     |     K86K87 |    K118K119 | v[27]
    // Reg 7 [0:7]       |     K88K89 |    K120K121 | v[28]
    // Reg 7 [8:15]      |     K90K91 |    K122K123 | v[29]
    // Reg 7 [16:23]     |     K92K93 |    K124K125 | v[30]
    // Reg 7 [24:31]     |     K94K95 |    K126K127 | v[31]
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 16  ...  31|  16  ...  31|
    // Thread Id         |  0  ...  15|  16  ...  31|
    // Reg 8 [0:7]       |     K0K1   |     K32K33  | v[32]
    // Reg 8 [8:15]      |     K2K3   |     K34K35  | v[33]
    // Reg 8 [16:23]     |     K4K5   |     K36K37  | v[34]
    // Reg 8 [24:31]     |     K6K7   |     K38K39  | v[35]
    // Reg 9 [0:7]       |     K8K9   |     K40K41  | v[36]
    // Reg 9 [8:15]      |     K10K11 |     K42K43  | v[37]
    // Reg 9 [16:23]     |     K12K13 |     K44K45  | v[38]
    // Reg 9 [24:31]     |     K14K15 |     K46K47  | v[39]
    // Reg 10 [0:7]      |     K16K17 |     K48K49  | v[40]
    // Reg 10 [8:15]     |     K18K19 |     K50K51  | v[41]
    // Reg 10 [16:23]    |     K20K21 |     K52K53  | v[42]
    // Reg 10 [24:31]    |     K22K23 |     K54K55  | v[43]
    // Reg 11 [0:7]      |     K24K25 |     K56K57  | v[44]
    // Reg 11 [8:15]     |     K26K27 |     K58K59  | v[45]
    // Reg 11 [16:23]    |     K28K29 |     K60K61  | v[46]
    // Reg 11 [24:31]    |     K30K31 |     K62K63  | v[47]
    // Reg 12 [0:7]      |     K64K65 |     K96K97  | v[48]
    // Reg 12 [8:15]     |     K66K67 |     K98K99  | v[49]
    // Reg 12 [16:23]    |     K68K69 |    K100K101 | v[50]
    // Reg 12 [24:31]    |     K70K71 |    K102K103 | v[51]
    // Reg 13 [0:7]      |     K72K73 |    K104K105 | v[52]
    // Reg 13 [8:15]     |     K74K75 |    K106K107 | v[53]
    // Reg 13 [16:23]    |     K76K77 |    K108K109 | v[54]
    // Reg 13 [24:31]    |     K78K79 |    K110K111 | v[55]
    // Reg 14 [0:7]      |     K80K81 |    K112K113 | v[56]
    // Reg 14 [8:15]     |     K82K83 |    K114K115 | v[57]
    // Reg 14 [16:23]    |     K84K85 |    K116K117 | v[58]
    // Reg 14 [24:31]    |     K86K87 |    K118K119 | v[59]
    // Reg 15 [0:7]      |     K89K90 |    K120K121 | v[60]
    // Reg 15 [8:15]     |     K91K92 |    K122K123 | v[61]
    // Reg 15 [16:23]    |     K93K94 |    K124K125 | v[62]
    // Reg 15 [24:31]    |     K95K96 |    K126K127 | v[63]

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 32; // WMMA uses wave32

    // FP4 chunk_size = 32, num_chunks = 2, packed_size = 2

    constexpr index_t num_chunks = 2;

    // Here we want to load from rows of A in chunks of 64 or 32 elements each.
    constexpr uint32_t chunk_size = is_packed_type_v<AType> ? 32 : 16;

    // each chunk is separated by offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_M; // 64 or 32

    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M, (threadIdx.x / BLOCK_M) * chunk_size);
    auto majorStepCoord2D = std::make_pair(0, chunk_offset);

    // Flatten to 1D row_major offsets.
    auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

    using ARawT = typename scalar_type<AFragT>::type;
    using AScalarChunkT =
        typename vector_type<ARawT, scalar_type<AFragT>::vector_size / num_chunks>::type;

    union
    {
        AFragT frag;
        AScalarChunkT chunks[num_chunks];
    } fragA{};

    const AScalarChunkT* fragPtr;

    // BLOCK_K is a stride in A matrix
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

    // Register Mapping for 16x128 for FP4:
    // Size              |   BLOCK_N  |   BLOCK_N   |
    // N                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 [0:7]       |     K0K1   |     K32K33  | v[0]
    // Reg 0 [8:15]      |     K2K3   |     K34K35  | v[1]
    // Reg 0 [16:23]     |     K4K5   |     K36K37  | v[2]
    // Reg 0 [24:31]     |     K6K7   |     K38K39  | v[3]
    // Reg 1 [0:7]       |     K8K9   |     K40K41  | v[4]
    // Reg 1 [8:15]      |     K10K11 |     K42K43  | v[5]
    // Reg 1 [16:23]     |     K12K13 |     K44K45  | v[6]
    // Reg 1 [24:31]     |     K14K15 |     K46K47  | v[7]
    // Reg 2 [0:7]       |     K16K17 |     K48K49  | v[8]
    // Reg 2 [8:15]      |     K18K19 |     K50K51  | v[9]
    // Reg 2 [16:23]     |     K20K21 |     K52K53  | v[10]
    // Reg 2 [24:31]     |     K22K23 |     K54K55  | v[11]
    // Reg 3 [0:7]       |     K24K25 |     K56K57  | v[12]
    // Reg 3 [8:15]      |     K26K27 |     K58K59  | v[13]
    // Reg 3 [16:23]     |     K28K29 |     K60K61  | v[14]
    // Reg 3 [24:31]     |     K30K31 |     K62K63  | v[15]
    // Reg 4 [0:7]       |     K64K65 |     K96K97  | v[16]
    // Reg 4 [8:15]      |     K66K67 |     K98K99  | v[17]
    // Reg 4 [16:23]     |     K68K69 |    K100K101 | v[18]
    // Reg 4 [24:31]     |     K70K71 |    K102K103 | v[19]
    // Reg 5 [0:7]       |     K72K73 |    K104K105 | v[20]
    // Reg 5 [8:15]      |     K74K75 |    K106K107 | v[21]
    // Reg 5 [16:23]     |     K76K77 |    K108K109 | v[22]
    // Reg 5 [24:31]     |     K78K79 |    K110K111 | v[23]
    // Reg 6 [0:7]       |     K80K81 |    K112K113 | v[24]
    // Reg 6 [8:15]      |     K82K83 |    K114K115 | v[25]
    // Reg 6 [16:23]     |     K84K85 |    K116K117 | v[26]
    // Reg 6 [24:31]     |     K86K87 |    K118K119 | v[27]
    // Reg 7 [0:7]       |     K88K89 |    K120K121 | v[28]
    // Reg 7 [8:15]      |     K90K91 |    K122K123 | v[29]
    // Reg 7 [16:23]     |     K92K93 |    K124K125 | v[30]
    // Reg 7 [24:31]     |     K94K95 |    K126K127 | v[31]

    // clang-format on

    static constexpr int32_t WAVE_SIZE = 32;

    // FP4 chunk_size = 32, num_chunks = 2, packed_size = 2

    constexpr index_t num_chunks = 2;

    // Here we want to load from cols of B in chunks of 64 or 32 elements each.
    constexpr uint32_t chunk_size = is_packed_type_v<BType> ? 32 : 16;

    // each chunk is separated by an offset
    static constexpr uint32_t chunk_offset = chunk_size * WAVE_SIZE / BLOCK_N; // 64

    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N) * chunk_size, threadIdx.x % BLOCK_N);

    // Flatten to 1D col_major offsets.
    auto col_major = [](auto const& coord, auto ld) { return coord.first + coord.second * ld; };

    auto majorStepCoord2D = std::make_pair(chunk_offset, 0);

    using BRawT = typename scalar_type<BFragT>::type;
    using BScalarChunkT =
        typename vector_type<BRawT, scalar_type<BFragT>::vector_size / num_chunks>::type;

    union
    {
        BFragT frag;
        BScalarChunkT chunks[num_chunks];
    } fragB{};

    const BScalarChunkT* fragPtr;

    // BLOCK_K is a stride in B matrix
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
          int32_t BLOCK_X,
          index_t num_steps>
__device__ AFragT load_mx_A_row_major(AType const* input_ptr,
                                      ScaleType const* scale_ptr,
                                      ScaleFragT& fragX)
{
    // clang-format off

    // Register Mapping for 16x128 for FP4:
    // Thread Id |   0 ... 15   |   16 ... 31   |
    // M         |   0 ... 15   |    0 ... 15   |
    // Register  |--------------|---------------|
    // Reg 0-3   |   K0  - K31  |   K32 - K63   |
    // Reg 4-7   |   K64 - K95  |   K96 - K127  |
    // Reg 8     |   Scale[0-3] |   Scale[0-3]  |

    // Register Mapping for 16x128 for FP4, scale block size 16:
    // Size              |   BLOCK_M  |   BLOCK_M   |
    // M                 | 0  ...  15 |  0  ...  15 |
    // Thread Id         | 0  ...  15 | 16  ...  31 |
    // Register Element  |------------|-------------|
    // Reg 0 - 3         |    K0-K31  |    K32-K63  |
    // Reg 4 - 7         |    K64-K95 |    K96-K127 |
    // Reg 8 - 9         | Scale[0-7] |  Scale[0-7] |

    // Register Mapping for 32x128 for FP4:
    // Thread Id |      0  ...  15    |      16  ...  31      |
    // Register  |--------------------|-----------------------|
    // Reg 0-3   | M=thId,    K0-K31  | M=thId % 16, K32-K63  |
    // Reg 4-7   | M=thId,    K64-K95 | M=thId % 16, K96-K127 |
    // Reg 8-11  | M=thId+16, K0-K31  | M=thId,      K32-K63  |
    // Reg 12-15 | M=thId+16, K64-K95 | M=thId,      K96-K127 |
    // Reg 16    | Scale[M=thId,0-3]  | Scale[M=thId,0-3]     |

    // Register Mapping for 32x128 for FP4, scale block size 16:
    // Thread Id |      0  ...  15    |      16  ...  31      |
    // Register  |--------------------|-----------------------|
    // Reg 0-3   | M=thId,    K0-K31  | M=thId % 16, K32-K63  |
    // Reg 4-7   | M=thId,    K64-K95 | M=thId % 16, K96-K127 |
    // Reg 8-11  | M=thId+16, K0-K31  | M=thId,      K32-K63  |
    // Reg 12-15 | M=thId+16, K64-K95 | M=thId,      K96-K127 |
    // Reg 16-17 | Scale[M=thId,0-7]  | Scale[M=thId,0-7]     |

    // clang-format on

    auto startCoord2D = std::make_pair(threadIdx.x % BLOCK_M, (threadIdx.x / BLOCK_M));

    index_t startOffset = 0;
    if constexpr(num_steps == 1)
    {
        startOffset = startCoord2D.first * (BLOCK_K / BLOCK_X);
    }
    else if constexpr(num_steps == 2)
    {
        constexpr index_t stride = BLOCK_M * (BLOCK_K / BLOCK_X);
        startOffset = startCoord2D.first * (BLOCK_K / BLOCK_X) + startCoord2D.second * stride;
    }

    auto& scale_vec = fragX.template AsType<ScaleType>();
    static_for<0, scalar_type<ScaleFragT>::vector_size, 1>{}(
        [&](auto i) { scale_vec(Number<i.value>{}) = scale_ptr[startOffset + i.value]; });

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

    // Register Mapping for 16x128 for FP4:
    // Thread Id |   0 ... 15   |   16 ... 31   |
    // N         |   0 ... 15   |    0 ... 15   |
    // Register  |--------------|---------------|
    // Reg 0-3   |   K0  - K31  |   K32 - K63   |
    // Reg 4-7   |   K64 - K95  |   K96 - K127  |
    // Reg 8     |   Scale[0-3] |   Scale[0-3]  |

    // clang-format on

    auto startCoord2D = std::make_pair((threadIdx.x / BLOCK_N), threadIdx.x % BLOCK_N);
    auto col_major    = [](auto const& coord, auto ld) { return coord.second * ld; };
    auto startOffset  = col_major(startCoord2D, BLOCK_K / BLOCK_X);

    auto& scale_vec = fragX.template AsType<ScaleType>();
    static_for<0, scalar_type<ScaleFragT>::vector_size, 1>{}(
        [&](auto i) { scale_vec(Number<i.value>{}) = scale_ptr[startOffset + i.value]; });

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

        auto startOffset = row_major(startCoord2D, Dim);
        auto kOffset     = row_major(stepCoord2D, Dim);

        for(uint32_t i = 0; i < vectorSize(cFrag); ++i)
        {
            output[startOffset + i * kOffset] = cFrag[i];
        }
    }
};

template <typename CType, typename CFragT>
struct store_C_row_major<CType, CFragT, 32, 16>
{
    __device__ void operator()(CType* output, CFragT cFrag)
    {
        static constexpr uint32_t VW  = vectorSize(cFrag);
        static constexpr uint32_t Dim = 32;

        auto startCoord2D = std::make_pair((threadIdx.x / Dim) * VW, threadIdx.x % Dim);
        auto stepCoord2D  = std::make_pair(1u, 0u);

        auto row_major = [](auto const& coord, auto ld) { return coord.first * ld + coord.second; };

        auto startOffset = row_major(startCoord2D, Dim);
        auto kOffset     = row_major(stepCoord2D, Dim);

        for(uint32_t i = 0; i < vectorSize(cFrag); ++i)
        {
            output[startOffset + i * kOffset] = cFrag[i];
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
          typename CLayout,
          index_t num_steps>
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
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);

    using AFragT =
        typename vector_type<PackedAType, BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_a>::type;
    using AFragPartT =
        typename vector_type<PackedAType,
                             BLOCK_M * BLOCK_K / WAVE_SIZE / packed_size_a / num_steps>::type;
    using BFragT =
        typename vector_type<PackedBType, BLOCK_K * BLOCK_N / WAVE_SIZE / packed_size_b>::type;
    using CFragT     = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using CFragPartT = typename vector_type<CType, BLOCK_M * BLOCK_N / WAVE_SIZE / num_steps>::type;
    using AccumFragT = vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>;
    using RawAccumFragT = typename vector_type<AccType, BLOCK_M * BLOCK_N / WAVE_SIZE>::type;
    using AScaleFragT =
        typename vector_type<AScaleType,
                             BLOCK_K / BLOCK_X>::type; // packed BLOCK_K / BLOCK_X scale values
    using BScaleFragT =
        typename vector_type<BScaleType,
                             BLOCK_K / BLOCK_X>::type; // packed BLOCK_K / BLOCK_X scale values

    // Create frags
    auto fragA        = AFragT{};
    auto fragB        = BFragT{};
    auto fragC        = CFragT{};
    auto fragAcc      = AccumFragT{0};
    auto fragXa       = AScaleFragT{};
    auto fragXa_dummy = AScaleFragT{};
    auto fragXb       = BScaleFragT{};

    // Load the inputs
    if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    {
        if constexpr(num_steps == 1)
        {
            fragA = load_mx_A_row_major<PackedAType,
                                        AFragT,
                                        AScaleType,
                                        AScaleFragT,
                                        BLOCK_M,
                                        BLOCK_K,
                                        BLOCK_X,
                                        num_steps>(a, xa, fragXa);
        }
        else if constexpr(num_steps == 2)
        {
            // load fragA in two runs
            union
            {
                AFragT fragA_full{};
                AFragPartT fragA_part[2];
            } fragA_union{};

            fragA_union.fragA_part[0] = load_mx_A_row_major<PackedAType,
                                                            AFragPartT,
                                                            AScaleType,
                                                            AScaleFragT,
                                                            BLOCK_M / num_steps,
                                                            BLOCK_K,
                                                            BLOCK_X,
                                                            num_steps>(a, xa, fragXa);

            constexpr index_t a_offset = BLOCK_M * BLOCK_K / packed_size_a / num_steps;
            fragA_union.fragA_part[1] =
                load_mx_A_row_major<PackedAType,
                                    AFragPartT,
                                    AScaleType,
                                    AScaleFragT,
                                    BLOCK_M / num_steps,
                                    BLOCK_K,
                                    BLOCK_X,
                                    num_steps>(a + a_offset,
                                               xa,
                                               fragXa_dummy); // scales already loaded in fragXa
            // pack chunks of fragA together
            fragA = fragA_union.fragA_full;
        }
        else
        {
            printf("This load pattern is not implemented\n");
        }
    }
    else
    {
        printf("This layout is not implemented\n");
    }

    if constexpr(is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
    {
        printf("This layout is not implemented\n");
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
    using wmma = wmma_scale_type_selector<BLOCK_M, BLOCK_N, BLOCK_X, AScaleFragT, BScaleFragT>;
    wmma::template run<>(fragA, fragXa, fragB, fragXb, fragAcc);

    for(int i = 0; i < vectorSize(fragC); ++i)
    {
        fragC[i] = type_convert<CType>(fragAcc.template AsType<RawAccumFragT>()[Number<0>{}][i]);
    }

    if constexpr(is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
    {
        if constexpr(num_steps == 1)
        {
            store_C_row_major<CType, CFragT, BLOCK_M, BLOCK_N>{}(c, fragC);
        }
        else if constexpr(num_steps == 2)
        {
            union
            {
                CFragT fragC_full{};
                CFragPartT fragC_part[2];
            } fragC_union;
            // unpack fragC into chunks
            fragC_union.fragC_full = fragC;
            // store C in two runs
            constexpr index_t c_offset = BLOCK_M * BLOCK_N / num_steps;
            store_C_row_major<CType, CFragPartT, BLOCK_M / num_steps, BLOCK_N>{}(
                c, fragC_union.fragC_part[0]);
            store_C_row_major<CType, CFragPartT, BLOCK_M / num_steps, BLOCK_N>{}(
                c + c_offset, fragC_union.fragC_part[1]);
        }
        else
        {
            printf("This store pattern is not implemented\n");
        }
    }
    else
    {
        printf("This layout is not implemented\n");
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

    const int cold_iters = 1;
    printf("Warm up %d times\n", cold_iters);

    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const AScaleType*>(a_scales_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BScaleType*>(b_scales_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));

    // warm up
    for(int i = 0; i < cold_iters; ++i)
    {
        kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<const AScaleType*>(a_scales_device_buf.GetDeviceBuffer()),
                          static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                          static_cast<const BScaleType*>(b_scales_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
        hip_check_error(hipGetLastError());
    }

    const int num_repeat = 1;

    hipEvent_t start, stop;

    hip_check_error(hipEventCreate(&start));
    hip_check_error(hipEventCreate(&stop));

    hip_check_error(hipDeviceSynchronize());
    hip_check_error(hipEventRecord(start));

    // time the kernel
    for(int i = 0; i < num_repeat; ++i)
    {
        kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<const AScaleType*>(a_scales_device_buf.GetDeviceBuffer()),
                          static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                          static_cast<const BScaleType*>(b_scales_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
        hip_check_error(hipGetLastError());
    }

    hip_check_error(hipEventRecord(stop));
    hip_check_error(hipEventSynchronize(stop));

    float total_time = 0;

    hip_check_error(hipEventElapsedTime(&total_time, start, stop));

    hip_check_error(hipEventDestroy(start));
    hip_check_error(hipEventDestroy(stop));

    printf("Kernel took %f ms in average over %d runs\n", total_time / num_repeat, num_repeat);

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
            a_scales.GenerateTensorValue(GeneratorTensor_1<AScaleType>{1.0f});
            b_n_k.GenerateTensorValue(GeneratorTensor_1<PackedBType>{1.0f});
            b_scales.GenerateTensorValue(GeneratorTensor_1<BScaleType>{1.0f});
            break;
        case 2:
            a_m_k.GenerateTensorValue(GeneratorTensor_3<PackedAType>{-2.0, 2.0});
            a_scales.GenerateTensorValue(GeneratorTensor_2<AScaleType>{0, 4});
            b_n_k.GenerateTensorValue(GeneratorTensor_3<PackedBType>{-2.0, 2.0});
            b_scales.GenerateTensorValue(GeneratorTensor_2<BScaleType>{0, 4});
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

} // namespace mx_wmma_test
} // namespace ck
