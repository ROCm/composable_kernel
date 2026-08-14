// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <stdexcept>

namespace ck_tile {

/// @brief Pre-shuffle scale buffer for gfx1250 wmma mx scale instruction.
///
/// Reorganizes the scale data from row-major (MN x K) layout into the M/N-fastest layout
/// consumed by the unified mx_gemm scale descriptor ([packs_mn, packs_k, MThreadPerXdl]),
/// so consecutive lanes read consecutive addresses (coalesced) on every K iteration.
///
/// scale16 and scale32 share ONE layout, parameterized by MThreadPerXdl == the WMMA WarpTile
/// M/N (the number of lanes with a distinct scale row per warp) -- NOT by ScaleBlockSize:
///   - a 32x32 wmma has 32 distinct scale lanes -> MThreadPerXdl = 32;
///   - a 16x16 wmma has 16 (the other wavefront lanes replicate the same scale rows) -> 16.
/// Both scale16 and scale32 are used with 32x32 and 16x16 wmma, so MThreadPerXdl must come from
/// the caller's WarpTile, not from ScaleBlockSize. ScaleBlockSize only changes how many K-packs
/// a lane holds: scale16 halves the K span per scale, so a lane holds twice as many int32 K-packs
/// and the device reads two adjacent int32 as one int64_t -- a device-side read concern; the host
/// always writes plain int32-packed (PackSize=4) bytes in the same M-fastest order.
/// The scale values are unchanged; only their position is permuted.
///
/// @tparam ScaleType Scale data type (e.g., e8m0_t)
/// @tparam ScaleBlockSize The block size for microscaling (16 or 32)
/// @tparam KStride Whether K is the fast-moving dimension of the source
/// @param MThreadPerXdl The WMMA WarpTile M (for A scales) or N (for B scales) -- the number of
///        lanes holding a distinct scale row per warp. This, NOT ScaleBlockSize, selects the
///        layout: e.g. a 32x32 wmma has 32 distinct scale lanes, a 16x16 wmma has 16 (the other
///        wavefront lanes replicate the same scale rows). ScaleBlockSize only changes how many
///        K-packs a lane holds (the device reads two adjacent int32 as one int64 for scale16).
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                   ScaleType* dst,
                                   ck_tile::index_t MN,
                                   ck_tile::index_t K,
                                   ck_tile::index_t MThreadPerXdl)
{
    static_assert((ScaleBlockSize == 32 || ScaleBlockSize == 16) && sizeof(ScaleType) == 1,
                  "wrong! only support 8-bit scale with ScaleBlockSize=32 or 16");

    constexpr ck_tile::long_index_t PackSize = 4; // e8m0 scales per packed int32_t

    // Contract: the device descriptor pads with integer_divide_ceil(MN, MThreadPerXdl) and reads
    // int32 K-packs, so the caller must pass an MN already padded to a multiple of MThreadPerXdl
    // and a K (num_scale_k) that is a multiple of PackSize, with src/dst allocated to match.
    // Fail fast rather than writing past dst (or laying out inconsistently with the descriptor).
    if(K % PackSize != 0)
        throw std::runtime_error("preShuffleScaleBuffer_gfx1250: num_scale_k (K) must be a "
                                 "multiple of 4 (e8m0 scales packed per int32).");
    if(MThreadPerXdl <= 0 || MN % MThreadPerXdl != 0)
        throw std::runtime_error("preShuffleScaleBuffer_gfx1250: MN must be padded to a multiple "
                                 "of MThreadPerXdl (the WMMA WarpTile M/N); pass the padded MN and "
                                 "allocate src/dst accordingly.");

    const ck_tile::long_index_t packs_k = K / PackSize;

    for(ck_tile::long_index_t mn = 0; mn < MN; ++mn)
    {
        const ck_tile::long_index_t pm = mn / MThreadPerXdl;
        const ck_tile::long_index_t mt = mn % MThreadPerXdl;
        for(ck_tile::long_index_t k = 0; k < K; ++k)
        {
            const ck_tile::long_index_t pk  = k / PackSize;
            const ck_tile::long_index_t sub = k % PackSize;
            const ck_tile::long_index_t outputIndex =
                ((pm * packs_k + pk) * MThreadPerXdl + mt) * PackSize + sub;

            if constexpr(KStride)
                dst[outputIndex] = src[mn * K + k];
            else
                dst[outputIndex] = src[k * MN + mn];
        }
    }
}

// Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
// Each int32_t contains MNPack * KPack e8m0_t values with byte layout matching
// the GPU tile distribution: values are XdlMNThread apart in M and XdlKThread apart in K.
// byte[ik * MNPack + imn] = e8m0 at strided (mn, k) position
// kLast=true for A scales (layout [M, K/32]), kLast=false for B scales (layout [K/32, N])
template <ck_tile::index_t MNPack      = 2,
          ck_tile::index_t KPack       = 2,
          ck_tile::index_t XdlMNThread = 16,
          ck_tile::index_t XdlKThread  = 4,
          typename ScaleType>
void preShuffleScaleBuffer_gfx950(const ScaleType* src,
                                  ScaleType* packed,
                                  ck_tile::index_t MN,
                                  ck_tile::index_t K_scale,
                                  bool kLast)
{
    const ck_tile::long_index_t MN_packed             = MN / MNPack;
    const ck_tile::long_index_t K_packed              = K_scale / KPack;
    constexpr ck_tile::long_index_t NumScalesPerDword = 4 / sizeof(ScaleType);

    for(ck_tile::long_index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
    {
        for(ck_tile::long_index_t packed_k = 0; packed_k < K_packed; packed_k++)
        {
            ck_tile::long_index_t mn_lane  = packed_mn % XdlMNThread;
            ck_tile::long_index_t mn_group = packed_mn / XdlMNThread;
            ck_tile::long_index_t k_lane   = packed_k % XdlKThread;
            ck_tile::long_index_t k_group  = packed_k / XdlKThread;
            for(ck_tile::long_index_t ik = 0; ik < KPack; ik++)
            {
                for(ck_tile::long_index_t imn = 0; imn < MNPack; imn++)
                {
                    ck_tile::long_index_t byteIdx = ik * MNPack + imn;
                    ck_tile::long_index_t orig_mn =
                        mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                    ck_tile::long_index_t orig_k =
                        k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                    ck_tile::long_index_t inputIndex =
                        kLast ? orig_k + orig_mn * K_scale : orig_mn + orig_k * MN;
                    ScaleType v = src[inputIndex];
                    ck_tile::long_index_t outputIndex =
                        byteIdx + (packed_mn % XdlMNThread) * NumScalesPerDword +
                        packed_k * XdlMNThread * NumScalesPerDword +
                        (packed_mn / XdlMNThread) * XdlMNThread * NumScalesPerDword * K_packed;
                    packed[outputIndex] = v;
                }
            }
        }
    }
}

template <ck_tile::index_t NWarp,
          ck_tile::index_t NPerBlock,
          ck_tile::index_t XdlMNThread,
          typename ScaleType>
auto preShuffleScaleBufferPermuteN_gfx950(
    const ScaleType* src, ScaleType* shuffled, ck_tile::index_t MN, ck_tile::index_t K, bool kLast)
{
    constexpr ck_tile::long_index_t MNXdlPack  = 2;
    constexpr ck_tile::long_index_t KXdlPack   = 2;
    constexpr ck_tile::long_index_t NRepeat    = NPerBlock / NWarp / XdlMNThread;
    constexpr ck_tile::long_index_t XdlKThread = ck_tile::get_warp_size() / XdlMNThread;

    if(K % (KXdlPack * XdlKThread) != 0)
    {
        throw std::runtime_error("wrong! K must be a multiple of (KXdlPack * XdlKThread)");
    }
    const ck_tile::long_index_t K0 = K / KXdlPack / XdlKThread;

    for(ck_tile::long_index_t n = 0; n < MN; ++n)
    {
        for(ck_tile::long_index_t k = 0; k < K; ++k)
        {
            const ck_tile::long_index_t n0     = n / NPerBlock;
            const ck_tile::long_index_t tempn0 = n % NPerBlock;
            const ck_tile::long_index_t n1     = tempn0 / (XdlMNThread * NRepeat);
            const ck_tile::long_index_t tempn1 = tempn0 % (XdlMNThread * NRepeat);
            const ck_tile::long_index_t n2     = tempn1 / (NRepeat);
            const ck_tile::long_index_t tempn2 = tempn1 % (NRepeat);
            const ck_tile::long_index_t n3     = tempn2 % MNXdlPack;
            const ck_tile::long_index_t n4     = tempn2 / MNXdlPack;

            const ck_tile::long_index_t k0    = k / (XdlKThread * KXdlPack);
            const ck_tile::long_index_t tempk = k % (XdlKThread * KXdlPack);
            const ck_tile::long_index_t k1    = tempk % XdlKThread;
            const ck_tile::long_index_t k2    = tempk / XdlKThread;

            const ck_tile::long_index_t outputIndex =
                n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 * NWarp *
                    (NRepeat / MNXdlPack) +
                n1 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                n2 * MNXdlPack * KXdlPack + k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                k1 * MNXdlPack * KXdlPack * XdlMNThread + k2 * MNXdlPack +
                n4 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 * NWarp + n3;

            ck_tile::long_index_t inputIndex = kLast ? k + n * K : n + k * MN;

            if(n < MN)
            {
                shuffled[outputIndex] = src[inputIndex];
            }
            else
            {
                shuffled[outputIndex] = ScaleType{};
            }
        }
    }

    return shuffled;
}

} // namespace ck_tile
