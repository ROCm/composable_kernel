// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

/// @brief Pre-shuffle scale buffer for gfx1250 wmma mx scale instruction.
///
/// Reorganizes the scale data from row-major (MN x K) layout to the hardware-specific
/// layout expected by the gfx1250 wmma instruction.
///
/// @tparam ScaleType Scale data type (e.g., e8m0_t)
/// @tparam ScaleBlockSize The block size for microscaling (e.g., 32)
/// @tparam KStride Whether K is the fast-moving dimension
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                   ScaleType* dst,
                                   ck_tile::index_t MN,
                                   ck_tile::index_t K)
{
    static_assert((ScaleBlockSize == 32 || ScaleBlockSize == 16) && sizeof(ScaleType) == 1,
                  "wrong! only support 8-bit scale with ScaleBlockSize=32 or 16");

    // ScaleBlockSize == 16: the natural row-major scale layout already matches the gfx1250
    // wmma scale distribution (one e8m0 per 16 K-elements lands warp-aligned), so the
    // device-side shuffle is the identity transform for all K.
    if constexpr(ScaleBlockSize == 16)
    {
        for(ck_tile::long_index_t mn = 0; mn < MN; ++mn)
            for(ck_tile::long_index_t k = 0; k < K; ++k)
            {
                if constexpr(KStride)
                    dst[mn * K + k] = src[mn * K + k];
                else
                    dst[mn * K + k] = src[k * MN + mn];
            }
        return;
    }

    constexpr ck_tile::long_index_t MPerXdlops = 16;
    constexpr ck_tile::long_index_t KPerXdlops = 128;

    ck_tile::long_index_t MNPack = 2;
    ck_tile::long_index_t KPack  = 1;

    ck_tile::long_index_t MNStep = MPerXdlops;
    ck_tile::long_index_t KStep  = KPerXdlops / ScaleBlockSize;

    ck_tile::long_index_t K0 = K / KPack / KStep;

    for(ck_tile::long_index_t mn = 0; mn < MN; ++mn)
    {
        ck_tile::long_index_t iMNRepeat = mn / (MNStep * MNPack);
        ck_tile::long_index_t tempmn    = mn % (MNStep * MNPack);

        for(ck_tile::long_index_t k = 0; k < K; ++k)
        {
            ck_tile::long_index_t iKRepeat = k / (KStep * KPack);
            ck_tile::long_index_t tempk    = k % (KStep * KPack);

            ck_tile::long_index_t outputIndex =
                (iMNRepeat * MNPack * MNStep) * (KStep * KPack * K0) +
                (iKRepeat * KStep * KPack) * (MNStep * MNPack) + tempmn * (KStep * KPack) + tempk;

            if constexpr(KStride)
            {
                dst[outputIndex] = src[mn * K + k];
            }
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
