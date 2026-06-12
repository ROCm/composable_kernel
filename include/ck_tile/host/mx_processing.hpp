// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
// Each int32_t contains MNPack * KPack e8m0_t values with byte layout matching
// the GPU tile distribution: values are XdlMNThread apart in M and XdlKThread apart in K.
// byte[ik * MNPack + imn] = e8m0 at strided (mn, k) position
// kLast=true for A scales (layout [M, K/32]), kLast=false for B scales (layout [K/32, N])
template <index_t MNPack = 2, index_t KPack = 2, index_t XdlMNThread = 16, index_t XdlKThread = 4>
auto packScalesMNxK(const HostTensor<e8m0_t>& src, const bool kLast)
{
    auto src_lengths        = src.get_lengths();
    const index_t MN        = kLast ? src_lengths[0] : src_lengths[1];
    const index_t K_scale   = kLast ? src_lengths[1] : src_lengths[0];
    const index_t MN_packed = MN / MNPack;
    const index_t K_packed  = K_scale / KPack;

    // Output as flat vector of int32_t (row-major [MN/MNPack, K/32/KPack])
    HostTensor<int32_t> packed(HostTensorDescriptor(
        {static_cast<std::size_t>(MN_packed), static_cast<std::size_t>(K_packed)},
        {static_cast<std::size_t>(K_packed), static_cast<std::size_t>(1)}));

    for(index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
    {
        for(index_t packed_k = 0; packed_k < K_packed; packed_k++)
        {
            uint32_t val     = 0;
            index_t mn_lane  = packed_mn % XdlMNThread;
            index_t mn_group = packed_mn / XdlMNThread;
            index_t k_lane   = packed_k % XdlKThread;
            index_t k_group  = packed_k / XdlKThread;
            for(index_t ik = 0; ik < KPack; ik++)
            {
                for(index_t imn = 0; imn < MNPack; imn++)
                {
                    index_t byteIdx = ik * MNPack + imn;
                    index_t orig_mn = mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                    index_t orig_k  = k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                    e8m0_t v = kLast ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                    val |= (static_cast<uint32_t>(v.get()) << (byteIdx * 8));
                }
            }
            packed(packed_mn, packed_k) = static_cast<int32_t>(val);
        }
    }
    return packed;
}

template <index_t XdlMNThread, typename dtype>
auto preShuffleScale(ck_tile::HostTensor<dtype>& src, const bool kLast)
{
    auto src_lengths = src.get_lengths();
    const index_t MN = kLast ? src_lengths[0] : src_lengths[1];
    const index_t K  = kLast ? src_lengths[1] : src_lengths[0];

    constexpr index_t MNXdlPack  = 2;
    constexpr index_t KXdlPack   = 2;
    constexpr index_t XdlKThread = get_warp_size() / XdlMNThread;

    const auto MNPadded = integer_least_multiple(MN, XdlMNThread * MNXdlPack);
    HostTensor<dtype> shuffled(HostTensorDescriptor({static_cast<std::size_t>(MNPadded * K)},
                                                    {static_cast<std::size_t>(1)}));

    if(K % (KXdlPack * XdlKThread) != 0)
    {
        throw std::runtime_error("wrong! K must be a multiple of (KXdlPack * XdlKThread)");
    }

    const index_t K0 = K / KXdlPack / XdlKThread;

    for(index_t n = 0; n < MNPadded; ++n)
    {
        for(index_t k = 0; k < K; ++k)
        {
            const index_t n0    = n / (XdlMNThread * MNXdlPack);
            const index_t tempn = n % (XdlMNThread * MNXdlPack);
            const index_t n1    = tempn % XdlMNThread;
            const index_t n2    = tempn / XdlMNThread;

            const index_t k0    = k / (XdlKThread * KXdlPack);
            const index_t tempk = k % (XdlKThread * KXdlPack);
            const index_t k1    = tempk % XdlKThread;
            const index_t k2    = tempk / XdlKThread;

            const index_t outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                                        k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                                        k1 * MNXdlPack * KXdlPack * XdlMNThread +
                                        n1 * MNXdlPack * KXdlPack + k2 * MNXdlPack + n2;

            if(n < MN)
            {
                shuffled(outputIndex) = kLast ? src(n, k) : src(k, n);
            }
            else
            {
                shuffled(outputIndex) = dtype{};
            }
        }
    }

    return shuffled;
}

template <index_t NWarp, index_t NPerBlock, index_t XdlMNThread, typename dtype>
auto preShuffleScalePermuteN(const HostTensor<dtype>& src, const bool kLast)
{
    auto src_lengths = src.get_lengths();
    const index_t MN = kLast ? src_lengths[0] : src_lengths[1];
    const index_t K  = kLast ? src_lengths[1] : src_lengths[0];

    constexpr index_t MNXdlPack  = 2;
    constexpr index_t KXdlPack   = 2;
    constexpr index_t NRepeat    = NPerBlock / NWarp / XdlMNThread;
    constexpr index_t XdlKThread = get_warp_size() / XdlMNThread; // 4

    const index_t MNPadded = integer_least_multiple(MN, NPerBlock);
    HostTensor<dtype> shuffled(HostTensorDescriptor({static_cast<std::size_t>(MNPadded * K)},
                                                    {static_cast<std::size_t>(1)}));

    if(K % (KXdlPack * XdlKThread) != 0)
    {
        throw std::runtime_error("wrong! K must be a multiple of (KXdlPack * XdlKThread)");
    }
    const index_t K0 = K / KXdlPack / XdlKThread;

    for(index_t n = 0; n < MNPadded; ++n)
    {
        for(index_t k = 0; k < K; ++k)
        {
            const index_t n0     = n / NPerBlock;
            const index_t tempn0 = n % NPerBlock;
            const index_t n1     = tempn0 / (XdlMNThread * NRepeat);
            const index_t tempn1 = tempn0 % (XdlMNThread * NRepeat);
            const index_t n2     = tempn1 / (NRepeat);
            const index_t tempn2 = tempn1 % (NRepeat);
            const index_t n3     = tempn2 % MNXdlPack;
            const index_t n4     = tempn2 / MNXdlPack;

            const index_t k0    = k / (XdlKThread * KXdlPack);
            const index_t tempk = k % (XdlKThread * KXdlPack);
            const index_t k1    = tempk % XdlKThread;
            const index_t k2    = tempk / XdlKThread;

            const index_t outputIndex =
                n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 * NWarp *
                    (NRepeat / MNXdlPack) +
                n1 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                n2 * MNXdlPack * KXdlPack + k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                k1 * MNXdlPack * KXdlPack * XdlMNThread + k2 * MNXdlPack +
                n4 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 * NWarp + n3;

            if(n < MN)
            {
                shuffled(outputIndex) = kLast ? src(n, k) : src(k, n);
            }
            else
            {
                shuffled(outputIndex) = dtype{};
            }
        }
    }

    return shuffled;
}

} // namespace ck_tile
