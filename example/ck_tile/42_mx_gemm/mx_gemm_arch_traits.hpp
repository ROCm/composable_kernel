// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "mx_gemm.hpp"

template <typename GemmConfig>
struct MXGemmArchTraits
{
    using Config = GemmConfig;

    template <bool KLast, typename dtype>
    static auto preShuffleScale(const ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];

        constexpr std::size_t MNXdlPack   = 2;
        constexpr std::size_t KXdlPack    = 2;
        constexpr std::size_t XdlMNThread = Config::N_Warp_Tile;
        constexpr std::size_t XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

        const auto MNPadded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);
        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor(
            {static_cast<std::size_t>(MNPadded * K)}, {static_cast<std::size_t>(1)}));

        const std::size_t K0 = K / KXdlPack / XdlKThread;

        for(std::size_t n = 0; n < static_cast<std::size_t>(MNPadded); ++n)
        {
            for(std::size_t k = 0; k < static_cast<std::size_t>(K); ++k)
            {
                const auto n0    = n / (XdlMNThread * MNXdlPack);
                const auto tempn = n % (XdlMNThread * MNXdlPack);
                const auto n1    = tempn % XdlMNThread;
                const auto n2    = tempn / XdlMNThread;

                const auto k0    = k / (XdlKThread * KXdlPack);
                const auto tempk = k % (XdlKThread * KXdlPack);
                const auto k1    = tempk % XdlKThread;
                const auto k2    = tempk / XdlKThread;

                const auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                                         k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                                         k1 * MNXdlPack * KXdlPack * XdlMNThread +
                                         n1 * MNXdlPack * KXdlPack + k2 * MNXdlPack + n2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(k, n) : dtype{};
            }
        }

        return shuffled;
    }
};
