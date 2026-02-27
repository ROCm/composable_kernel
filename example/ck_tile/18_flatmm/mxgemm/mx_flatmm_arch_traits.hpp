// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile {
namespace core {
namespace arch {

// Use the amdgcn_target_id enum from arch.hpp
using TargetId = amdgcn_target_id;

} // namespace arch
} // namespace core
} // namespace ck_tile

// Base FlatmmConfig with 16x16 warp tile (for non-GFX1250)
struct MXFlatmmConfigBase16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct MXfp4_FlatmmConfig16 : public MXFlatmmConfigBase16
{
    static constexpr ck_tile::index_t N_Tile = 512;
};

// Architecture traits for MX Flatmm - Primary template (gfx950 implementation)
template <ck_tile::core::arch::TargetId Arch, typename FlatmmConfig>
struct MXFlatmmArchTraits
{
    static constexpr int BlockedXDLN_PerWarp = 2; // determined by scale shuffle pattern

    using Config = FlatmmConfig;

    template <typename MXPipelineProblem>
    using MXFlatmmPipeline = ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<MXPipelineProblem>;

    static constexpr int GetNLane() { return Config::N_Warp_Tile; }

    template <bool KLast, typename dtype>
    static auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];

        size_t MNXdlPack   = 2;
        size_t KXdlPack    = 2;
        size_t XdlMNThread = Config::N_Warp_Tile; // 16
        size_t XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

        const auto MN_Paded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);

        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

        size_t K0 = K / KXdlPack / XdlKThread; // KRepeat

        // The 4 16x128 building blocks will be packed into 1 32x256 for F4
        // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

        // unfold the MN32xK(256/32) scale buffer
        //    4            16             2           2
        // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
        // Then, MNRepeat->KRepeat

        for(size_t n = 0; n < MN_Paded; ++n)
        {
            for(size_t k = 0; k < K; ++k)
            {
                auto n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
                auto tempn = n % (XdlMNThread * MNXdlPack);
                auto n1    = tempn % XdlMNThread; // i XdlMNThread
                auto n2    = tempn / XdlMNThread; // i MNXdlPack

                auto k0    = k / (XdlKThread * KXdlPack); // i KRepeat
                auto tempk = k % (XdlKThread * KXdlPack);
                auto k1    = tempk % XdlKThread; // i XdlKThread
                auto k2    = tempk / XdlKThread; // i KXdlPack

                auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                                   k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                                   k1 * MNXdlPack * KXdlPack * XdlMNThread +
                                   n1 * MNXdlPack * KXdlPack + k2 * MNXdlPack + n2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
            }
        }
        return shuffled;
    }
};

using MXFlatmm_GFX950_FP4FP4_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXfp4_FlatmmConfig16>;
using MXFlatmm_GFX950_FP8FP8_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;
using MXFlatmm_GFX950_FP6FP6_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;
using MXFlatmm_GFX950_FP8FP4_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;
using MXFlatmm_GFX950_FP4FP8_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;
