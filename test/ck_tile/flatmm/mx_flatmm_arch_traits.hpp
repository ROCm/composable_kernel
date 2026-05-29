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

using MXFp6FlatmmConfigBase16 = MXFlatmmConfigBase16;

// Base FlatmmConfig with 32x32 warp tile (for GFX1250 TDM)
struct MXFlatmmConfigBase32TDM
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
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

// Architecture traits for MX Flatmm - Primary template (gfx950 implementation)
template <ck_tile::core::arch::TargetId Arch, typename FlatmmConfig>
struct MXFlatmmArchTraits
{
    static constexpr int BlockedXDLN_PerWarp = 2; // (NXdlPack) determined by scale shuffle pattern

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

template <typename FlatmmConfig>
struct MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, FlatmmConfig>
    : public MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, FlatmmConfig>
{

    template <bool KLast, typename dtype>
    static auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];
        // K  -> K/KPack,     KPack(KPack is used to make sure int32 alignment)
        // MN -> MN/WarpSize, WarpSize
        //  MN/WarpSize, K/KPack, WarpSize, KPack
        size_t KPack = sizeof(int32_t) / sizeof(dtype); // scale always use fp8; KPack = 4
        size_t K0    = K / KPack;
        size_t M1    = 32; // this is used to align 32x32x128 block scaled wmma

        const auto MN_Paded = ck_tile::integer_least_multiple(MN, 32);

        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

        for(size_t n = 0; n < MN_Paded; ++n)
        {
            for(size_t k = 0; k < K; ++k)
            {
                auto n0 = n / M1; // i MNRepeat
                auto n1 = n % M1; // i M1

                auto k0 = k / KPack; // i KRepeat
                auto k2 = k % KPack; // i K1

                auto outputIndex = n0 * K0 * M1 * KPack + k0 * M1 * KPack + n1 * KPack + k2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
            }
        }
        return shuffled;
    }
};

template <ck_tile::core::arch::TargetId Arch, typename FlatmmConfig>
struct MXFlatmmTDMArchTraits;

// Architecture traits for MX Flatmm - GFX1250 TDM
template <typename FlatmmConfig>
struct MXFlatmmTDMArchTraits<ck_tile::core::arch::TargetId::GFX1250, FlatmmConfig>
{
    static constexpr int BlockedXDLN_PerWarp = 1;

    using Config = FlatmmConfig;

    template <typename MXPipelineProblem>
    using MXFlatmmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegTDM<MXPipelineProblem>;

    static constexpr int GetNLane()
    {
        // gfx1250 uses 32x32x128 wmma, but still use 16 NLanes for weight preshuffle
        return 16;
    }

    template <bool KLast, typename dtype>
    static auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];
        // K  -> K/KPack,     KPack(KPack is used to make sure int32 alignment)
        // MN -> MN/WarpSize, WarpSize
        //  MN/WarpSize, K/KPack, WarpSize, KPack
        size_t KPack = sizeof(int32_t) / sizeof(dtype); // scale always use fp8; KPack = 4
        size_t K0    = K / KPack;
        size_t M1    = 32; // this is used to align 32x32x128 block scaled wmma

        const auto MN_Paded = ck_tile::integer_least_multiple(MN, 32);

        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

        for(size_t n = 0; n < MN_Paded; ++n)
        {
            for(size_t k = 0; k < K; ++k)
            {
                auto n0 = n / M1; // i MNRepeat
                auto n1 = n % M1; // i M1

                auto k0 = k / KPack; // i KRepeat
                auto k2 = k % KPack; // i K1

                auto outputIndex = n0 * K0 * M1 * KPack + k0 * M1 * KPack + n1 * KPack + k2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
            }
        }
        return shuffled;
    }
};

using MXFlatmmTDM_GFX1250_FP4FP4_Traits =
    MXFlatmmTDMArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase32TDM>;
using MXFlatmmTDM_GFX1250_FP8FP8_Traits =
    MXFlatmmTDMArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase32TDM>;
using MXFlatmmTDM_GFX1250_FP8FP4_Traits =
    MXFlatmmTDMArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase32TDM>;
using MXFlatmmTDM_GFX1250_FP4FP8_Traits =
    MXFlatmmTDMArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase32TDM>;

using MXFlatmm_GFX1250_FP4FP4_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXfp4_FlatmmConfig16>;
using MXFlatmm_GFX1250_FP8FP8_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase16>;
using MXFlatmm_GFX1250_FP6FP6_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase16>;
using MXFlatmm_GFX1250_FP8FP4_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase16>;
using MXFlatmm_GFX1250_FP4FP8_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX1250, MXFlatmmConfigBase16>;

// Helper to get current target ID based on compile-time macros
constexpr ck_tile::core::arch::TargetId GetCurrentTargetId()
{
#if defined(CK_USE_GFX1250)
    return ck_tile::core::arch::TargetId::GFX1250;
#else
    return ck_tile::core::arch::TargetId::GFX950; // Default fallback
#endif
}
