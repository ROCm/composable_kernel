// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

// can remove all bank conflicts, but drop the performance for some cases
// Probably it is limited by compiler optimization.
#define CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD 0
namespace ck_tile {

namespace detail {

// Keep the byte-based padding contract local to FMHA: unlike GEMM's policy, which consumes raw
// hardware fields, qr_tdm uses one configuration to couple writer encoding, reader descriptors,
// and LDS capacity.
inline constexpr index_t kQrTdmLdsAccessBytes = 16;

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t integer_log2_exact()
{
    static_assert(X > 0 && (X & (X - 1)) == 0,
                  "integer_log2_exact requires a positive power of two");

    index_t value  = X;
    index_t result = 0;
    while(value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

template <bool Enabled, index_t IntervalBytes, index_t PadBytes>
inline constexpr bool is_valid_lds_padding_config_v =
    (!Enabled && IntervalBytes == 0 && PadBytes == 0) ||
    (Enabled && IntervalBytes >= 8 && IntervalBytes <= 1024 && IntervalBytes % 4 == 0 &&
     (IntervalBytes & (IntervalBytes - 1)) == 0 && PadBytes >= 4 && PadBytes <= 512 &&
     PadBytes % 4 == 0);

template <bool Enabled_, index_t IntervalBytes_, index_t PadBytes_>
struct LdsPaddingConfig
{
    static_assert(is_valid_lds_padding_config_v<Enabled_, IntervalBytes_, PadBytes_>,
                  "invalid LDS padding configuration");

    static constexpr bool kEnabled          = Enabled_;
    static constexpr index_t kIntervalBytes = IntervalBytes_;
    static constexpr index_t kPadBytes      = PadBytes_;
};

template <typename Config>
struct EncodedTdmPadding
{
    private:
    static constexpr index_t raw_pad_interval = [] {
        if constexpr(Config::kEnabled)
        {
            return integer_log2_exact<Config::kIntervalBytes / 4>() - 1;
        }
        else
        {
            return 0;
        }
    }();

    static constexpr index_t raw_pad_amount = [] {
        if constexpr(Config::kEnabled)
        {
            return Config::kPadBytes / 4 - 1;
        }
        else
        {
            return 0;
        }
    }();

    static_assert(!Config::kEnabled || (raw_pad_interval >= 0 && raw_pad_interval <= 7),
                  "TDM padding interval field overflows");
    static_assert(!Config::kEnabled || (raw_pad_amount >= 0 && raw_pad_amount <= 127),
                  "TDM padding amount field overflows");
    static_assert(!Config::kEnabled ||
                      ((index_t{1} << (raw_pad_interval + 1)) * 4 == Config::kIntervalBytes),
                  "TDM padding interval does not round-trip");
    static_assert(!Config::kEnabled || ((raw_pad_amount + 1) * 4 == Config::kPadBytes),
                  "TDM padding amount does not round-trip");

    public:
    static constexpr bool kEnabled        = Config::kEnabled;
    static constexpr index_t kPadInterval = raw_pad_interval;
    static constexpr index_t kPadAmount   = raw_pad_amount;
};

template <
    typename DataType,
    index_t Rows,
    index_t Cols,
    typename PaddingConfig,
    index_t AccessBytes,
    typename std::enable_if<!PaddingConfig::kEnabled || numeric_traits<DataType>::PackedSize == 1,
                            bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_row_major_lds_descriptor()
{
    static_assert(Rows > 0 && Cols > 0, "LDS descriptor dimensions must be positive");
    static_assert(AccessBytes > 0 &&
                      (AccessBytes * numeric_traits<DataType>::PackedSize) % sizeof(DataType) == 0,
                  "LDS access width must contain a whole number of elements");

    constexpr index_t AccessElements =
        AccessBytes * numeric_traits<DataType>::PackedSize / sizeof(DataType);

    if constexpr(!PaddingConfig::kEnabled)
    {
        return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                            make_tuple(number<Cols>{}, number<1>{}),
                                            number<AccessElements>{},
                                            number<1>{});
    }
    else
    {
        static_assert(numeric_traits<DataType>::PackedSize == 1,
                      "qr_tdm LDS padding does not support packed data types");

        constexpr index_t ElementBytes  = sizeof(DataType);
        constexpr index_t LogicalBytes  = Rows * Cols * ElementBytes;
        constexpr index_t IntervalBytes = PaddingConfig::kIntervalBytes;
        constexpr index_t PadBytes      = PaddingConfig::kPadBytes;

        static_assert(IntervalBytes % ElementBytes == 0,
                      "LDS padding interval must contain a whole number of elements");
        static_assert(PadBytes % ElementBytes == 0,
                      "LDS padding amount must contain a whole number of elements");
        static_assert(LogicalBytes % IntervalBytes == 0,
                      "padded LDS descriptor requires complete logical intervals");

        constexpr index_t ElementsPerInterval = IntervalBytes / ElementBytes;
        constexpr index_t PadElements         = PadBytes / ElementBytes;
        constexpr index_t NumIntervals        = LogicalBytes / IntervalBytes;

        constexpr auto interval_desc = make_naive_tensor_descriptor(
            make_tuple(number<NumIntervals>{}, number<ElementsPerInterval>{}),
            make_tuple(number<ElementsPerInterval + PadElements>{}, number<1>{}),
            number<AccessElements>{},
            number<1>{});

        constexpr auto flat_desc = transform_tensor_descriptor(
            interval_desc,
            make_tuple(make_merge_transform_v3_division_mod(
                make_tuple(number<NumIntervals>{}, number<ElementsPerInterval>{}))),
            make_tuple(sequence<0, 1>{}),
            make_tuple(sequence<0>{}));

        return transform_tensor_descriptor(
            flat_desc,
            make_tuple(make_unmerge_transform(make_tuple(number<Rows>{}, number<Cols>{}))),
            make_tuple(sequence<0>{}),
            make_tuple(sequence<0, 1>{}));
    }
}

template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmLdsArenaLayout;

inline constexpr bool is_qr_tdm_padding_supported_arch_v =
#if(defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)) || \
    (!defined(__HIP_DEVICE_COMPILE__) && defined(CK_USE_GFX1250))
    true;
#else
    false;
#endif

template <typename Problem>
inline constexpr bool is_qr_tdm_padding_enabled_problem_v =
    is_qr_tdm_padding_supported_arch_v &&
    ((std::is_same_v<typename Problem::QDataType, bf16_t> &&
      std::is_same_v<typename Problem::KDataType, bf16_t> &&
      std::is_same_v<typename Problem::VDataType, bf16_t>) ||
     (std::is_same_v<typename Problem::QDataType, half_t> &&
      std::is_same_v<typename Problem::KDataType, half_t> &&
      std::is_same_v<typename Problem::VDataType, half_t>)) &&
    Problem::BlockFmhaShape::NumWarps == 4 &&
    (Problem::BlockFmhaShape::kM0 == 64 || Problem::BlockFmhaShape::kM0 == 128) &&
    Problem::BlockFmhaShape::kN0 == 64 && Problem::BlockFmhaShape::kK0 == 32 &&
    Problem::BlockFmhaShape::kN1 == 128 && Problem::BlockFmhaShape::kK1 == 32 &&
    Problem::BlockFmhaShape::kQKHeaddim == 128 && Problem::BlockFmhaShape::kSubQKHeaddim == 128 &&
    Problem::BlockFmhaShape::IsVLayoutRowMajor &&
    std::is_same_v<typename Problem::BlockFmhaShape::Gemm0BlockWarps, sequence<4, 1, 1>> &&
    std::is_same_v<typename Problem::BlockFmhaShape::Gemm1BlockWarps, sequence<4, 1, 1>> &&
    std::is_same_v<typename Problem::BlockFmhaShape::Gemm0WarpTile, sequence<16, 16, 32>> &&
    std::is_same_v<typename Problem::BlockFmhaShape::Gemm1WarpTile, sequence<16, 16, 32>> &&
    numeric_traits<typename Problem::QDataType>::PackedSize == 1 &&
    numeric_traits<typename Problem::KDataType>::PackedSize == 1 &&
    numeric_traits<typename Problem::VDataType>::PackedSize == 1;

template <typename Problem, bool Enabled = is_qr_tdm_padding_enabled_problem_v<Problem>>
struct QrTdmPaddingSelection
{
    using Q = LdsPaddingConfig<false, 0, 0>;
    using K = LdsPaddingConfig<false, 0, 0>;
    using V = LdsPaddingConfig<false, 0, 0>;
};

template <typename Problem>
struct QrTdmPaddingSelection<Problem, true>
{
    // Measured production configuration for gfx1250 BF16/FP16 d=128 qr_tdm.
    using Q = LdsPaddingConfig<false, 0, 0>;
    using K = LdsPaddingConfig<true, 256, 16>;
    using V = LdsPaddingConfig<true, 256, 32>;
};

} // namespace detail

// This pipeline is qkv all located in LDS, targeting gfx1250
struct BlockFmhaPipelineQRKSVSTdmDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ true,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopy = */ true,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    using LdsPaddingConfigQ = typename detail::QrTdmPaddingSelection<Problem>::Q;

    template <typename Problem>
    using LdsPaddingConfigK = typename detail::QrTdmPaddingSelection<Problem>::K;

    template <typename Problem>
    using LdsPaddingConfigV = typename detail::QrTdmPaddingSelection<Problem>::V;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout = detail::QrTdmLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        // gfx1250 wave32 + d>=128 needs KVector >= 4 to satisfy
        // base async distribution static_assert (WarpSize*KVector >= kKPerBlock).
        // Choose b128 (dwordx4) as starting point; b64 fallback if perf problems.
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
#if defined(__gfx125__)
        constexpr index_t MaxLoadSizeInBytes = 16; // dwordx4 = b128
#else
        constexpr index_t MaxLoadSizeInBytes = 4; // dword (matches base async default)
#endif
        return MaxLoadSizeInBytes * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        // Same rationale as GetAlignmentK; V is loaded with async copy too.
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
#if defined(__gfx125__)
        constexpr index_t MaxLoadSizeInBytes = 16; // dwordx4 = b128
#else
        constexpr index_t MaxLoadSizeInBytes = 4; // dword
#endif
        return MaxLoadSizeInBytes * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);
    }

    // Trivial tile-major Q dram dist (mirror of MakeKDramTileDistribution).
    // TDM writes LDS in box-major order; the Q LDS desc is plain row-major.
    // This distribution makes each thread's per-call footprint exactly one
    // contiguous (kMPerBlock/warpNum x kKPerBlock) tile, so the box-major
    // write lands on the row-major strip the reader expects. The QK GEMM
    // (Q as A operand) register distribution is unchanged.
    //
    // `BypassLDS` is kept for non-TDM call sites that bypass LDS entirely.
    template <typename Problem, bool BypassLDS = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        if constexpr(!BypassLDS)
        {
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;
            constexpr index_t warpNum    = kBlockSize / get_warp_size();

            static_assert(kMPerBlock % warpNum == 0,
                          "kMPerBlock must be divisible by warpNum for trivial tile-major Q dist");

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,                                    // R: empty
                    tuple<sequence<warpNum, kMPerBlock / warpNum>, // X[0]: M-axis, warp split
                          sequence<kKPerBlock>>, // X[1]: K-axis, single full vector
                    tuple<sequence<1>>,          // PsToRH (warp dim mapping)
                    tuple<sequence<0>>,          // PsToRH_lid
                    sequence<1, 2>,              // YsToD outer
                    sequence<1, 0>>{},           // YsToD inner
                bool_constant<true>{});          // IsWarpLevelParallelOnly
        }
        else
        {
            using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
            constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            constexpr auto q_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<2, 1>,
                sequence<0, 0>>{};

            constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

            return q_block_dstr;
        }
    }

    // Trivial tile-major K dram dist, mirrors the ColMajor-B path in
    // gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp (MakeBDramTileDistribution).
    //
    // TDM writes LDS in box-major order (each thread writes a contiguous box
    // of shape `box_dim`). The K LDS descriptor (MakeKLdsBlockDescriptor) is
    // plain row-major (kN0, kK0); the ds_load reader (MakeKRegTileDistribution)
    // is built from WarpGemm::BWarpDstrEncoding and assumes that contiguous-row
    // interface. The distribution below makes thread-i's per-call footprint
    // exactly one contiguous (kN0/warpNum x kK0) tile so the box-major write
    // lands on the row-major strip the reader expects. The QK GEMM (K as B
    // operand) register distribution is unchanged.
    //
    // `LoadOnce` selects kSubQKHeaddim instead of kK0 for the K-axis length,
    // matching the single-load (decode) variant of the kernel.
    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;
        constexpr index_t warpNum = kBlockSize / get_warp_size();

        static_assert(kNPerBlock % warpNum == 0,
                      "kNPerBlock must be divisible by warpNum for trivial tile-major K dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                    // R: empty
                tuple<sequence<warpNum, kNPerBlock / warpNum>, // X[0]: N-axis, warp split
                      sequence<kKPerBlock>>, // X[1]: K-axis, single full vector per thread
                tuple<sequence<1>>,          // PsToRH (warp dim mapping)
                tuple<sequence<0>>,          // PsToRH_lid
                sequence<1, 2>,              // YsToD outer
                sequence<1, 0>>{},           // YsToD inner
            bool_constant<true>{});          // IsWarpLevelParallelOnly
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto q_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

        return q_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return static_cast<index_t>(detail::kQrTdmLdsAccessBytes / sizeof(QDataType));
    }

    // Plain row-major Q LDS desc. TDM box-major write cannot produce an XOR'd
    // layout, so the Xor template param on the original generic descriptor
    // was unreachable from this pipeline and has been removed.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;
        using DataType               = typename Problem::QDataType;
        using Padding                = LdsPaddingConfigQ<Problem>;

        return detail::make_qr_tdm_row_major_lds_descriptor<DataType,
                                                            kMPerBlock,
                                                            kKPerBlock,
                                                            Padding,
                                                            detail::kQrTdmLdsAccessBytes>();
    }

    // Plain row-major K LDS desc; same no-swizzle rationale as Q above.
    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;

        using DataType = typename Problem::KDataType;
        using Padding  = LdsPaddingConfigK<Problem>;

        return detail::make_qr_tdm_row_major_lds_descriptor<DataType,
                                                            kNPerBlock,
                                                            kKPerBlock,
                                                            Padding,
                                                            detail::kQrTdmLdsAccessBytes>();
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        using DataType               = typename Problem::VDataType;
        using Padding                = LdsPaddingConfigV<Problem>;

        static_assert(!Xor, "qr_tdm V LDS descriptor must remain row-major");
        return detail::make_qr_tdm_row_major_lds_descriptor<DataType,
                                                            kKPerBlock,
                                                            kNPerBlock,
                                                            Padding,
                                                            detail::kQrTdmLdsAccessBytes>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::PDataType,
                               typename Problem::VDataType,
                               typename Problem::OaccDataType,
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false,
                               false,
                               ((Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 16 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32) ||
                                (Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 32 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 16))
                                   ? WGAttrNumAccessEnum::Double
                                   : WGAttrNumAccessEnum::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto k_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto k_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            k_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto k_block_dstr = make_static_tile_distribution(k_block_dstr_encode);

        return k_block_dstr;
    }

    // V dram dist for the TDM writer path. Trivial tile-major mirrors the
    // K dist (MakeKDramTileDistribution above): warp-split V's seq dim
    // (kKPerBlock=kN0) into warpNum chunks; each thread loads the full V
    // hdim vector (kNPerBlock=kN1). IsWarpLevelParallelOnly=true matches K.
    //
    // The baseline B1 5D async-style scatter dist is incompatible with TDM
    // box-major writes -- the dist projection scatters thread bytes to LDS
    // positions that ds_load_tr_b128 reads as garbage. The trivial tile-major
    // form produces a plain row-major LDS layout that the read view + standard
    // MakeVRegTileDistribution can consume correctly.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1; // V hdim,    128
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0; // V seq dim, 64
        constexpr index_t warpNum    = kBlockSize / get_warp_size();

        static_assert(kKPerBlock % warpNum == 0,
                      "V kN0 (seq) must be divisible by warpNum for trivial tile-major V dist");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                    // R: empty
                tuple<sequence<warpNum, kKPerBlock / warpNum>, // X[0]: V seq dim, warp split (8, 8)
                      sequence<kNPerBlock>>, // X[1]: V hdim, single full vector per thread (128)
                tuple<sequence<1>>,          // PsToRH: warp dim mapping
                tuple<sequence<0>>,          // PsToRH_lid
                sequence<1, 2>,              // YsToD outer
                sequence<1, 0>>{},           // YsToD inner
            bool_constant<true>{});          // IsWarpLevelParallelOnly
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto p_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto p_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            p_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto p_block_dstr = make_static_tile_distribution(p_block_dstr_encode);

        return p_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto v_block_dstr =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(v_block_dstr_encode),
                                          typename Problem::VDataType>::TransposedDstrEncode{});

        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemNPackS()
    {
        using SDataType = remove_cvref_t<typename Problem::SaccDataType>;
        return static_cast<index_t>(16 / sizeof(SDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = GetSmemNPackS<Problem>();

        constexpr auto s_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        constexpr auto s_lds_block_desc = transform_tensor_descriptor(
            s_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return s_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTileK     = Problem::BlockFmhaShape::kN0;

        // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
        constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = kKPerBlock / (K2 * K3);
        constexpr index_t K0 = kTileK / kKPerBlock;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        constexpr auto s2_block_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 2>>,
                                       sequence<1, 2, 2, 2>,
                                       sequence<0, 0, 1, 3>>{};

        constexpr auto s2_block_dstr = make_static_tile_distribution(s2_block_dstr_encoding);

        return s2_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem, LoadOnce>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeS()
    {
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        return NWarp > 1 ? MakeSLdsBlockDescriptor<Problem>().get_element_space_size() *
                               sizeof(typename Problem::SaccDataType)
                         : 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr ck_tile::index_t kM0 = Problem::BlockFmhaShape::kM0;
        if constexpr(kM0 > 64)
        {
            // Prefill: same layout as qr_async_trload kernel allocations.
            // Two K buffers (ping/pong) + two V buffers (ping/pong).
            return 2 * GetSmemSizeK<Problem, true>() + 2 * GetSmemSizeV<Problem>();
        }
        else
        {
            // Decode: single buffer; Q, K, S, V laid out sequentially.
            return max(GetSmemSizeQ<Problem>(),
                       GetSmemSizeK<Problem>() + GetSmemSizeS<Problem>() + GetSmemSizeV<Problem>());
        }
    }
};

namespace detail {

// One 256-byte-aligned arena backs every qr_tdm LDS region. Q intentionally aliases K0 at offset
// zero and, on decode, overlaps the later S/V regions: the pipeline loads Q into registers and
// completes the tensor-count barrier before K/V overwrite that storage. For the measured d=128
// K+V configuration the layouts are:
//   prefill: Q/K0=0, K1=17408, V0=34816, V1=53248, end=71680
//   decode:  Q/K0=0, S=4336, V0=4352, end=22784
// Region alignment follows one gfx1250 LDS bank row.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmLdsArenaLayout
{
    using Shape = typename Problem::BlockFmhaShape;

    static constexpr bool kDoubleBuffer       = Shape::kM0 > 64;
    static constexpr index_t kArenaAlignment  = 256;
    static constexpr index_t kRegionAlignment = 256;
    static constexpr index_t kSRequiredAlignment =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemNPackS<Problem>() *
        sizeof(typename Problem::SaccDataType);

    static constexpr auto q_descriptor =
        make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                             Shape::kM0,
                                             Shape::kSubQKHeaddim,
                                             QPadding,
                                             kQrTdmLdsAccessBytes>();
    static constexpr auto k_descriptor = make_qr_tdm_row_major_lds_descriptor <
                                         typename Problem::KDataType,
                          Shape::kN0, kDoubleBuffer ? Shape::kSubQKHeaddim : Shape::kK0, KPadding,
                          kQrTdmLdsAccessBytes > ();
    static constexpr auto v_descriptor =
        make_qr_tdm_row_major_lds_descriptor<typename Problem::VDataType,
                                             Shape::kN0,
                                             Shape::kN1,
                                             VPadding,
                                             kQrTdmLdsAccessBytes>();

    static constexpr index_t kQBytes =
        q_descriptor.get_element_space_size() * sizeof(typename Problem::QDataType);
    static constexpr index_t kKBytes =
        k_descriptor.get_element_space_size() * sizeof(typename Problem::KDataType);
    static constexpr index_t kVBytes =
        v_descriptor.get_element_space_size() * sizeof(typename Problem::VDataType);
    static constexpr index_t kSBytes =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemSizeS<Problem>();

    static constexpr index_t kQOffset  = 0;
    static constexpr index_t kK0Offset = 0;
    static constexpr index_t kK1Offset =
        kDoubleBuffer ? integer_least_multiple(kK0Offset + kKBytes, kRegionAlignment) : 0;
    static constexpr index_t kSOffset =
        kDoubleBuffer ? 0 : integer_least_multiple(kK0Offset + kKBytes, kSRequiredAlignment);
    static constexpr index_t kV0Offset = [] {
        if constexpr(kDoubleBuffer)
        {
            constexpr index_t kKRegionEnd = kK1Offset + kKBytes;
            return integer_least_multiple(max(kKRegionEnd, kQBytes), kRegionAlignment);
        }
        else
        {
            return integer_least_multiple(kSOffset + kSBytes, kRegionAlignment);
        }
    }();
    static constexpr index_t kV1Offset =
        kDoubleBuffer ? integer_least_multiple(kV0Offset + kVBytes, kRegionAlignment) : kV0Offset;
    static constexpr index_t kArenaBytes = [] {
        if constexpr(kDoubleBuffer)
        {
            return integer_least_multiple(kV1Offset + kVBytes, kArenaAlignment);
        }
        else
        {
            return integer_least_multiple(max(kQBytes, kV0Offset + kVBytes), kArenaAlignment);
        }
    }();

    static constexpr bool kHasProductionAlignment =
        kQOffset % kRegionAlignment == 0 && kK0Offset % kRegionAlignment == 0 &&
        (!kDoubleBuffer || kK1Offset % kRegionAlignment == 0) &&
        kV0Offset % kRegionAlignment == 0 && (!kDoubleBuffer || kV1Offset % kRegionAlignment == 0);

    static_assert(kHasProductionAlignment);
    static_assert(kQOffset + kQBytes <= kArenaBytes);
    static_assert(kV0Offset + kVBytes <= kArenaBytes);
    static_assert(!kDoubleBuffer || kK0Offset + kKBytes <= kK1Offset);
    static_assert(!kDoubleBuffer || kK1Offset + kKBytes <= kV0Offset);
    static_assert(!kDoubleBuffer || kQOffset + kQBytes <= kV0Offset);
    static_assert(kDoubleBuffer || kK0Offset + kKBytes <= kSOffset);
    static_assert(kDoubleBuffer || kSOffset + kSBytes <= kV0Offset);
    static_assert(!kDoubleBuffer || !KPadding::kEnabled ||
                  (kK1Offset - kK0Offset) % KPadding::kIntervalBytes == 0);
    static_assert(!kDoubleBuffer || !VPadding::kEnabled ||
                  (kV1Offset - kV0Offset) % VPadding::kIntervalBytes == 0);
    static_assert(kArenaBytes <= 128 * 1024);
    static_assert(integer_least_multiple(kArenaBytes, 64 * 1024) * 2 <= 320 * 1024);
};

template <typename TensorTag, typename Problem, bool LoadOnce>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_writer_distribution()
{
    using Policy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;

    if constexpr(TensorTag::Id == 0)
    {
        return Policy::template MakeQDramTileDistribution<Problem>();
    }
    else if constexpr(TensorTag::Id == 1)
    {
        return Policy::template MakeKDramTileDistribution<Problem, LoadOnce>();
    }
    else
    {
        static_assert(TensorTag::Id == 2, "unknown qr_tdm tensor tag");
        return Policy::template MakeVDramTileDistribution<Problem>();
    }
}

template <typename TensorTag, typename Problem>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_reader_distribution()
{
    using Policy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;

    if constexpr(TensorTag::Id == 0)
    {
        return Policy::template MakeQRegTileDistribution<Problem>();
    }
    else if constexpr(TensorTag::Id == 1)
    {
        return Policy::template MakeKRegTileDistribution<Problem>();
    }
    else
    {
        static_assert(TensorTag::Id == 2, "unknown qr_tdm tensor tag");
        return Policy::template MakeVRegTileDistribution<Problem>();
    }
}

template <typename TensorTag, typename Problem, bool LoadOnce = false>
CK_TILE_HOST_DEVICE constexpr bool validate_qr_tdm_issue_geometry()
{
    using Shape        = typename Problem::BlockFmhaShape;
    using DataType     = std::conditional_t<TensorTag::Id == 0,
                                            typename Problem::QDataType,
                                            std::conditional_t<TensorTag::Id == 1,
                                                               typename Problem::KDataType,
                                                               typename Problem::VDataType>>;
    using Padding      = typename TensorTag::PaddingConfig;
    constexpr auto d   = make_qr_tdm_writer_distribution<TensorTag, Problem, LoadOnce>();
    using Distribution = remove_cvref_t<decltype(d)>;

    constexpr index_t Rows        = TensorTag::Id == 0 ? Shape::kM0 : Shape::kN0;
    constexpr index_t Cols        = TensorTag::Id == 0 ? Shape::kSubQKHeaddim
                                    : TensorTag::Id == 1 ? (LoadOnce ? Shape::kSubQKHeaddim : Shape::kK0)
                                                         : Shape::kN1;
    constexpr index_t NumWaves    = Problem::kBlockSize / get_warp_size();
    constexpr index_t RowsPerWave = Rows / NumWaves;
    constexpr auto raw_box_dim    = to_sequence(d.get_ys_to_d_descriptor().get_lengths()).reverse();
    constexpr auto lds_desc =
        make_qr_tdm_row_major_lds_descriptor<DataType, Rows, Cols, Padding, kQrTdmLdsAccessBytes>();

    static_assert(numeric_traits<DataType>::PackedSize == 1);
    static_assert(Distribution::NDimP == 1);
    static_assert(raw_box_dim.size() == 2);
    static_assert(Rows % NumWaves == 0);

    bool valid = Problem::kBlockSize == 128 && Shape::kQKHeaddim == 128 &&
                 Shape::kSubQKHeaddim == 128 && NumWaves == 4 && raw_box_dim[number<0>{}] == Cols &&
                 raw_box_dim[number<1>{}] == RowsPerWave &&
                 raw_box_dim[number<0>{}] * raw_box_dim[number<1>{}] * sizeof(DataType) ==
                     RowsPerWave * Cols * sizeof(DataType) &&
                 RowsPerWave * NumWaves == Rows;

    for(index_t wave = 0; wave < NumWaves; ++wave)
    {
        const auto adaptor_coord = make_tensor_adaptor_coordinate(
            d.get_ps_ys_to_xs_adaptor(),
            container_concat(array<index_t, 1>{wave}, array<index_t, Distribution::NDimY>{0}));
        const auto origin = adaptor_coord.get_bottom_index();
        const index_t logical_byte_origin =
            (origin[number<0>{}] * Cols + origin[number<1>{}]) * sizeof(DataType);
        const index_t physical_byte_origin =
            Padding::kEnabled
                ? logical_byte_origin +
                      (logical_byte_origin / Padding::kIntervalBytes) * Padding::kPadBytes
                : logical_byte_origin;

        valid = valid && origin[number<0>{}] == wave * RowsPerWave && origin[number<1>{}] == 0 &&
                (!Padding::kEnabled || logical_byte_origin % Padding::kIntervalBytes == 0) &&
                physical_byte_origin % kQrTdmLdsAccessBytes == 0 &&
                lds_desc.calculate_offset(origin) * sizeof(DataType) == physical_byte_origin;
    }

    if constexpr(TensorTag::Id == 1 && LoadOnce)
    {
        constexpr index_t k0_loops = Shape::kQKHeaddim / Shape::kK0;
        valid                      = valid && Shape::kQKHeaddim == Shape::kSubQKHeaddim &&
                Shape::kSubQKHeaddim % Shape::kK0 == 0 &&
                k0_loops * Shape::kK0 == Shape::kQKHeaddim && Cols == Shape::kSubQKHeaddim &&
                lds_desc.get_length(number<1>{}) == Shape::kSubQKHeaddim && Shape::kK0 == 32;
    }

    return valid;
}

template <typename TensorTag, typename Problem>
CK_TILE_HOST_DEVICE constexpr bool validate_qr_tdm_reader_segments()
{
    using Shape        = typename Problem::BlockFmhaShape;
    using DataType     = std::conditional_t<TensorTag::Id == 0,
                                            typename Problem::QDataType,
                                            std::conditional_t<TensorTag::Id == 1,
                                                               typename Problem::KDataType,
                                                               typename Problem::VDataType>>;
    using Padding      = typename TensorTag::PaddingConfig;
    constexpr auto d   = make_qr_tdm_reader_distribution<TensorTag, Problem>();
    using Distribution = remove_cvref_t<decltype(d)>;

    constexpr bool IsPrefill         = Shape::kM0 > 64;
    constexpr index_t Rows           = TensorTag::Id == 0 ? Shape::kM0 : Shape::kN0;
    constexpr index_t Cols           = TensorTag::Id == 0 ? Shape::kSubQKHeaddim
                                       : TensorTag::Id == 1 ? (IsPrefill ? Shape::kSubQKHeaddim : Shape::kK0)
                                                            : Shape::kN1;
    constexpr index_t WindowRows     = TensorTag::Id == 2 ? Shape::kK1 : Rows;
    constexpr index_t WindowCols     = TensorTag::Id == 1 ? Shape::kK0 : Cols;
    constexpr index_t RowWindows     = TensorTag::Id == 2 ? Rows / WindowRows : 1;
    constexpr index_t ColWindows     = TensorTag::Id == 1 && IsPrefill ? Cols / WindowCols : 1;
    constexpr index_t VectorElements = kQrTdmLdsAccessBytes / sizeof(DataType);

    static_assert(numeric_traits<DataType>::PackedSize == 1);
    static_assert(Distribution::NDimP == 2);

    constexpr auto lds_desc =
        make_qr_tdm_row_major_lds_descriptor<DataType, Rows, Cols, Padding, kQrTdmLdsAccessBytes>();
    using LdsView = decltype(make_tensor_view<address_space_enum::lds>(
        static_cast<DataType*>(nullptr), lds_desc));
    using ReaderWindow =
        decltype(make_tile_window(std::declval<LdsView>(),
                                  make_tuple(number<WindowRows>{}, number<WindowCols>{}),
                                  array<index_t, 2>{0, 0},
                                  d));
    using ReaderTraits = typename ReaderWindow::Traits;

    constexpr auto safe_vectors = ReaderWindow::get_window_adaptor_ys_safe_vector_length_strides();
    constexpr auto safe_lengths = safe_vectors[number<0>{}];
    constexpr auto safe_strides = safe_vectors[number<1>{}];
    constexpr index_t VectorDim = ReaderTraits::VectorDimY;

    bool valid = ReaderTraits::ScalarPerVector == VectorElements &&
                 ReaderTraits::ScalarPerVector * sizeof(DataType) == kQrTdmLdsAccessBytes &&
                 safe_lengths[VectorDim] >= VectorElements && safe_strides[VectorDim] == 1 &&
                 lds_desc.get_length(number<0>{}) == Rows &&
                 lds_desc.get_length(number<1>{}) == Cols &&
                 (TensorTag::Id != 1 || WindowCols == Shape::kK0) &&
                 (!Padding::kEnabled || Padding::kIntervalBytes % kQrTdmLdsAccessBytes == 0) &&
                 (!Padding::kEnabled || Padding::kPadBytes % kQrTdmLdsAccessBytes == 0);

    for(index_t row_window = 0; row_window < RowWindows; ++row_window)
    {
        for(index_t col_window = 0; col_window < ColWindows; ++col_window)
        {
            const index_t row          = row_window * WindowRows;
            const index_t col          = col_window * WindowCols;
            const index_t logical_byte = (row * Cols + col) * sizeof(DataType);
            const index_t physical_byte =
                lds_desc.calculate_offset(make_tuple(row, col)) * sizeof(DataType);

            valid = valid && row < Rows && col < Cols && logical_byte % kQrTdmLdsAccessBytes == 0 &&
                    physical_byte % kQrTdmLdsAccessBytes == 0 &&
                    (!Padding::kEnabled ||
                     logical_byte % Padding::kIntervalBytes + kQrTdmLdsAccessBytes <=
                         Padding::kIntervalBytes);
        }
    }

    if constexpr(TensorTag::kTranspose)
    {
        valid = valid &&
                TransposeTileDistrChecker<Distribution, DataType, DefaultTranspose<DataType>>::
                    distr_encoding_valid &&
                DefaultTranspose<DataType>::SubtileMinorDimension == VectorElements;
    }

    return valid;
}

} // namespace detail

} // namespace ck_tile
