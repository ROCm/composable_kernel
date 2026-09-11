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
        return static_cast<index_t>(16 / sizeof(QDataType));
    }

    // Plain row-major Q LDS desc. TDM box-major write cannot produce an XOR'd
    // layout, so the Xor template param on the original generic descriptor
    // was unreachable from this pipeline and has been removed.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        return make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                                            make_tuple(number<kKPerBlock>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    // Plain row-major K LDS desc; same no-swizzle rationale as Q above.
    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;

        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        return make_naive_tensor_descriptor(make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                            make_tuple(number<kKPerBlock>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t kKPack = GetSmemKPackV<Problem>();

        constexpr auto v_lds_block_desc = [&]() {
            if constexpr(Xor)
            {
                constexpr auto XorGroupSize =
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{});

#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                constexpr auto LDSLayerSize  = 256 / sizeof(typename Problem::VDataType);
                constexpr auto XorLengthFold = LDSLayerSize / kNPerBlock;

                if constexpr(XorLengthFold > 1)
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock / XorLengthFold>{},
                                   number<LDSLayerSize / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<LDSLayerSize>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(
                            make_xor_transform(make_tuple(number<kKPerBlock / XorLengthFold>{},
                                                          number<LDSLayerSize / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    constexpr auto v_lds_block_desc_tmp = transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock / XorLengthFold>{}),
                            make_unmerge_transform(make_tuple(number<XorLengthFold>{},
                                                              number<kNPerBlock / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_tmp,
                        make_tuple(
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kKPerBlock / XorLengthFold>{}, number<XorLengthFold>{})),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock>{},
                                   number<kNPerBlock / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<kNPerBlock>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(make_xor_transform(make_tuple(
                                       number<kKPerBlock>{}, number<kNPerBlock / XorGroupSize>{})),
                                   make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock>{}),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
                    make_tuple(number<kNPerBlock>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});
            }
        }();

        return v_lds_block_desc;
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

    // -------------------------------------------------------------------------
    // TDM LDS padding config
    //
    // Mirrors the formula in
    //   gemm_universal_pipeline_ag_bg_cr_policy.hpp:1131 GetLdsPaddingConfig
    // adapted to fmha shape names. Returns a tuple (IsPadding, PadAmount,
    // PadInterval) suitable for filling TDMConfig::pad_config (see
    // amd_tdm_descriptor.hpp). Field semantics:
    //   * pad_amount:  N -> add (N+1) dwords padding
    //   * pad_interval N -> insert padding every 2^(N+1) dwords
    //
    // Branch selection rationale (do NOT silently change without re-verifying):
    //   * Q : non-tr-load   (output via plain ds_load, like K)  -> "else" branch
    //   * K : non-tr-load   (output via plain ds_load)          -> "else" branch
    //   * V : tr-load       (output via ds_load_tr_b128)        -> "if"   branch
    // -------------------------------------------------------------------------

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        // Q LDS padding DISABLED. Enabling the writer-side padding alone
        // misaligns the plain row-major Q LDS reader; padding is a
        // bank-conflict perf optimization, not a correctness requirement,
        // and is deferred to a follow-up perf pass (same as K/V below).
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        // K LDS padding DISABLED. Same rationale as Q above; original
        // padded implementation mirrors
        // gemm_universal_pipeline_ag_bg_cr_policy.hpp:1131.
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        // V LDS padding currently DISABLED. Re-enabling the writer-side config
        // alone misaligns the reader (V LDS read view is plain row-major and
        // not padding-aware). A full re-enable requires a padding-aware
        // MakeVLdsBlockDescriptor mirroring gemm pipeline's
        // MakeBLdsBlockDescriptorForTrLoad
        // (gemm_universal_pipeline_ag_bg_cr_policy.hpp:1297-1410). Padding is
        // a bank-conflict-avoidance perf optimization, not a correctness
        // requirement; deferred to a follow-up perf pass.
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }
};

} // namespace ck_tile
