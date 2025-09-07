// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace ck_tile {

struct BlockFmhaV3PipelineDefaultPolicy
{
    static constexpr ck_tile::index_t NumWarpPerGroup = 4;
    static constexpr ck_tile::index_t NumThreadPerWarpGroup =
        NumWarpPerGroup * ck_tile::get_warp_size();

    // TODO: GetAlignment*() currently didn't consider if need padding or not
    //       so in pipeline still need check padding requirement
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentK()
    {
        using namespace ck_tile;
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
#if defined(__gfx950__)
        constexpr index_t MaxReadSizeInBytes = 16;
#else
        constexpr index_t MaxReadSizeInBytes = 4;
#endif
        return MaxReadSizeInBytes / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentV()
    {
        using namespace ck_tile;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
#if defined(__gfx950__)
        constexpr index_t MaxReadSizeInBytes = 16;
#else
        constexpr index_t MaxReadSizeInBytes = 4;
#endif
        return MaxReadSizeInBytes / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        using namespace ck_tile;

        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemVPackK()
    {
        using namespace ck_tile;

        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using namespace ck_tile;

        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::VDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t NPerThread     = kMaxVecLoad;
        constexpr index_t NThreads       = kNPerBlock / NPerThread;
        constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t KPerThread     = kKPerBlock / (KThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakePRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;

        return make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using namespace ck_tile;

        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = ck_tile::detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        // compute the endcoding before transpose
        constexpr auto v_block_dstr =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(v_block_dstr_encode),
                                          typename Problem::VDataType>::TransposedDstrEncode{});

        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using namespace ck_tile;

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

        constexpr auto warp_gemm = []() {
            if constexpr(std::is_same_v<typename Problem::QDataType, half_t> &&
                         std::is_same_v<typename Problem::KDataType, half_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                /// NOTICE: in order to use load_tile_transpose() later for V tile, we cannot use
                /// WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution here
                return WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution<>{};
            }
            else if constexpr(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                              std::is_same_v<typename Problem::KDataType, bf16_t> &&
                              std::is_same_v<typename Problem::SaccDataType, float>)
            {
                /// NOTICE: in order to use load_tile_transpose() later for V tile, we cannot use
                /// WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution here
                return WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                decltype(warp_gemm),
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPVBlockGemm()
    {
        using namespace ck_tile;

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
        /// NOTICE: in order to use load_tile_transpose() later for V tiles, we have to pass
        /// WGAttrNumAccessEnum::Double instead of WGAttrNumAccessEnum::Single
        using WarpGemm = WarpGemmDispatcher<typename Problem::PDataType,
                                            typename Problem::VDataType,
                                            typename Problem::OaccDataType,
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                            true,
                                            false,
                                            false,
                                            WGAttrNumAccessEnum::Double>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;
        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    static constexpr ck_tile::index_t kKLdsPadInBytes = 4 * 4;  // 4 dwords
    static constexpr ck_tile::index_t kVLdsPadInBytes = 4 * 16; // 16 dwords

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr auto k_lds_block_desc =
            make_naive_tensor_descriptor(make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                         make_tuple(number<kKPerBlock>{}, number<1>{}),
                                         number<8>{},
                                         number<1>{});

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr auto v_lds_block_desc =
            make_naive_tensor_descriptor(make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
                                         make_tuple(number<kNPerBlock>{}, number<1>{}),
                                         number<8>{},
                                         number<1>{});

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
            constexpr index_t WarpSize   = ck_tile::get_warp_size();

            constexpr index_t KPack   = GetSmemKPackK<Problem>(); // this is for lds
            constexpr index_t KVector = GetAlignmentK<Problem>(); // this is for global load
            constexpr index_t kPad    = KPack;

            static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector;
            constexpr index_t LaneGroups = WarpSize / LanesPerK;
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

            return NumIssues * NumWarps * (WarpSize * KVector + kPad);
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType                = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks        = 32; // TODO: need change based on arch
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
            constexpr index_t kKPack       = GetSmemKPackK<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return max(SingleKSize, SingleVSize);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        using namespace ck_tile;

        constexpr index_t k_element_space_size =
            MakeKLdsBlockDescriptor<Problem>().get_element_space_size();

        constexpr index_t v_element_space_size =
            MakeVLdsBlockDescriptor<Problem>().get_element_space_size();

        static_assert(ck_tile::max(k_element_space_size, v_element_space_size) <=
                      GetSingleSmemElementSpaceSize<Problem>());

        /// TODO: override GetSingleSmemElementSpaceSize() to align with MakeKLdsBlockDescriptor() &
        /// MakeVLdsBlockDescriptor()
        static_assert(std::is_same_v<typename Problem::KDataType, typename Problem::VDataType>);
        constexpr index_t kv_element_space_size_in_bytes =
            GetSingleSmemElementSpaceSize<Problem>() * sizeof(typename Problem::KDataType);

        return kv_element_space_size_in_bytes;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return 4 * GetSmemSizeKV<Problem>();
    }
};

} // namespace ck_tile
