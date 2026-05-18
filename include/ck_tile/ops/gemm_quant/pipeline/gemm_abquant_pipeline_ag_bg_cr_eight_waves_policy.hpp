// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp"

namespace ck_tile {
namespace detail {

template <typename Problem>
struct GemmABQuantPipelineAgBgCrAsyncPolicy
{
    static constexpr auto I0             = number<0>{};
    static constexpr auto I1             = number<1>{};
    static constexpr auto I2             = number<2>{};
    static constexpr auto WGAccessDouble = WGAttrNumAccessEnum::Double;

    using ALayout          = remove_cvref_t<typename Problem::ALayout>;
    using BLayout          = remove_cvref_t<typename Problem::BLayout>;
    using ADataType        = remove_cvref_t<typename Problem::ADataType>;
    using BDataType        = remove_cvref_t<typename Problem::BDataType>;
    using CDataType        = remove_cvref_t<typename Problem::CDataType>;
    using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
    using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
    static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>, "Wrong!");
    static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>, "Wrong!");
    static_assert(std::is_same_v<AComputeDataType, fp8_t> ||
                  std::is_same_v<AComputeDataType, bf8_t>);
    static_assert(std::is_same_v<BComputeDataType, fp8_t> ||
                  std::is_same_v<BComputeDataType, bf8_t>);
    static_assert(std::is_same_v<CDataType, float>);

    using BlockGemmShape = typename Problem::BlockGemmShape;
    using BlockWarps     = typename BlockGemmShape::BlockWarps;
    using WarpTile       = typename BlockGemmShape::WarpTile;

    static constexpr index_t BlockSize  = Problem::kBlockSize;
    static constexpr index_t MPerBlock  = BlockGemmShape::kM;
    static constexpr index_t NPerBlock  = BlockGemmShape::kN;
    static constexpr index_t KPerBlock  = BlockGemmShape::kK;
    static constexpr index_t WarpTileM  = WarpTile::at(I0);
    static constexpr index_t WarpTileN  = WarpTile::at(I1);
    static constexpr index_t WarpTileK  = WarpTile::at(I2);
    static constexpr index_t MWarpTiles = MPerBlock / WarpTileM;
    static constexpr index_t NWarpTiles = NPerBlock / WarpTileN;
    static constexpr index_t KWarpTiles = KPerBlock / WarpTileK;

    using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
    using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

    static constexpr index_t KPerBlockAQ = KPerBlock / AQuantGroupSize::kK;
    static constexpr index_t KPerBlockBQ = KPerBlock / BQuantGroupSize::kK;

    static constexpr index_t MWarps       = BlockWarps::at(I0);
    static constexpr index_t NWarps       = BlockWarps::at(I1);
    static constexpr index_t KWarps       = BlockWarps::at(I2);
    static constexpr index_t MIterPerWarp = MWarpTiles / MWarps;
    static constexpr index_t NIterPerWarp = NWarpTiles / NWarps;
    static constexpr index_t KPerWarp     = KPerBlock / KWarps;
    static constexpr index_t NPerWarp     = NPerBlock / NWarps;
    static_assert(NWarps == 2, "KWarps == 2 for ping-pong!");
    static_assert(KWarpTiles == KWarps, "Wrong!");

    static constexpr index_t KPerWarpAQ  = KPerWarp / Problem::AQuantGroupSize::kK;
    static constexpr index_t NPerWarpBQ  = NPerWarp / Problem::BQuantGroupSize::kN;
    static constexpr index_t KPerWarpkBQ = KPerWarp / Problem::BQuantGroupSize::kK;
    static_assert(Problem::AQuantGroupSize::kM == 1 && Problem::AQuantGroupSize::kK == WarpTileK);

    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t warp_num  = BlockSize / warp_size;
    static_assert(warp_size == 64, "Wrong!");
    static_assert(warp_num * warp_size == BlockSize, "Wrong!");

    static_assert(sizeof(ADataType) == sizeof(BDataType), "Wrong!");
    static constexpr index_t ElementSize = sizeof(ADataType);
    static constexpr index_t K2          = Problem::VectorLoadSize / ElementSize; // 16
    static constexpr index_t K1          = WarpTile::at(I2) / K2;                 // 8
    static constexpr index_t K0          = KPerWarp / (K1 * K2);
    static_assert(K0 * K1 * K2 == KPerWarp, "Wrong!");
    static_assert(K0 == 1, "Wrong!");

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto GetKStepAQ() { return KPerBlockAQ; }
    CK_TILE_HOST_DEVICE static constexpr auto GetKStepBQ() { return KPerBlockBQ; }

    CK_TILE_HOST_DEVICE static constexpr auto MakeAQBlockDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<                          //
                sequence<NWarps, warp_size / WarpTileM>,         // ?, 4
                tuple<sequence<MIterPerWarp, MWarps, WarpTileM>, // ?,?,16
                      sequence<KWarps, KPerWarpAQ>>,             // 1, 1
                tuple<sequence<2, 0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 0, 1>, sequence<1, 2>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQBlockDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<                                             //
                sequence<MWarps, warp_size>,                                        // 4,64
                tuple<sequence<NWarps, NPerWarpBQ>, sequence<KWarps, KPerWarpkBQ>>, // 2,1 1,1
                tuple<sequence<2, 1, 0>, sequence<0>>,
                tuple<sequence<0, 0, 0>, sequence<1>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");
        static_assert(Problem::TransposeC, "Wrong!");

        using WarpGemm = WarpGemmDispatcher<AComputeDataType,
                                            BComputeDataType,
                                            CDataType,
                                            WarpTileM,
                                            WarpTileN,
                                            WarpTileK,
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            WGAccessDouble>;

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ADataType,
                                                                      BDataType,
                                                                      CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return ABQuantBlockUniversalGemmAsBsCrAsync<Problem, BlockGemmPolicy>{};
    }
};
} // namespace detail

struct GemmABQuantPipelineAgBgCrAsyncPolicy : public GemmPipelineAgBgCrCompAsyncEightWavesPolicy
{

#define FORWARD_METHOD_(method)                                               \
    template <typename Problem, typename... Args>                             \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)          \
    {                                                                         \
        return detail::GemmABQuantPipelineAgBgCrAsyncPolicy<Problem>::method( \
            std::forward<Args>(args)...);                                     \
    }

    FORWARD_METHOD_(GetVectorSizeAQ);
    FORWARD_METHOD_(GetVectorSizeBQ);
    FORWARD_METHOD_(MakeAQBlockDistribution);
    FORWARD_METHOD_(MakeBQBlockDistribution);
    FORWARD_METHOD_(GetBlockGemm);
    FORWARD_METHOD_(GetKStepAQ);
    FORWARD_METHOD_(GetKStepBQ);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
