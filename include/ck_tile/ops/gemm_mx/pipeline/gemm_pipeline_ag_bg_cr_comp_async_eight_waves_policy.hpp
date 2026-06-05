// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp"
#include "ck_tile/ops/gemm_mx/block/block_mx_gemm_areg_breg_creg_eight_waves_v1.hpp"

namespace ck_tile {
namespace detail {

template <typename Problem>
struct MXGemmPipelineAgBgCrCompAsyncEightWavesPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    // MX scaling configuration: each e8m0 scale covers 32 elements in K
    static constexpr int BlockScaleSize = 32;

    using ALayout          = remove_cvref_t<typename Problem::ALayout>;
    using BLayout          = remove_cvref_t<typename Problem::BLayout>;
    using ADataType        = remove_cvref_t<typename Problem::ADataType>;
    using BDataType        = remove_cvref_t<typename Problem::BDataType>;
    using CDataType        = remove_cvref_t<typename Problem::CDataType>;
    using AComputeDataType = remove_cvref_t<typename Problem::AComputeDataType>;
    using BComputeDataType = remove_cvref_t<typename Problem::BComputeDataType>;
    using ComputeDataType  = AComputeDataType;
    static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>, "Wrong!");
    static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>, "Wrong!");
    static_assert(is_any_of<AComputeDataType, fp8_t, bf8_t, pk_fp4_t, pk_fp6x16_t>::value);
    static_assert(is_any_of<BComputeDataType, fp8_t, bf8_t, pk_fp4_t, pk_fp6x16_t>::value);
    static_assert(std::is_same_v<AComputeDataType, BComputeDataType>);
    static_assert(std::is_same_v<CDataType, float>);

    using BlockGemmShape = typename Problem::BlockGemmShape;
    using BlockWarps     = typename BlockGemmShape::BlockWarps;
    using WarpTile       = typename BlockGemmShape::WarpTile;

    static constexpr index_t BlockSize  = Problem::kBlockSize;
    static constexpr index_t MPerBlock  = BlockGemmShape::kM;
    static constexpr index_t NPerBlock  = BlockGemmShape::kN;
    static constexpr index_t KPerBlock  = BlockGemmShape::kK;
    static constexpr index_t MWarps     = BlockGemmShape::BlockWarps::at(I0);
    static constexpr index_t NWarps     = BlockGemmShape::BlockWarps::at(I1);
    static constexpr index_t KWarps     = BlockGemmShape::BlockWarps::at(I2);
    static constexpr index_t WarpTileM  = WarpTile::at(I0);
    static constexpr index_t WarpTileN  = WarpTile::at(I1);
    static constexpr index_t WarpTileK  = WarpTile::at(I2);
    static constexpr index_t MWarpTiles = MPerBlock / WarpTileM;
    static constexpr index_t NWarpTiles = NPerBlock / WarpTileN;
    static constexpr index_t KWarpTiles = KPerBlock / WarpTileK;

    // XdlPack: how many e8m0_t scale values are packed into one int32_t per dimension
    // Host packs MXdlPack * KXdlPack e8m0_t into one int32_t for A scales
    // Host packs NXdlPack * KXdlPack e8m0_t into one int32_t for B scales
    static constexpr int MXdlPack = 2;
    static constexpr int NXdlPack = 2;
    static constexpr int KXdlPack = 2;

    // Compute effective XdlPack sizes (fall back to 1 when iter count < pack)
    static constexpr index_t MPerXdl      = WarpTile::at(I0);
    static constexpr index_t NPerXdl      = WarpTile::at(I1);
    static constexpr index_t KPerXdl      = WarpTile::at(I2);
    static constexpr index_t MIterPerWarp = MPerBlock / (MWarps * MPerXdl);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarps * NPerXdl);
    static constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;

    static constexpr index_t MXdlPackEff =
        (MIterPerWarp >= MXdlPack && MIterPerWarp % MXdlPack == 0) ? MXdlPack : 1;
    static constexpr index_t NXdlPackEff =
        (NIterPerWarp >= NXdlPack && NIterPerWarp % NXdlPack == 0) ? NXdlPack : 1;
    static constexpr index_t KXdlPackEff =
        (KIterPerWarp >= KXdlPack && KIterPerWarp % KXdlPack == 0) ? KXdlPack : 1;

    static constexpr index_t KPerBlockScale = KPerBlock / BlockScaleSize / KXdlPackEff;

    static constexpr index_t KPerWarp = KPerBlock / KWarps;
    static constexpr index_t NPerWarp = NPerBlock / NWarps;
    static_assert(NWarps == 2, "NWarps == 2 for ping-pong!");

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

    CK_TILE_HOST_DEVICE static constexpr auto GetKStepAQ() { return KPerBlockScale; }
    CK_TILE_HOST_DEVICE static constexpr auto GetKStepBQ() { return KPerBlockScale; }

    CK_TILE_HOST_DEVICE static constexpr auto GetInstCountAQ()
    {
        return (MIterPerWarp / MXdlPackEff) * (KIterPerWarp / KXdlPackEff);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetInstCountBQ()
    {
        return (NIterPerWarp / NXdlPackEff) * (KIterPerWarp / KXdlPackEff);
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeAQBlockDistribution()
    {
        constexpr index_t K_Lane = get_warp_size() / WarpTileM;

        constexpr index_t KPerLane = WarpTileK / BlockScaleSize / K_Lane;

        constexpr index_t MIterPerWarp_packed = MIterPerWarp / MXdlPackEff;
        constexpr index_t KIterPerWarp_packed = KIterPerWarp / KXdlPackEff;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<NWarps>,                                       // repeat over MWarps
                tuple<sequence<MWarps, MIterPerWarp_packed, WarpTileM>, // M dimension (first)
                      sequence<KIterPerWarp_packed, K_Lane, KPerLane>>, // K dimension (second)
                tuple<sequence<0, 1>, sequence<2, 1>>, // <MWarps, NWarps>, <K_Lane, WarpTileM>
                tuple<sequence<0, 0>, sequence<1, 2>>,
                sequence<2, 1, 2>, // <KIterPerWarp, MIterPerWarp, KPerLane>
                sequence<0, 1, 2>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeBQBlockDistribution()
    {
        constexpr index_t K_Lane = get_warp_size() / WarpTileN;

        constexpr index_t KPerLane = WarpTileK / BlockScaleSize / K_Lane;

        constexpr index_t NIterPerWarp_packed = NIterPerWarp / NXdlPackEff;
        constexpr index_t KIterPerWarp_packed = KIterPerWarp / KXdlPackEff;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<MWarps>,                                              // repeat over MWarps
                tuple<sequence<2, NIterPerWarp_packed, NWarps / 2, WarpTileN>, // N dimension
                                                                               // (first)
                      sequence<KIterPerWarp_packed, K_Lane, KPerLane>>, // K dimension (second)
                tuple<sequence<1, 0, 1>, sequence<2, 1>>, // <MWarps, NWarps>, <K_Lane, MPerXdl>
                tuple<sequence<0, 0, 2>, sequence<1, 3>>,
                sequence<2, 1, 2>, // <KIterPerWarp, NIterPerWarp, KPerLane>
                sequence<0, 1, 2>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        constexpr auto wg_attr_num_access =
            (std::is_same_v<ADataType, fp8_t> || std::is_same_v<BDataType, fp8_t>)
                ? WGAttrNumAccessEnum::Double
                : WGAttrNumAccessEnum::Single;

        using WarpGemm = WarpGemmDispatcher<ADataType,
                                            BDataType,
                                            CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockMXGemmARegBRegCRegEightWavesV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace detail

struct MXGemmPipelineAgBgCrCompAsyncEightWavesPolicy
    : public GemmPipelineAgBgCrCompAsyncEightWavesPolicy
{

#define FORWARD_METHOD_(method)                                                        \
    template <typename Problem, typename... Args>                                      \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)                   \
    {                                                                                  \
        return detail::MXGemmPipelineAgBgCrCompAsyncEightWavesPolicy<Problem>::method( \
            std::forward<Args>(args)...);                                              \
    }

    FORWARD_METHOD_(MakeAQBlockDistribution);
    FORWARD_METHOD_(MakeBQBlockDistribution);
    FORWARD_METHOD_(GetBlockGemm);
    FORWARD_METHOD_(GetKStepAQ);
    FORWARD_METHOD_(GetKStepBQ);
    FORWARD_METHOD_(GetInstCountAQ);
    FORWARD_METHOD_(GetInstCountBQ);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
