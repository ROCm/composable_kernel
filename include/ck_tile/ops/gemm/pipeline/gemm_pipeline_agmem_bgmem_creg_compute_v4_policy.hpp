// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct GemmPipelineAGmemBGmemCregComputeV4DefaultPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr auto ATileAccessPattern = tile_distribution_pattern::thread_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::thread_raked;

    /**
     * @brief Get the maximum global memory vector load size.
     *
     * @tparam Problem      The UniversalGemmPipelineProblem object.
     * @tparam DataType     The tensor data type we're considering.
     * @tparam MNPerBlock   The MPerBlock or NPerBlock value depending on tensor (A/B).
     * @tparam XPerTile     The contiguous Tile dimension size.
     * @return Maximum DRAM vector load size.
     */
    template <typename Problem, typename DataType, index_t MNPerBlock, index_t XPerTile>
    CK_TILE_HOST_DEVICE static constexpr auto GetGlobalVectorLoadSize()
    {
        constexpr index_t BlockSize           = Problem::kBlockSize;
        constexpr index_t KPerBlock           = Problem::BlockGemmShape::kK;
        constexpr index_t elements_per_thread = MNPerBlock * KPerBlock / BlockSize;

        // Assume DataType is even!
        if constexpr(XPerTile % (16 / sizeof(DataType)) == 0 &&
                     elements_per_thread % (16 / sizeof(DataType)) == 0)
        {
            return (16 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (8 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (8 / sizeof(DataType)) == 0)
        {
            return (8 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= 4 && XPerTile % (4 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (4 / sizeof(DataType)) == 0)
        {
            return (4 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= 2 && XPerTile % (2 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (2 / sizeof(DataType)) == 0)
        {
            return (2 / sizeof(DataType));
        }
        else
        {
            return 1;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeA()
    {
        using ALayout               = remove_cvref_t<typename Problem::ALayout>;
        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, KPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, MPerBlock>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeB()
    {
        using BLayout               = remove_cvref_t<typename Problem::BLayout>;
        using BDataType             = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, NPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, KPerBlock>();
        }
    }

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @tparam Problem - Gemm pipeline problem class.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using BlockGemm = remove_cvref_t<decltype(GetBlockGemm<Problem>())>;
        using WG        = typename BlockGemm::WarpGemm;

        constexpr bool TransposeC = Problem::TransposeC;
        using CLayout             = typename Problem::CLayout;
        using CWarpDstr           = typename WG::CWarpDstr;

        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has just a single item in Mdim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        using BlockGemm         = decltype(GetBlockGemm<Problem>());
        constexpr index_t KPack = BlockGemm::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackB()
    {
        using BlockGemm         = decltype(GetBlockGemm<Problem>());
        constexpr index_t KPack = BlockGemm::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        // TODO: this 8 is AK1! should be a policy parameter!
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kMPerBlock>{}, number<8>{}),
            make_tuple(number<kMPerBlock * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                       make_merge_transform(make_tuple(number<kKPerBlock>{} / 8, number<8>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / 8>{}, number<kNPerBlock>{}, number<8>{}),
            make_tuple(number<(kNPerBlock)*8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                       make_merge_transform(make_tuple(number<kKPerBlock / 8>{}, number<8>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a =
            integer_least_multiple(sizeof(typename Problem::ADataType) *
                                       MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                                   16);
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t smem_size_b =
            integer_least_multiple(sizeof(typename Problem::BDataType) *
                                       MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
                                   16);
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();

        return smem_size_a + smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();

        // Tile: MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                          MPerBlock,
                                                                          KPerBlock,
                                                                          VecLoadSize,
                                                                          ATileAccessPattern>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
        // Tile: KPerBlock X MPerBlock
        else
        {
            using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                          KPerBlock,
                                                                          MPerBlock,
                                                                          VecLoadSize,
                                                                          ATileAccessPattern>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();

        // Tile: KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                          KPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
        // Tile: NPerBlock X KPerBlock
        else
        {
            using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                          NPerBlock,
                                                                          KPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegTileDistribution()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      KPerBlock,
                                                                      MPerBlock,
                                                                      VecLoadSize,
                                                                      ATileAccessPattern>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBRegTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();

        using TileEncodingPattern = TileDistributionEncodingPattern2D<BlockSize,
                                                                      KPerBlock,
                                                                      NPerBlock,
                                                                      VecLoadSize,
                                                                      BTileAccessPattern>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Problem::TransposeC;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using AccDataType     = float;
        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                AccDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
