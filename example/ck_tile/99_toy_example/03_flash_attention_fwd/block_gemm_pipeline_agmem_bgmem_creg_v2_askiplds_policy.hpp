// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "blockgemm_pipeline_agmem_bgmem_creg_policy_impl.hpp"
#include "../../../example/ck_tile/99_toy_example/02_gemm/block_gemm_pipeline_agmem_bgmem_creg.hpp"

namespace ck_tile {

// NOTE: Assume A is K-Major
struct BlockGemmPipelineAGmemBGmemCRegSkipALdsPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_a_reg_block_descriptor<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        return policy_impl::make_b_lds_block_descriptor_3d_pad<Problem>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_a_dram_tile_distribution_skip_lds<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        return policy_impl::make_b_dram_tile_distribution<Problem>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        using BlockGemmPolicy = BlockGemmARegBSmemCRegV1K8Policy;

        return BlockGemmARegBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};

template <index_t AKDim_>
struct BlockGemmPipelineAGmemBGmemCRegSkipALdsPersistentQRegCachePolicy
    : BlockGemmPipelineAGmemBGmemCRegSkipALdsPolicy
{
    static constexpr index_t AKDim = AKDim_;

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = remove_cvref_t<decltype(blockgemm)>;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = AKDim;

        constexpr auto config =
            BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<NWarp>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
            tuple<sequence<1, 0>>,
            tuple<sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

        return a_block_dstr;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        return MakeARegBlockDescriptor<Problem>();
    }
};

} // namespace ck_tile
