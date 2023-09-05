// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/grid/grid_gemm.hpp"
#include "ck/tile_program/grid/grid_gemm_policy.hpp"
#include "ck/tile_program/grid/grid_gemm_problem.hpp"

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementFunction,
          typename BElementFunction,
          typename CElementFunction,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock>
struct Gemm
{
    static_assert(std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor> &&
                  std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor> &&
                  std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>);

    using Problem = ck::tile_program::grid::GridGemmProblem<ADataType,
                                                            BDataType,
                                                            AccDataType,
                                                            CDataType,
                                                            AElementFunction,
                                                            BElementFunction,
                                                            CElementFunction>;

    using Policy = ck::tile_program::grid::GridGemmPolicy<
        kBlockSize,
        kMPerBlock,
        kNPerBlock,
        kKPerBlock,
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2,
        ck::Tuple<ck::tile_program::grid::DefaultBlock2TileMap,
                  ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>>;

    using GridGemm = ck::tile_program::grid::GridGemm<Problem, Policy>;

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const ADataType* p_a,
                                        const BDataType* p_b,
                                        CDataType* p_c,
                                        ck::index_t M,
                                        ck::index_t N,
                                        ck::index_t K,
                                        ck::index_t Lda,
                                        ck::index_t Ldb,
                                        ck::index_t Ldc,
                                        const AElementFunction& a_element_func,
                                        const BElementFunction& b_element_func,
                                        const CElementFunction& c_element_func) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // FIXME: assume RCR layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<32>{}, Number<1>{});

        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<32>{}, Number<1>{});

        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        GridGemm{}(ps,
                   a_dram_grid,
                   b_dram_grid,
                   c_dram_grid,
                   a_element_func,
                   b_element_func,
                   c_element_func);
    }
};
