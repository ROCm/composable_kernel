// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"

namespace ck {

template <typename EDataType,
          typename HDataType,
          typename MeanDataType,
          typename VarDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename EHGridDesc_M_N,
          typename MeanVarCountGridDesc_M_N,
          typename GammaBetaGridDesc_N,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t NThreadClusterSize,
          index_t MThreadSliceSize,
          index_t NThreadSliceSize,
          index_t ESrcYDstVectorDim,
          index_t ESrcVectorSize,
          index_t YDstVectorSize,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct GridwiseWelfordSecondHalfLayernorm2d
{
    static constexpr bool reorder_thread_cluster = (ESrcYDstVectorDim == 0);

    using ThreadClusterLengths_M_N = Sequence<MThreadClusterSize, NThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_N{}, ThreadClusterArrangeOrder{});

    using ThreadReduceSrcDesc_M_1 = decltype(
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}, Number<1>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelfordMerge<ComputeDataType, ThreadReduceSrcDesc_M_1, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<ComputeDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_N,
                                              ThreadClusterArrangeOrder>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t N_BlockTileSize = NThreadClusterSize * NThreadSliceSize;

    __device__ static void Run(const EDataType* __restrict__ p_e_grid,
                               const MeanDataType* __restrict__ p_in_welford_mean_grid,
                               const VarDataType* __restrict__ p_in_welford_var_grid,
                               const int32_t* __restrict__ p_in_welford_count_grid,
                               const GammaDataType* __restrict__ p_gamma_grid,
                               const BetaDataType* __restrict__ p_beta_grid,
                               HDataType* __restrict__ p_h_grid,
                               const EHGridDesc_M_N& e_grid_desc_m_n,
                               const EHGridDesc_M_N& h_grid_desc_m_n,
                               const MeanVarCountGridDesc_M_N& mean_var_count_grid_desc_m_n,
                               const GammaBetaGridDesc_N& gamma_grid_desc_n,
                               const GammaBetaGridDesc_N& beta_grid_desc_n,
                               index_t gemm_nblock_,
                               index_t num_mean_var_count_k_block_tile_iteration,
                               index_t num_xy_k_block_tile_iteration,
                               ComputeDataType epsilon)
    {
        ignore = p_e_grid;
        ignore = p_in_welford_mean_grid;
        ignore = p_in_welford_var_grid;
        ignore = p_in_welford_count_grid;
        ignore = p_gamma_grid;
        ignore = p_beta_grid;
        ignore = p_h_grid;
        ignore = e_grid_desc_m_n;
        ignore = h_grid_desc_m_n;
        ignore = mean_var_count_grid_desc_m_n;
        ignore = gamma_grid_desc_n;
        ignore = beta_grid_desc_n;
        ignore = gemm_nblock_;
        ignore = num_mean_var_count_k_block_tile_iteration;
        ignore = num_xy_k_block_tile_iteration;
        ignore = epsilon;

    } // run
};

} // namespace ck
