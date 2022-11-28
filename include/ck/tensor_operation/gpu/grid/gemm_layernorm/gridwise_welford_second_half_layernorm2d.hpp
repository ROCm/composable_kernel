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
          typename ComputeDataType,
          typename XYGridDesc_M_N,
          typename MeanVarGridDesc_M_N,
          typename GammaBetaGridDesc_N,
          typename MeanVarGridDesc_M,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t NThreadClusterSize,
          index_t MThreadSliceSize,
          index_t NThreadSliceSize,
          index_t XSrcYDstVectorDim,
          index_t XSrcVectorSize,
          index_t YDstVectorSize,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct GridwiseWelfordSecondHalfLayernorm2d
{
    static constexpr bool reorder_thread_cluster = (XSrcYDstVectorDim == 0);

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
                               const MeanDataType* __restrict__ p_mean_grid,
                               const VarDataType* __restrict__ p_var_grid,
                               HDataType* __restrict__ p_h_grid,
                               /*const MeanVarGridDesc_M_N& mean_grid_desc_m_k,
                               const MeanVarGridDesc_M_N& var_grid_desc_m_k,
                               const GammaBetaGridDesc_N& gamma_grid_desc_m,
                               const GammaBetaGridDesc_N& beta_grid_desc_m,
                               const MeanVarGridDesc_M& mean_var_grid_desc_m,*/
                               index_t blkgroup_size)
    {
        ignore = p_e_grid;
        ignore = p_mean_grid;
        ignore = p_var_grid;
        ignore = p_h_grid;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / blkgroup_size;
        const index_t block_local_id  = block_global_id % blkgroup_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_n_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_N        = Sequence<MThreadSliceSize, NThreadSliceSize>;
        using ThreadBufferLengths_M          = Sequence<MThreadSliceSize>;
        using ThreadBufferLengths_M_1        = Sequence<MThreadSliceSize, 1>;
        constexpr auto thread_buffer_desc_m_ = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<NThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_m_1 = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));
/*
        auto threadwise_mean_load_m_n =
            ThreadwiseTensorSliceTransfer_v2<MeanDataType,
                                             ComputeDataType,
                                             MeanVarGridDesc_M_N,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             Sequence<0, 1>,
                                             1,
                                             1,
                                             1,
                                             true>(
                mean_grid_desc_m_n,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id * 1));*/

    } // run
};

} // namespace ck
