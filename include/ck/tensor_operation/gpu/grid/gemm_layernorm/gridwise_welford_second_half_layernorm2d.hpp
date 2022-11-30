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
          index_t ESrcHDstVectorDim,
          index_t ESrcVectorSize,
          index_t HDstVectorSize,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct GridwiseWelfordSecondHalfLayernorm2d
{
    static_assert((ESrcHDstVectorDim == 0 && MThreadSliceSize % ESrcVectorSize == 0) ||
                      (ESrcHDstVectorDim == 1 && NThreadSliceSize % ESrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((ESrcHDstVectorDim == 0 && MThreadSliceSize % HDstVectorSize == 0) ||
                      (ESrcHDstVectorDim == 1 && NThreadSliceSize % HDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (ESrcHDstVectorDim == 0);

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

    static constexpr index_t M_BlockTileSize     = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t N_BlockTileSize     = NThreadClusterSize * NThreadSliceSize;
    static constexpr index_t N_BlockTileStepSize = NThreadClusterSize * ESrcVectorSize;

    static constexpr auto EThreadBufferNumber     = Number<NThreadSliceSize / ESrcVectorSize>{};
    static constexpr auto GammaThreadBufferNumber = Number<NThreadSliceSize / ESrcVectorSize>{};
    static constexpr auto BetaThreadBufferNumber  = Number<NThreadSliceSize / ESrcVectorSize>{};
    static constexpr auto HThreadBufferNumber     = Number<NThreadSliceSize / ESrcVectorSize>{};

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
                               index_t numMeanVarCountBlockTileIteration_N,
                               index_t numEBlockTileIteration_N,
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
        ignore = numMeanVarCountBlockTileIteration_N;
        ignore = numEBlockTileIteration_N;
        ignore = epsilon;

        // float mean = static_cast<float>(p_in_welford_mean_grid[0]);
        // float var  = static_cast<float>(p_in_welford_var_grid[0]);
        // int count  = p_in_welford_count_grid[0];
        // if(get_thread_global_1d_id() == 0)
        //     printf("kernel mean = %f, var = %f, count = %d\n", mean, var, count);

        float mean = static_cast<float>(p_in_welford_mean_grid[0]);
        if(get_thread_global_1d_id() == 0)
            printf("mean = %f\n", mean);

        int s = static_cast<int>(mean_var_count_grid_desc_m_n.GetElementSpaceSize());
        if(get_thread_global_1d_id() == 0)
            printf("mean_var_count_grid_desc_m_n.GetElementSpaceSize() = %d\n", s);

        // using ThreadBufferLengths_1_1 = Sequence<1, 1>;
        // constexpr auto thread_buffer_desc_1_1 =
        //     make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<1>{}));
        // constexpr auto grid_desc_1_1 =
        //     make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<1>{}));

        // const auto mean_grid = make_dynamic_buffer<AddressSpaceEnum::Global>(
        //     p_in_welford_mean_grid, mean_var_count_grid_desc_m_n.GetElementSpaceSize());
        // StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, 1, true> mean_thread;

        // float mean1 = (mean_grid.template Get<MeanDataType>(0, true));
        // if(get_thread_global_1d_id() == 0)
        //     printf("global mean = %f\n", mean1);

        // auto threadwise_mean_load_m_k =
        //     ThreadwiseTensorSliceTransfer_v2<MeanDataType,
        //                                      ComputeDataType,
        //                                      decltype(mean_var_count_grid_desc_m_n),
        //                                      decltype(thread_buffer_desc_1_1),
        //                                      ThreadBufferLengths_1_1,
        //                                      Sequence<0, 1>,
        //                                      1,
        //                                      1,
        //                                      1,
        //                                      true>(mean_var_count_grid_desc_m_n,
        //                                            make_multi_index(0, 0));

        // threadwise_mean_load_m_k.Run(mean_var_count_grid_desc_m_n,
        //                              mean_grid,
        //                              thread_buffer_desc_1_1,
        //                              make_tuple(I0, I0),
        //                              mean_thread);

        // if(get_thread_global_1d_id() == 0)
        //     printf("threadwise mean = %f\n", mean_thread(Number<0>{}));

        // // Thread/Block id
        // const index_t thread_local_id = get_thread_local_1d_id();
        // const index_t block_global_id = get_block_1d_id();
        // const auto thread_cluster_idx =
        //     thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));
        // const auto thread_m_cluster_id = thread_cluster_idx[I0];
        // const auto thread_n_cluster_id = thread_cluster_idx[I1];

        // // step1: Merge mean and variance
        // using ThreadBufferLengths_M_1         = Sequence<MThreadSliceSize, 1>;
        // constexpr auto thread_buffer_desc_m_1 = make_naive_tensor_descriptor_packed(
        //     make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

        // auto threadwise_mean_load_m_k =
        //     ThreadwiseTensorSliceTransfer_v2<MeanDataType,
        //                                      ComputeDataType,
        //                                      MeanVarCountGridDesc_M_N,
        //                                      decltype(thread_buffer_desc_m_1),
        //                                      ThreadBufferLengths_M_1,
        //                                      Sequence<0, 1>,
        //                                      1,
        //                                      1,
        //                                      1,
        //                                      true>(mean_var_count_grid_desc_m_n,
        //                                            make_multi_index(0, 0));

        // auto threadwise_var_load_m_k =
        //     ThreadwiseTensorSliceTransfer_v2<VarDataType,
        //                                      ComputeDataType,
        //                                      MeanVarCountGridDesc_M_N,
        //                                      decltype(thread_buffer_desc_m_1),
        //                                      ThreadBufferLengths_M_1,
        //                                      Sequence<0, 1>,
        //                                      1,
        //                                      1,
        //                                      1,
        //                                      true>(
        //         mean_var_count_grid_desc_m_n,
        //         make_multi_index(block_global_id * M_BlockTileSize +
        //                              thread_m_cluster_id * MThreadSliceSize,
        //                          thread_n_cluster_id));

        // auto threadwise_count_load_m_k =
        //     ThreadwiseTensorSliceTransfer_v2<int32_t,
        //                                      int32_t,
        //                                      MeanVarCountGridDesc_M_N,
        //                                      decltype(thread_buffer_desc_m_1),
        //                                      ThreadBufferLengths_M_1,
        //                                      Sequence<0, 1>,
        //                                      1,
        //                                      1,
        //                                      1,
        //                                      true>(
        //         mean_var_count_grid_desc_m_n,
        //         make_multi_index(block_global_id * M_BlockTileSize +
        //                              thread_m_cluster_id * MThreadSliceSize,
        //                          thread_n_cluster_id));

        // const auto welford_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        //     p_in_welford_mean_grid, mean_var_count_grid_desc_m_n.GetElementSpaceSize());

        // const auto welford_var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        //     p_in_welford_var_grid, mean_var_count_grid_desc_m_n.GetElementSpaceSize());

        // const auto welford_count_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        //     p_in_welford_count_grid, mean_var_count_grid_desc_m_n.GetElementSpaceSize());

        // StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
        //     in_welford_mean_thread_buf;
        // StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
        //     in_welford_var_thread_buf;
        // StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
        //     in_welford_count_thread_buf;

        // StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
        //     welford_mean_thread_buf;
        // StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
        //     welford_var_thread_buf;
        // StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
        //     welford_count_thread_buf;

        // constexpr auto mean_var_count_thread_copy_step_m_n =
        //     make_multi_index(0, NThreadClusterSize);

        // static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
        //     welford_mean_thread_buf(I)  = type_convert<ComputeDataType>(0.0f);
        //     welford_var_thread_buf(I)   = type_convert<ComputeDataType>(0.0f);
        //     welford_count_thread_buf(I) = 0;
        // });

        // for(index_t reducedTiles = 0; reducedTiles < numMeanVarCountBlockTileIteration_N;
        //     ++reducedTiles)
        // {
        //     threadwise_mean_load_m_k.Run(mean_var_count_grid_desc_m_n,
        //                                  welford_mean_global_val_buf,
        //                                  thread_buffer_desc_m_1,
        //                                  make_tuple(I0, I0),
        //                                  in_welford_mean_thread_buf);

        //     // threadwise_var_load_m_k.Run(mean_var_count_grid_desc_m_n,
        //     //                             welford_var_global_val_buf,
        //     //                             thread_buffer_desc_m_1,
        //     //                             make_tuple(I0, I0),
        //     //                             in_welford_var_thread_buf);

        //     // threadwise_count_load_m_k.Run(mean_var_count_grid_desc_m_n,
        //     //                               welford_count_global_val_buf,
        //     //                               thread_buffer_desc_m_1,
        //     //                               make_tuple(I0, I0),
        //     //                               in_welford_count_thread_buf);

        //     // ThreadwiseWelford::Run(in_welford_mean_thread_buf,
        //     //                        in_welford_var_thread_buf,
        //     //                        in_welford_count_thread_buf,
        //     //                        welford_mean_thread_buf,
        //     //                        welford_var_thread_buf,
        //     //                        welford_count_thread_buf);

        //     // threadwise_mean_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_n,
        //     //                                             mean_var_count_thread_copy_step_m_n);
        //     // threadwise_var_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_n,
        //     //                                            mean_var_count_thread_copy_step_m_n);
        //     // threadwise_count_load_m_k.MoveSrcSliceWindow(mean_var_count_grid_desc_m_n,
        //     //                                              mean_var_count_thread_copy_step_m_n);

        //     static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
        //         if(get_thread_global_1d_id() == 0)
        //             printf("mean = %f, var = %f, count = %d\n",
        //                    in_welford_mean_thread_buf(I),
        //                    in_welford_var_thread_buf(I),
        //                    in_welford_count_thread_buf(I));
        //     });
        // }

        // static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
        //     if constexpr(I > 0)
        //         block_sync_lds();

        //     if(get_thread_global_1d_id() == 0)
        //         printf("count = %d\n", welford_count_thread_buf(I));

        //     BlockwiseWelford::Run(
        //         welford_mean_thread_buf(I), welford_var_thread_buf(I),
        //         welford_count_thread_buf(I));
        // });

    } // run
};

} // namespace ck
