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
          typename MeanVarGridDesc_M_NBlock,
          typename CountGridDesc_M_NBlock,
          typename GammaBetaGridDesc_N,
          typename HElementwiseOperation,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t NThreadClusterSize,
          index_t MThreadSliceSize,
          index_t NThreadSliceSize,
          index_t ESrcVectorSize,
          index_t HDstVectorSize,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorSize>
struct GridwiseWelfordSecondHalfLayernorm2d
{
    static_assert(NThreadSliceSize % ESrcVectorSize == 0 &&
                      NThreadSliceSize % GammaSrcVectorSize == 0 &&
                      NThreadSliceSize % BetaSrcVectorSize == 0,
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(NThreadSliceSize % HDstVectorSize == 0,
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using ThreadClusterLengths_M_N   = Sequence<MThreadClusterSize, NThreadClusterSize>;
    using ThreadBufferDimAccessOrder = Sequence<0, 1>;
    using ThreadClusterArrangeOrder  = Sequence<0, 1>;

    static constexpr auto thread_cluster_desc_m_n =
        make_cluster_descriptor(ThreadClusterLengths_M_N{}, ThreadClusterArrangeOrder{});

    using ThreadBufferLengths_M_N                = Sequence<MThreadSliceSize, NThreadSliceSize>;
    static constexpr auto thread_buffer_desc_m_n = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<NThreadSliceSize>{}));

    using ThreadBufferLengths_M_1 = Sequence<MThreadSliceSize, 1>;
    static constexpr auto thread_buffer_desc_m_1 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}, Number<1>{}));

    using ThreadBufferLengths_N = Sequence<NThreadSliceSize>;
    static constexpr auto thread_buffer_desc_n =
        make_naive_tensor_descriptor_packed(make_tuple(Number<NThreadSliceSize>{}));

    using ThreadReduceSrcDesc_M_1 = decltype(thread_buffer_desc_m_1);
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelfordMerge<ComputeDataType, ThreadReduceSrcDesc_M_1, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<ComputeDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_N,
                                              ThreadClusterArrangeOrder>;

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
                               const MeanVarGridDesc_M_NBlock& mean_var_grid_desc_m_n,
                               const CountGridDesc_M_NBlock& count_grid_desc_m_n,
                               const GammaBetaGridDesc_N& gamma_grid_desc_n,
                               const GammaBetaGridDesc_N& beta_grid_desc_n,
                               index_t numMeanVarCountBlockTileIteration_N,
                               index_t numNormBlockTileIteration_N,
                               ComputeDataType epsilon,
                               HElementwiseOperation h_element_op)
    {
        // Thread/Block id
        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const auto thread_cluster_idx =
            thread_cluster_desc_m_n.CalculateBottomIndex(make_multi_index(thread_local_id));
        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_n_cluster_id = thread_cluster_idx[I1];

        // Global Memory
        const auto e_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_m_n.GetElementSpaceSize());

        const auto welford_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_mean_grid, mean_var_grid_desc_m_n.GetElementSpaceSize());

        const auto welford_var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_var_grid, mean_var_grid_desc_m_n.GetElementSpaceSize());

        const auto welford_count_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_welford_count_grid, count_grid_desc_m_n.GetElementSpaceSize());

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_grid, gamma_grid_desc_n.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_grid, beta_grid_desc_n.GetElementSpaceSize());

        auto h_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_h_grid, h_grid_desc_m_n.GetElementSpaceSize());

        // VGPR
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            in_welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            in_welford_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            in_welford_count_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            welford_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            welford_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            welford_count_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     ComputeDataType,
                     MThreadSliceSize * ESrcVectorSize,
                     true>
            e_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr,
                     ComputeDataType,
                     MThreadSliceSize * GammaSrcVectorSize,
                     true>
            gamma_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr,
                     ComputeDataType,
                     MThreadSliceSize * BetaSrcVectorSize,
                     true>
            beta_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr,
                     ComputeDataType,
                     MThreadSliceSize * HDstVectorSize,
                     true>
            h_thread_buf;

        // IO
        auto threadwise_mean_load_m_nblock =
            ThreadwiseTensorSliceTransfer_v2<MeanDataType,
                                             ComputeDataType,
                                             MeanVarGridDesc_M_NBlock,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             ThreadBufferDimAccessOrder,
                                             1,
                                             1,
                                             1,
                                             true>(
                mean_var_grid_desc_m_n,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id));

        auto threadwise_var_load_m_nblock =
            ThreadwiseTensorSliceTransfer_v2<VarDataType,
                                             ComputeDataType,
                                             MeanVarGridDesc_M_NBlock,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             ThreadBufferDimAccessOrder,
                                             1,
                                             1,
                                             1,
                                             true>(
                mean_var_grid_desc_m_n,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id));

        auto threadwise_count_load_m_nblock =
            ThreadwiseTensorSliceTransfer_v2<int32_t,
                                             int32_t,
                                             CountGridDesc_M_NBlock,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             ThreadBufferDimAccessOrder,
                                             1,
                                             1,
                                             1,
                                             true>(
                count_grid_desc_m_n,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id));

        auto threadwise_e_load_m_n =
            ThreadwiseTensorSliceTransfer_v2<EDataType,
                                             ComputeDataType,
                                             decltype(e_grid_desc_m_n),
                                             decltype(thread_buffer_desc_m_n),
                                             ThreadBufferLengths_M_N,
                                             ThreadBufferDimAccessOrder,
                                             1, // SrcVectorDim
                                             ESrcVectorSize,
                                             1,
                                             true>(
                e_grid_desc_m_n,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id * NThreadSliceSize));

        auto threadwise_gamma_load_m_n =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             ComputeDataType,
                                             decltype(gamma_grid_desc_n),
                                             decltype(thread_buffer_desc_n),
                                             ThreadBufferLengths_N,
                                             Sequence<0>, // DimAccessOrder,
                                             0,           // SrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_n, make_multi_index(thread_n_cluster_id * NThreadSliceSize));

        auto threadwise_beta_load_m_n =
            ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                             ComputeDataType,
                                             decltype(beta_grid_desc_n),
                                             decltype(thread_buffer_desc_n),
                                             ThreadBufferLengths_N,
                                             Sequence<0>, // DimAccessOrder,
                                             0,           // SrcVectorDim,
                                             BetaSrcVectorSize,
                                             1,
                                             true>(
                beta_grid_desc_n, make_multi_index(thread_n_cluster_id * NThreadSliceSize));

        auto threadwise_h_store_m_n =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               HDataType,
                                               decltype(thread_buffer_desc_m_n),
                                               decltype(h_grid_desc_m_n),
                                               HElementwiseOperation,
                                               ThreadBufferLengths_M_N,
                                               ThreadBufferDimAccessOrder,
                                               1, // DstVectorDim
                                               HDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                h_grid_desc_m_n,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_n_cluster_id * NThreadSliceSize),
                h_element_op);

        // step1: Merge mean and variance
        constexpr auto mean_var_count_thread_copy_step_m_n =
            make_multi_index(0, NThreadClusterSize);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            welford_mean_thread_buf(I)  = type_convert<ComputeDataType>(0.0f);
            welford_var_thread_buf(I)   = type_convert<ComputeDataType>(0.0f);
            welford_count_thread_buf(I) = 0;
        });

        for(index_t reducedTiles = 0; reducedTiles < numMeanVarCountBlockTileIteration_N;
            ++reducedTiles)
        {
            threadwise_mean_load_m_nblock.Run(mean_var_grid_desc_m_n,
                                              welford_mean_global_val_buf,
                                              thread_buffer_desc_m_1,
                                              make_tuple(I0, I0),
                                              in_welford_mean_thread_buf);

            threadwise_var_load_m_nblock.Run(mean_var_grid_desc_m_n,
                                             welford_var_global_val_buf,
                                             thread_buffer_desc_m_1,
                                             make_tuple(I0, I0),
                                             in_welford_var_thread_buf);

            threadwise_count_load_m_nblock.Run(count_grid_desc_m_n,
                                               welford_count_global_val_buf,
                                               thread_buffer_desc_m_1,
                                               make_tuple(I0, I0),
                                               in_welford_count_thread_buf);

            ThreadwiseWelford::Run(in_welford_mean_thread_buf,
                                   in_welford_var_thread_buf,
                                   in_welford_count_thread_buf,
                                   welford_mean_thread_buf,
                                   welford_var_thread_buf,
                                   welford_count_thread_buf);

            threadwise_mean_load_m_nblock.MoveSrcSliceWindow(mean_var_grid_desc_m_n,
                                                             mean_var_count_thread_copy_step_m_n);
            threadwise_var_load_m_nblock.MoveSrcSliceWindow(mean_var_grid_desc_m_n,
                                                            mean_var_count_thread_copy_step_m_n);
            threadwise_count_load_m_nblock.MoveSrcSliceWindow(count_grid_desc_m_n,
                                                              mean_var_count_thread_copy_step_m_n);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseWelford::Run(
                welford_mean_thread_buf(I), welford_var_thread_buf(I), welford_count_thread_buf(I));
        });

        // step2: normalization
        for(index_t reducedTiles = 0; reducedTiles < numNormBlockTileIteration_N; ++reducedTiles)
        {
            // h[m, n] = [(e[m, n] - mean[m]) / sqrt(var[m] + eps)] * gamma[n] + beta[n]
            threadwise_e_load_m_n.Run(e_grid_desc_m_n,
                                      e_global_val_buf,
                                      thread_buffer_desc_m_n,
                                      make_tuple(I0, I0),
                                      e_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto m) {
                auto divisor = 1 / __builtin_amdgcn_sqrtf(welford_var_thread_buf(m) + epsilon);
                static_for<0, NThreadSliceSize, 1>{}([&](auto n) {
                    constexpr auto m_n = thread_buffer_desc_m_n.CalculateOffset(make_tuple(m, n));
                    h_thread_buf(Number<m_n>{}) =
                        (e_thread_buf(Number<m_n>{}) - welford_mean_thread_buf(m)) * divisor;
                });
            });

            threadwise_gamma_load_m_n.Run(gamma_grid_desc_n,
                                          gamma_global_val_buf,
                                          thread_buffer_desc_n,
                                          make_tuple(I0),
                                          gamma_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto m) {
                static_for<0, NThreadSliceSize, 1>{}([&](auto n) {
                    constexpr auto m_n = thread_buffer_desc_m_n.CalculateOffset(make_tuple(m, n));
                    h_thread_buf(Number<m_n>{}) = h_thread_buf(Number<m_n>{}) * gamma_thread_buf(n);
                });
            });

            threadwise_beta_load_m_n.Run(beta_grid_desc_n,
                                         beta_global_val_buf,
                                         thread_buffer_desc_n,
                                         make_tuple(I0),
                                         beta_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto m) {
                static_for<0, NThreadSliceSize, 1>{}([&](auto n) {
                    constexpr auto m_n = thread_buffer_desc_m_n.CalculateOffset(make_tuple(m, n));
                    h_thread_buf(Number<m_n>{}) = h_thread_buf(Number<m_n>{}) + beta_thread_buf(n);
                });
            });

            threadwise_h_store_m_n.Run(thread_buffer_desc_m_n,
                                       make_tuple(I0, I0),
                                       h_thread_buf,
                                       h_grid_desc_m_n,
                                       h_global_val_buf);

            threadwise_e_load_m_n.MoveSrcSliceWindow(e_grid_desc_m_n,
                                                     make_multi_index(0, N_BlockTileSize));
            threadwise_gamma_load_m_n.MoveSrcSliceWindow(gamma_grid_desc_n,
                                                         make_multi_index(N_BlockTileSize));
            threadwise_beta_load_m_n.MoveSrcSliceWindow(beta_grid_desc_n,
                                                        make_multi_index(N_BlockTileSize));
            threadwise_h_store_m_n.MoveDstSliceWindow(h_grid_desc_m_n,
                                                      make_multi_index(0, N_BlockTileSize));
        }

    } // run
};

} // namespace ck
