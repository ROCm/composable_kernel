// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// Y = LayerNorm(A + B, Beta, Gamma)
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType,
          typename ElementwiseOperation,
          typename AccElementwiseOperation,
          typename GridDesc_M_K,
          typename GridDesc_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          bool SweepOnce>
struct GridwiseAddLayernormWelfordVariance_mk_to_mk
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((YDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (YDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XSrcVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<AccDataType, ThreadReduceSrcDesc_M_K, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<AccDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_K,
                                              ThreadClusterArrangeOrder>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static int GetKPerThread(const GridDesc_M_K& x_grid_desc_m_k,
                                        int thread_k_cluster_id)
    {
        int kPerBlock = x_grid_desc_m_k.GetTransforms()[I0].GetUpperLengths()[I1];
        int kPerThread =
            kPerBlock < K_BlockTileSize ? 0 : KThreadSliceSize * (kPerBlock / K_BlockTileSize);
        int kPerBlockTail = kPerBlock - kPerThread * KThreadClusterSize;

        if(kPerBlockTail > 0)
        {
            int thread_max_len = (thread_k_cluster_id + 1) * KThreadSliceSize;
            int delta          = thread_max_len - kPerBlockTail;
            delta              = math::clamp(thread_max_len - kPerBlockTail, 0, KThreadSliceSize);
            kPerThread += KThreadSliceSize - delta;
        }

        return kPerThread;
    }

    __device__ static void Run(const GridDesc_M_K& a_grid_desc_m_k,
                               const GridDesc_M_K& b_grid_desc_m_k,
                               const GridDesc_K& gamma_grid_desc_k,
                               const GridDesc_K& beta_grid_desc_k,
                               const GridDesc_M_K& y_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               AccDataType epsilon,
                               const ADataType* const __restrict__ p_a_global,
                               const BDataType* const __restrict__ p_b_global,
                               CDataType* const __restrict__ p_c_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const BetaDataType* const __restrict__ p_beta_global,
                               YDataType* const __restrict__ p_y_global,
                               const ElementwiseOperation elementwise_op,
                               const AccElementwiseOperation acc_elementwise_op)
    {
        if constexpr(SweepOnce)
        {
            num_k_block_tile_iteration = 1;
        }

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());
        
        auto c_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, y_grid_desc_m_k.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            a_thread_buf;
        
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            b_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            c_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, KThreadSliceSize, true> gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, KThreadSliceSize, true>& beta_thread_buf =
            gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> var_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        using ThreadBufferLengths_K           = Sequence<KThreadSliceSize>;
        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));
        constexpr auto thread_buffer_desc_k =
            make_naive_tensor_descriptor_packed(make_tuple(Number<KThreadSliceSize>{}));

        auto threadwise_a_load = ThreadwiseTensorSliceTransfer_v2<ADataType,
                                                                  AccDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            a_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));
        
        auto threadwise_b_load = ThreadwiseTensorSliceTransfer_v2<BDataType,
                                                                  AccDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            b_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_c_load = ThreadwiseTensorSliceTransfer_v2<CDataType,
                                                                  AccDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            a_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             AccDataType,
                                             GridDesc_K,
                                             decltype(thread_buffer_desc_k),
                                             ThreadBufferLengths_K,
                                             Sequence<0>,
                                             0,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_k, make_multi_index(thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_beta_load = ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                                                     AccDataType,
                                                                     GridDesc_K,
                                                                     decltype(thread_buffer_desc_k),
                                                                     ThreadBufferLengths_K,
                                                                     Sequence<0>,
                                                                     0,
                                                                     BetaSrcVectorSize,
                                                                     1,
                                                                     true>(
            beta_grid_desc_k, make_multi_index(thread_k_cluster_id * KThreadSliceSize));
        
        auto threadwise_c_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               CDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               AccElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                a_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
                acc_elementwise_op);

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               AccElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
                acc_elementwise_op);

        // Copy x from Cache
        // one pass: fwd, second pass: bwd
        constexpr auto thread_copy_fwd_step_k = make_multi_index(SweepOnce ? 0 : K_BlockTileSize);
        constexpr auto thread_copy_bwd_step_k = make_multi_index(SweepOnce ? 0 : -K_BlockTileSize);

        constexpr auto thread_copy_fwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : K_BlockTileSize);
        constexpr auto thread_copy_bwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        const auto a_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m_k.GetElementSpaceSize());
        
        const auto b_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m_k.GetElementSpaceSize());

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_k.GetElementSpaceSize());

        auto threadwise_welford       = ThreadwiseWelford();
        threadwise_welford.max_count_ = GetKPerThread(a_grid_desc_m_k, thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {

            threadwise_a_load.Run(a_grid_desc_m_k,
                                  a_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  a_thread_buf);

            threadwise_b_load.Run(b_grid_desc_m_k,
                                  b_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  b_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                        elementwise_op(c_thread_buf(Number<offset_m_k>{}),a_thread_buf(Number<offset_m_k>{}),b_thread_buf(Number<offset_m_k>{}));
                        //c_thread_buf(Number<offset_m_k>{}) = a_thread_buf(Number<offset_m_k>{}) + b_thread_buf(Number<offset_m_k>{});
                });
            });
            threadwise_welford.Run(c_thread_buf, mean_thread_buf, var_thread_buf);

            threadwise_c_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   c_thread_buf,
                                   a_grid_desc_m_k,
                                   c_global_val_buf);
            threadwise_a_load.MoveSrcSliceWindow(a_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_b_load.MoveSrcSliceWindow(b_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_c_store.MoveDstSliceWindow(a_grid_desc_m_k, thread_copy_fwd_step_m_k);


        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            int count = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(mean_thread_buf(I), var_thread_buf(I), count);
        });

        auto thread_copy_tail_m_k = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_m_k;
        auto thread_copy_tail_k   = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_k;

        threadwise_c_load.MoveSrcSliceWindow(a_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_k, thread_copy_tail_k);
        threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_k, thread_copy_tail_k);
        threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            if constexpr(!SweepOnce)
            {
                threadwise_c_load.Run(a_grid_desc_m_k,
                                      c_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      c_thread_buf);
            }

            threadwise_gamma_load.Run(gamma_grid_desc_k,
                                      gamma_global_val_buf,
                                      thread_buffer_desc_k,
                                      make_tuple(I0),
                                      gamma_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {

                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    constexpr auto offset_k = thread_buffer_desc_k.CalculateOffset(make_tuple(iK));

                    // normalize
                    y_thread_buf(Number<offset_m_k>{}) =
                        (c_thread_buf(Number<offset_m_k>{}) - mean_thread_buf(iM)) /
                        sqrt(var_thread_buf(iM) + epsilon);

                    // gamma
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) * gamma_thread_buf(Number<offset_k>{});
                });
            });

            threadwise_beta_load.Run(beta_grid_desc_k,
                                     beta_global_val_buf,
                                     thread_buffer_desc_k,
                                     make_tuple(I0),
                                     beta_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    constexpr auto offset_k = thread_buffer_desc_k.CalculateOffset(make_tuple(iK));

                    // beta
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) + beta_thread_buf(Number<offset_k>{});
                });
            });

            threadwise_y_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   y_thread_buf,
                                   y_grid_desc_m_k,
                                   y_global_val_buf);

            threadwise_c_load.MoveSrcSliceWindow(a_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_k, thread_copy_bwd_step_k);
            threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_k, thread_copy_bwd_step_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_bwd_step_m_k);
        }

    }
};

} // namespace ck
