// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"

namespace ck {

// Tensor Shape
// dy, x = [M, K], gamma = [1, K], x_mean, inv_std = [M, 1]

// Flow:
// def normalization_backward_x(dy, x, gamma, x_mean, inv_std, reduce_axis, reduce_size):
//     ds = np.sum(dy * gamma * x, axis=reduce_axis, keepdims=True)
//     db = np.sum(dy * gamma, axis=reduce_axis, keepdims=True)
//     b = (db * x_mean - ds) * inv_std ** (3) / reduce_size
//     c = -b * x_mean - db * inv_std / reduce_size
//     dx = inv_std * dy * gamma + b * x + c
//     return dx

template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DXDataType,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t DYSrcVectorDim,
          index_t DYSrcVectorSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t MeanInvStdSrcVectorDim,
          index_t MeanInvStdSrcVectorSize,
          index_t DXDstVectorDim,
          index_t DXDstVectorSize,
          bool SweepOnce>
struct GridwiseNormalizationBwdX_mk_to_mk
{
    // if we just check ThreadSliceSize % VectorSize == 0, the performance may be poor (coalesce)
    static_assert(((DYSrcVectorDim == 0 && MThreadSliceSize == DYSrcVectorSize) ||
                   (DYSrcVectorDim == 1 && KThreadSliceSize == DYSrcVectorSize)),
                  "Invalid thread slice sizes and/or dy vector sizes configuration, please check!");

    static_assert(((XSrcVectorDim == 0 && MThreadSliceSize == XSrcVectorSize) ||
                   (XSrcVectorDim == 1 && KThreadSliceSize == XSrcVectorSize)),
                  "Invalid thread slice sizes and/or x vector sizes configuration, please check!");

    static_assert(
        ((GammaSrcVectorDim == 0 && MThreadSliceSize == GammaSrcVectorSize) ||
         (GammaSrcVectorDim == 1 && KThreadSliceSize == GammaSrcVectorSize)),
        "Invalid thread slice sizes and/or gamma vector sizes configuration, please check!");

    static_assert(
        ((MeanInvStdSrcVectorDim == 0 && MThreadSliceSize == MeanInvStdSrcVectorSize) ||
         (MeanInvStdSrcVectorDim == 1 && KThreadSliceSize == MeanInvStdSrcVectorSize)),
        "Invalid thread slice sizes and/or mean/inv_std vector sizes configuration, please check!");

    static_assert(((DXDstVectorDim == 0 && MThreadSliceSize == DXDstVectorSize) ||
                   (DXDstVectorDim == 1 && KThreadSliceSize == DXDstVectorSize)),
                  "Invalid thread slice sizes and/or dx vector sizes configuration, please check!");

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using DYThreadBufferDimAccessOrder =
        typename conditional<DYSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;
    using XThreadBufferDimAccessOrder =
        typename conditional<XSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;
    using GammaThreadBufferDimAccessOrder =
        typename conditional<GammaSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;
    using MeanInvStdThreadBufferDimAccessOrder =
        typename conditional<MeanInvStdSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;
    using DXThreadBufferDimAccessOrder =
        typename conditional<DXDstVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder = DYThreadBufferDimAccessOrder;
    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadBufferLengths_M_K = Sequence<MThreadSliceSize, KThreadSliceSize>;

    static constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    using BlockwiseSumReduce = PartitionedBlockwiseReduction<ComputeDataType,
                                                             BlockSize,
                                                             ThreadClusterLengths_M_K,
                                                             ThreadClusterArrangeOrder,
                                                             reduce::Add,
                                                             true>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const GridDesc_M_K& dy_grid_desc_m_k,
                               const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& mean_grid_desc_m_k,
                               const GridDesc_M_K& inv_std_grid_desc_m_k,
                               const GridDesc_M_K& dx_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               const DYDataType* const __restrict__ p_dy_global,
                               const XDataType* const __restrict__ p_x_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const MeanInvStdDataType* const __restrict__ p_mean_global,
                               const MeanInvStdDataType* const __restrict__ p_inv_std_global,
                               DXDataType* const __restrict__ p_dx_global)
    {
        // LDS
        __shared__ ComputeDataType p_reduce_work_buffer[BlockSize];

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        // Global
        const auto dy_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dy_global, dy_grid_desc_m_k.GetElementSpaceSize());

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean_global, mean_grid_desc_m_k.GetElementSpaceSize());

        const auto inv_std_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_inv_std_global, inv_std_grid_desc_m_k.GetElementSpaceSize());

        auto dx_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dx_global, dx_grid_desc_m_k.GetElementSpaceSize());

        // VGPR
        auto dy_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                          ComputeDataType,
                                          MThreadSliceSize * KThreadSliceSize,
                                          true>{};

        auto x_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                         ComputeDataType,
                                         MThreadSliceSize * KThreadSliceSize,
                                         true>{};

        auto gamma_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                             ComputeDataType,
                                             MThreadSliceSize * KThreadSliceSize,
                                             true>{};

        auto mean_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                            ComputeDataType,
                                            MThreadSliceSize * KThreadSliceSize,
                                            true>{};

        auto inv_std_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                               ComputeDataType,
                                               MThreadSliceSize * KThreadSliceSize,
                                               true>{};

        auto dx_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                          ComputeDataType,
                                          MThreadSliceSize * KThreadSliceSize,
                                          true>{};

        auto ds_thread_buf =
            StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>{};

        auto db_thread_buf =
            StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>{};

        // thread id
        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        // IO
        auto threadwise_dy_load = ThreadwiseTensorSliceTransfer_v2<DYDataType,
                                                                   ComputeDataType,
                                                                   GridDesc_M_K,
                                                                   decltype(thread_buffer_desc_m_k),
                                                                   ThreadBufferLengths_M_K,
                                                                   DYThreadBufferDimAccessOrder,
                                                                   DYSrcVectorDim,
                                                                   DYSrcVectorSize,
                                                                   1,
                                                                   false>(
            dy_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  ComputeDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  XThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  false>(
            x_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             XThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             false>(
                gamma_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_mean_load =
            ThreadwiseTensorSliceTransfer_v2<MeanInvStdDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             MeanInvStdThreadBufferDimAccessOrder,
                                             MeanInvStdSrcVectorDim,
                                             MeanInvStdSrcVectorSize,
                                             1,
                                             false>(
                mean_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_inv_std_load =
            ThreadwiseTensorSliceTransfer_v2<MeanInvStdDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             MeanInvStdThreadBufferDimAccessOrder,
                                             MeanInvStdSrcVectorDim,
                                             MeanInvStdSrcVectorSize,
                                             1,
                                             false>(
                inv_std_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dx_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               DXDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_K,
                                               DXThreadBufferDimAccessOrder,
                                               DXDstVectorDim,
                                               DXDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               false>(
                dx_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
                PassThroughOp{});

        ComputeDataType reduce_size = type_convert<ComputeDataType>(
            dy_grid_desc_m_k.GetTransforms()[I2].GetUpperLengths()[I0]);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            ds_thread_buf(I) = type_convert<ComputeDataType>(0.0f);
            db_thread_buf(I) = type_convert<ComputeDataType>(0.0f);
        });

        // Separate sweep once and sweep twice pipeline
        // Sweep once: for small k, if KThreadClusterSize * KThreadSliceSize > K
        // we don't need to use loop to read x, dy, gamma twice
        if constexpr(SweepOnce)
        {
            threadwise_dy_load.Run(dy_grid_desc_m_k,
                                   dy_global_val_buf,
                                   thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   dy_thread_buf);

            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                      gamma_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      gamma_thread_buf);

            threadwise_mean_load.Run(mean_grid_desc_m_k,
                                     mean_global_val_buf,
                                     thread_buffer_desc_m_k,
                                     make_tuple(I0, I0),
                                     mean_thread_buf);

            threadwise_inv_std_load.Run(inv_std_grid_desc_m_k,
                                        inv_std_global_val_buf,
                                        thread_buffer_desc_m_k,
                                        make_tuple(I0, I0),
                                        inv_std_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                constexpr auto offset_m =
                    Number<thread_buffer_desc_m.CalculateOffset(make_tuple(iM))>{};

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        Number<thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK))>{};

                    ds_thread_buf(offset_m) += dy_thread_buf[offset_m_k] *
                                               gamma_thread_buf[offset_m_k] *
                                               x_thread_buf[offset_m_k];

                    db_thread_buf(offset_m) +=
                        dy_thread_buf[offset_m_k] * gamma_thread_buf[offset_m_k];
                });
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, ds_thread_buf(I));
                block_sync_lds();
                BlockwiseSumReduce::Reduce(reduce_work_buf, db_thread_buf(I));
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                constexpr auto offset_m =
                    Number<thread_buffer_desc_m.CalculateOffset(make_tuple(iM))>{};

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        Number<thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK))>{};

                    // b  = (db * x_mean - ds) * rstd ** (3) / reduce_size
                    // c  = -b * x_mean - db * rstd / reduce_size
                    // dx = rstd * dy * gamma + b * x + c

                    ComputeDataType b = db_thread_buf[offset_m] * mean_thread_buf[offset_m_k] -
                                        ds_thread_buf[offset_m];

                    b *= inv_std_thread_buf[offset_m_k] * inv_std_thread_buf[offset_m_k] *
                         inv_std_thread_buf[offset_m_k] / reduce_size;

                    ComputeDataType c = -b * mean_thread_buf(offset_m_k);

                    c -= db_thread_buf[offset_m] * inv_std_thread_buf[offset_m_k] / reduce_size;

                    dx_thread_buf(offset_m_k) = dy_thread_buf[offset_m_k] *
                                                    gamma_thread_buf[offset_m_k] *
                                                    inv_std_thread_buf[offset_m_k] +
                                                b * x_thread_buf[offset_m_k] + c;
                });
            });

            threadwise_dx_store.Run(thread_buffer_desc_m_k,
                                    make_tuple(I0, I0),
                                    dx_thread_buf,
                                    dx_grid_desc_m_k,
                                    dx_global_val_buf);

        }    // end of sweep once
        else // Sweep Twice pipeline
        {
            constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileSize);

            for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
            {
                threadwise_dy_load.Run(dy_grid_desc_m_k,
                                       dy_global_val_buf,
                                       thread_buffer_desc_m_k,
                                       make_tuple(I0, I0),
                                       dy_thread_buf);

                threadwise_x_load.Run(x_grid_desc_m_k,
                                      x_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      x_thread_buf);

                threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                          gamma_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          gamma_thread_buf);

                threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, thread_copy_fwd_step_m_k);
                threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
                threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                         thread_copy_fwd_step_m_k);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    constexpr auto offset_m =
                        Number<thread_buffer_desc_m.CalculateOffset(make_tuple(iM))>{};

                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset_m_k =
                            Number<thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK))>{};

                        ds_thread_buf(offset_m) += dy_thread_buf[offset_m_k] *
                                                   gamma_thread_buf[offset_m_k] *
                                                   x_thread_buf[offset_m_k];

                        db_thread_buf(offset_m) +=
                            dy_thread_buf[offset_m_k] * gamma_thread_buf[offset_m_k];
                    });
                });
            } // end of first sweep

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, ds_thread_buf(I));
                block_sync_lds();
                BlockwiseSumReduce::Reduce(reduce_work_buf, db_thread_buf(I));
            });

            // reverse read for using dy, gamma and x in the cache
            constexpr auto thread_copy_bwd_step_m_k = make_multi_index(0, -K_BlockTileSize);
            auto thread_copy_tail_m_k = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_m_k;

            // move to tail
            threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_bwd_step_m_k);

            // move from start to tail
            threadwise_mean_load.MoveSrcSliceWindow(mean_grid_desc_m_k, thread_copy_tail_m_k);
            threadwise_inv_std_load.MoveSrcSliceWindow(inv_std_grid_desc_m_k, thread_copy_tail_m_k);
            threadwise_dx_store.MoveDstSliceWindow(dx_grid_desc_m_k, thread_copy_tail_m_k);

            for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
            {
                threadwise_dy_load.Run(dy_grid_desc_m_k,
                                       dy_global_val_buf,
                                       thread_buffer_desc_m_k,
                                       make_tuple(I0, I0),
                                       dy_thread_buf);

                threadwise_x_load.Run(x_grid_desc_m_k,
                                      x_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      x_thread_buf);

                threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                          gamma_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          gamma_thread_buf);

                threadwise_mean_load.Run(mean_grid_desc_m_k,
                                         mean_global_val_buf,
                                         thread_buffer_desc_m_k,
                                         make_tuple(I0, I0),
                                         mean_thread_buf);

                threadwise_inv_std_load.Run(inv_std_grid_desc_m_k,
                                            inv_std_global_val_buf,
                                            thread_buffer_desc_m_k,
                                            make_tuple(I0, I0),
                                            inv_std_thread_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    constexpr auto offset_m =
                        Number<thread_buffer_desc_m.CalculateOffset(make_tuple(iM))>{};

                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset_m_k =
                            Number<thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK))>{};

                        // b  = (db * x_mean - ds) * rstd ** (3) / reduce_size
                        // c  = -b * x_mean - db * rstd / reduce_size
                        // dx = rstd * dy * gamma + b * x + c

                        ComputeDataType b = db_thread_buf[offset_m] * mean_thread_buf[offset_m_k] -
                                            ds_thread_buf[offset_m];

                        b *= inv_std_thread_buf[offset_m_k] * inv_std_thread_buf[offset_m_k] *
                             inv_std_thread_buf[offset_m_k] / reduce_size;

                        ComputeDataType c = -b * mean_thread_buf(offset_m_k);

                        c -= db_thread_buf[offset_m] * inv_std_thread_buf[offset_m_k] / reduce_size;

                        dx_thread_buf(offset_m_k) = dy_thread_buf[offset_m_k] *
                                                        gamma_thread_buf[offset_m_k] *
                                                        inv_std_thread_buf[offset_m_k] +
                                                    b * x_thread_buf[offset_m_k] + c;
                    });
                });

                threadwise_dx_store.Run(thread_buffer_desc_m_k,
                                        make_tuple(I0, I0),
                                        dx_thread_buf,
                                        dx_grid_desc_m_k,
                                        dx_global_val_buf);

                threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, thread_copy_bwd_step_m_k);
                threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
                threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                         thread_copy_bwd_step_m_k);
                threadwise_mean_load.MoveSrcSliceWindow(mean_grid_desc_m_k,
                                                        thread_copy_bwd_step_m_k);
                threadwise_inv_std_load.MoveSrcSliceWindow(inv_std_grid_desc_m_k,
                                                           thread_copy_bwd_step_m_k);
                threadwise_dx_store.MoveDstSliceWindow(dx_grid_desc_m_k, thread_copy_bwd_step_m_k);
            }
        }
    }
};

} // namespace ck
