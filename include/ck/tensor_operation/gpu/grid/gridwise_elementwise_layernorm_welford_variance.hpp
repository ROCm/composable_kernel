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
template <typename InDataTypePointerTuple,
          typename CDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType,
          typename ElementwiseOperation,
          typename AccElementwiseOperation,
          typename InGrid2dDescTuple,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          bool SweepOnce>
struct GridwiseElementwiseLayernormWelfordVariance_mk_to_mk
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((YDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (YDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr index_t NumInput = InDataTypePointerTuple::Size();

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

    __device__ static void Run(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                               const GridDesc_M_K& c_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& beta_grid_desc_m_k,
                               const GridDesc_M_K& y_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               AccDataType epsilon,
                               const InDataTypePointerTuple p_in_global_tuple,
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

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                static_assert(in_grid_2d_desc_tuple[I].GetNumOfDimension() == 2); // matrix dimension

                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());

        auto c_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_grid_desc_m_k.GetElementSpaceSize());

        auto in_thread_buf_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * KThreadSliceSize,
                                    true>{};
            },
            Number<NumInput>{});

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            c_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true> gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>& beta_thread_buf =
            gamma_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> var_thread_buf;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K         = Sequence<MThreadSliceSize, KThreadSliceSize>;

        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        auto in_global_load_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return ThreadwiseTensorSliceTransfer_v2<
                       DataType,
                       AccDataType,
                       decltype(in_grid_2d_desc_tuple[I]),
                       decltype(thread_buffer_desc_m_k),
                       ThreadBufferLengths_M_K,    //
                       ThreadBufferDimAccessOrder, // DimAccessOrder
                       XSrcVectorDim,              // SrcVectorDim
                       XSrcVectorSize,
                       1,
                       false>{in_grid_2d_desc_tuple[I],
                              make_multi_index(block_global_id * M_BlockTileSize +
                                                   thread_m_cluster_id * MThreadSliceSize,
                                               thread_k_cluster_id * KThreadSliceSize)};
            },
            Number<NumInput>{});

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
            c_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             AccDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
            gamma_grid_desc_m_k, 
            make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                            thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_beta_load = ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                                                     AccDataType,
                                                                     GridDesc_M_K,
                                                                     decltype(thread_buffer_desc_m_k),
                                                                     ThreadBufferLengths_M_K,
                                                                     ThreadBufferDimAccessOrder,
                                                                     BetaSrcVectorDim,
                                                                     BetaSrcVectorSize,
                                                                     1,
                                                                     true>(
            beta_grid_desc_m_k, 
            make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                            thread_k_cluster_id * KThreadSliceSize));

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
                c_grid_desc_m_k,
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

        constexpr auto thread_copy_fwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : K_BlockTileSize);
        constexpr auto thread_copy_bwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_m_k.GetElementSpaceSize());

        auto threadwise_welford       = ThreadwiseWelford();
        threadwise_welford.max_count_ = GetKPerThread(c_grid_desc_m_k, thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            static_for<0, NumInput, 1>{}([&](auto I) {
                in_global_load_tuple(I).Run(in_grid_2d_desc_tuple[I],
                                            in_global_buf_tuple[I],
                                            thread_buffer_desc_m_k,
                                            make_tuple(I0, I0),
                                            in_thread_buf_tuple(I));

                in_global_load_tuple(I).MoveSrcSliceWindow(in_grid_2d_desc_tuple[I],
                                                           thread_copy_fwd_step_m_k);
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // get reference to in data
                    const auto in_data_refs = generate_tie(
                        // return type should be lvalue
                        [&](auto I) -> const auto& {
                            return in_thread_buf_tuple(I)(Number<offset_m_k>{});
                        },
                        Number<NumInput>{});

                    // get reference to dst data
                    auto out_data_refs = generate_tie(
                        // return type should be lvalue
                        [&](auto I) -> auto& { return c_thread_buf(Number<offset_m_k>{}); },
                        I1);

                    unpack2(elementwise_op, out_data_refs, in_data_refs);
                });
            });
            threadwise_welford.Run(c_thread_buf, mean_thread_buf, var_thread_buf);

            if constexpr(!SweepOnce) // if not sweeponce, store c into global memory for reuse in the next loop
            {
                threadwise_c_store.Run(thread_buffer_desc_m_k,
                                       make_tuple(I0, I0),
                                       c_thread_buf,
                                       c_grid_desc_m_k,
                                       c_global_val_buf);
                threadwise_c_store.MoveDstSliceWindow(c_grid_desc_m_k, thread_copy_fwd_step_m_k);
            }
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            int count = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(mean_thread_buf(I), var_thread_buf(I), count);
        });

        auto thread_copy_tail_m_k = (num_k_block_tile_iteration - 1) * thread_copy_fwd_step_m_k;

        if constexpr(!SweepOnce) //if not sweeponce, store c into global memory for reuse in the next loop
            threadwise_c_load.MoveSrcSliceWindow(c_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            if constexpr(!SweepOnce) //if not sweeponce, reload c from global memory for reuse
            {
                threadwise_c_load.Run(c_grid_desc_m_k,
                                      c_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      c_thread_buf);
            }

            threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                      gamma_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0,I0),
                                      gamma_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // normalize
                    y_thread_buf(Number<offset_m_k>{}) =
                        (c_thread_buf(Number<offset_m_k>{}) - mean_thread_buf(iM)) /
                        sqrt(var_thread_buf(iM) + epsilon);

                    // gamma
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) * gamma_thread_buf(Number<offset_m_k>{});
                });
            });

            threadwise_beta_load.Run(beta_grid_desc_m_k,
                                     beta_global_val_buf,
                                     thread_buffer_desc_m_k,
                                     make_tuple(I0, I0),
                                     beta_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));

                    // beta
                    y_thread_buf(Number<offset_m_k>{}) =
                        y_thread_buf(Number<offset_m_k>{}) + beta_thread_buf(Number<offset_m_k>{});
                });
            });

            threadwise_y_store.Run(thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   y_thread_buf,
                                   y_grid_desc_m_k,
                                   y_global_val_buf);

            if constexpr(!SweepOnce)
                threadwise_c_load.MoveSrcSliceWindow(c_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_bwd_step_m_k);
        }

    }
};

} // namespace ck
