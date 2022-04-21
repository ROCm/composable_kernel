/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_GRIDWISE_2D_NORMALIZE_MULTIBLOCK_HPP
#define CK_GRIDWISE_2D_NORMALIZE_MULTIBLOCK_HPP

#include "data_type.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "cluster_descriptor.hpp"
#include "element_wise_operation.hpp"

namespace ck {

template <typename GridwiseNormalization,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename InOutGridDesc_M_K,
          typename ScaleBiasMeanVarGridDesc_M,
          typename TernaryOperationNormalize>
__global__ void
kernel_normalize_multiblock(const InOutGridDesc_M_K in_out_grid_desc_m_k,
                            const ScaleBiasMeanVarGridDesc_M scale_bias_mean_var_grid_desc_m,
                            const TernaryOperationNormalize op_normalize,
                            index_t block_group_size,
                            index_t num_k_block_tile_iteration,
                            const InDataType* const __restrict__ p_in_global,
                            OutDataType* const __restrict__ p_out_global,
                            const AccDataType* const __restrict__ p_scale,
                            const AccDataType* const __restrict__ p_bias,
                            const AccDataType* const __restrict__ p_mean,
                            const AccDataType* const __restrict__ p_invVariance)
{
    GridwiseNormalization::Run(in_out_grid_desc_m_k,
                               scale_bias_mean_var_grid_desc_m,
                               op_normalize,
                               block_group_size,
                               num_k_block_tile_iteration,
                               p_in_global,
                               p_out_global,
                               p_scale,
                               p_bias,
                               p_mean,
                               p_invVariance);
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename InOutGridDesc_M_K,
          typename ScaleBiasMeanVarGridDesc_M,
          typename TernaryOperationNormalize,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InOutVectorDim,
          index_t InOutVectorSize,
          index_t ScaleBiasMeanVarVectorSize>
struct GridwiseNormalizeMultiblock_mk_input_m_scale_bias_mean_var
{
    static_assert(((InOutVectorDim == 0 && MThreadSliceSize % InOutVectorSize == 0) ||
                   (InOutVectorDim == 1 && KThreadSliceSize % InOutVectorSize == 0)) &&
                      (MThreadSliceSize % ScaleBiasMeanVarVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (InOutVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const InOutGridDesc_M_K in_out_grid_desc_m_k,
                               const ScaleBiasMeanVarGridDesc_M scale_bias_mean_var_grid_desc_m,
                               const TernaryOperationNormalize op_normalize,
                               index_t block_group_size,
                               index_t num_k_block_tile_iteration,
                               const InDataType* const __restrict__ p_in_global,
                               OutDataType* const __restrict__ p_out_global,
                               const AccDataType* const __restrict__ p_scale,
                               const AccDataType* const __restrict__ p_bias,
                               const AccDataType* const __restrict__ p_mean,
                               const AccDataType* const __restrict__ p_invVariance)

    {
        const auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global,
            in_out_grid_desc_m_k.GetElementSpaceSize(),
            type_convert<InDataType>(0.0f));
        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, in_out_grid_desc_m_k.GetElementSpaceSize());

        const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_scale,
            scale_bias_mean_var_grid_desc_m.GetElementSpaceSize(),
            type_convert<AccDataType>(0.0f));
        const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_bias,
            scale_bias_mean_var_grid_desc_m.GetElementSpaceSize(),
            type_convert<AccDataType>(0.0f));
        const auto mean_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean,
            scale_bias_mean_var_grid_desc_m.GetElementSpaceSize(),
            type_convert<AccDataType>(0.0f));
        const auto invVariance_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_invVariance,
            scale_bias_mean_var_grid_desc_m.GetElementSpaceSize(),
            type_convert<AccDataType>(0.0f));

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_out_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> scale_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> bias_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true>
            invVariance_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / block_group_size;
        const index_t block_local_id  = block_global_id % block_group_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        constexpr auto thread_m_k_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));
        constexpr auto thread_m_buffer_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

        const index_t assignedSizePerBlock = K_BlockTileSize * num_k_block_tile_iteration;

        auto threadwise_src_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             AccDataType,
                                             InOutGridDesc_M_K,
                                             decltype(thread_m_k_buffer_desc),
                                             Sequence<MThreadSliceSize, KThreadSliceSize>,
                                             ThreadBufferDimAccessOrder,
                                             InOutVectorDim,
                                             InOutVectorSize,
                                             1,
                                             false>(
                in_out_grid_desc_m_k,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_local_id * assignedSizePerBlock +
                                     thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dst_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               OutDataType,
                                               decltype(thread_m_k_buffer_desc),
                                               InOutGridDesc_M_K,
                                               PassThroughOp,
                                               Sequence<MThreadSliceSize, KThreadSliceSize>,
                                               ThreadBufferDimAccessOrder,
                                               InOutVectorDim,
                                               InOutVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               false>(
                in_out_grid_desc_m_k,
                make_multi_index(
                    blkgroup_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                    block_local_id * assignedSizePerBlock + thread_k_cluster_id * KThreadSliceSize),
                PassThroughOp{});

        auto threadwise_scale_bias_mean_var_load =
            ThreadwiseTensorSliceTransfer_v2<AccDataType,
                                             AccDataType,
                                             ScaleBiasMeanVarGridDesc_M,
                                             decltype(thread_m_buffer_desc),
                                             Sequence<MThreadSliceSize>,
                                             Sequence<0>,
                                             0,
                                             ScaleBiasMeanVarVectorSize,
                                             1,
                                             true>(
                scale_bias_mean_var_grid_desc_m,
                make_multi_index(blkgroup_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize));

        constexpr auto in_thread_copy_step = make_multi_index(0, K_BlockTileSize);

        index_t processed_tiles = 0;

        threadwise_scale_bias_mean_var_load.Run(scale_bias_mean_var_grid_desc_m,
                                                scale_global_buf,
                                                thread_m_buffer_desc,
                                                make_tuple(I0),
                                                scale_thread_buf);
        threadwise_scale_bias_mean_var_load.Run(scale_bias_mean_var_grid_desc_m,
                                                bias_global_buf,
                                                thread_m_buffer_desc,
                                                make_tuple(I0),
                                                bias_thread_buf);
        threadwise_scale_bias_mean_var_load.Run(scale_bias_mean_var_grid_desc_m,
                                                mean_global_buf,
                                                thread_m_buffer_desc,
                                                make_tuple(I0),
                                                mean_thread_buf);
        threadwise_scale_bias_mean_var_load.Run(scale_bias_mean_var_grid_desc_m,
                                                invVariance_global_buf,
                                                thread_m_buffer_desc,
                                                make_tuple(I0),
                                                invVariance_thread_buf);

        do
        {
            threadwise_src_load.Run(in_out_grid_desc_m_k,
                                    in_global_buf,
                                    thread_m_k_buffer_desc,
                                    make_tuple(I0, I0),
                                    in_out_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                static_for<0, KThreadSliceSize, 1>{}([&](auto J) {
                    constexpr auto offset = I * Number<KThreadSliceSize>{} + J;
                    op_normalize(in_out_thread_buf(offset),
                                 in_out_thread_buf(offset),
                                 mean_thread_buf(I),
                                 invVariance_thread_buf(I));

                    in_out_thread_buf(offset) =
                        in_out_thread_buf[offset] * scale_thread_buf[I] + bias_thread_buf[I];
                });
            });

            threadwise_dst_store.Run(thread_m_k_buffer_desc,
                                     make_tuple(I0, I0),
                                     in_out_thread_buf,
                                     in_out_grid_desc_m_k,
                                     out_global_buf);

            threadwise_src_load.MoveSrcSliceWindow(in_out_grid_desc_m_k, in_thread_copy_step);
            threadwise_dst_store.MoveDstSliceWindow(in_out_grid_desc_m_k, in_thread_copy_step);

            processed_tiles++;
        } while(processed_tiles < num_k_block_tile_iteration);
    };
};

} // namespace ck
#endif
