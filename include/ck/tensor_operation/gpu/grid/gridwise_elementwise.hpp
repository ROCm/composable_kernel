// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {

template <typename GridwiseElementwiseFunctor,
          typename InGridDescTuple,
          typename OutGridDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename Block2TileMap,
          typename ElementwiseOperation>
__global__ void 
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
kernel_elementwise(const InGridDescTuple in_grid_desc_tuple,
                                      const OutGridDescTuple out_grid_desc_tuple,
                                      const InDataTypePointerTuple p_in_global_tuple,
                                      const OutDataTypePointerTuple p_out_global_tuple,
                                      const Block2TileMap block_2_tile_map,
                                      const ElementwiseOperation elementwise_op)
{
    GridwiseElementwiseFunctor::Run(in_grid_desc_tuple,
                                      out_grid_desc_tuple,
                                      p_in_global_tuple,
                                      p_out_global_tuple,
                                      block_2_tile_map,
                                      elementwise_op);
}

template <typename InGridDescTuple,
          typename OutGridDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename Block2TileMap,
          typename ElementwiseOperation,
          index_t BlockSize,
          index_t M0PerBlock,
          index_t M1PerBlock,
          index_t M0PerThread,
          index_t M1PerThread,
          typename ThreadClusterArrangeOrder,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq,
          bool InOutSameVectorDim>
struct GridwiseElementwise
{
    static constexpr index_t NumInput  = InDataTypePointerTuple::Size();
    static constexpr index_t NumOutput = OutDataTypePointerTuple::Size();

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size() &&
                      NumInput == InGridDescTuple::Size() &&
                      NumOutput == OutGridDescTuple::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    __device__ static void Run(const InGridDescTuple& in_grid_desc_tuple,
                               const OutGridDescTuple& out_grid_desc_tuple,
                               const InDataTypePointerTuple& p_in_global_tuple,
                               const OutDataTypePointerTuple& p_out_global_tuple,
                               const Block2TileMap& block_2_tile_map,
                               const ElementwiseOperation& elementwise_op)
    {

        constexpr auto src_datas = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return     DataType{};
            },
            Number<NumInput>{});

        constexpr auto dst_datas = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return     DataType{};

            },
            Number<NumOutput>{});
        
        const auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto out_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_global_tuple[I], out_grid_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumOutput>{});

        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t m0_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * M0PerBlock);
        const index_t m1_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * M1PerBlock);
        const auto thread_grid_offset = make_multi_index(m0_block_data_idx_on_grid, m1_block_data_idx_on_grid);

        using ThisThreadBlock = ThisThreadBlock<BlockSize>;
        // If src and dst have same vector dim, then:
        //     M0 dim - for src and dst vector load/store
        // else:
        //     M0 dim - for src vector load
        //     M1 dim - for dst vector store
        using SrcDimAccessOrder = Sequence<1, 0>;
        using DstDimAccessOrder = std::conditional_t<InOutSameVectorDim, Sequence<1, 0>, Sequence<0, 1>>;
        using SrcVectorDim = Number<0>;
        using DstVectorDim = std::conditional_t<InOutSameVectorDim, Number<0>, Number<1>>;

        auto global_to_global_transfer = ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                           ElementwiseOperation,
                                           ElementwiseOperation,
                                           InMemoryDataOperationEnum::Set,
                                           Sequence<M0PerBlock, M1PerBlock>,
                                           Sequence<Number<M0PerBlock / M0PerThread>{}, Number<M1PerBlock / M1PerThread>{}>,
                                           ThreadClusterArrangeOrder,
                                        remove_cvref_t<decltype(src_datas.At(I0))>,
                                           remove_cvref_t<decltype(dst_datas.At(I0))>,
                                        decltype(in_grid_desc_tuple.At(I0)),
                                        decltype(out_grid_desc_tuple.At(I0)),
                                           SrcDimAccessOrder,
                                           DstDimAccessOrder,
                                           SrcVectorDim{},
                                           DstVectorDim{},
                                           InScalarPerVectorSeq::At(I0),
                                           OutScalarPerVectorSeq::At(I0),
                                            I1,
                                            I1,
                                            false,
                                            false,
                                            I1> {
                                                    in_grid_desc_tuple.At(I0),
                                                    thread_grid_offset,
                                                    elementwise_op,
                                                    out_grid_desc_tuple.At(I0),
                                                    thread_grid_offset,
                                                    elementwise_op
                                           };
        global_to_global_transfer.RunRead(in_grid_desc_tuple.At(I0), in_global_buf_tuple.At(I0),  I0);
        global_to_global_transfer.RunWrite(out_grid_desc_tuple.At(I0), out_global_buf_tuple.At(I0), I0);
    }
};

} // namespace ck
