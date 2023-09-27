// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename InputGridDesc,
          typename InputDataType,
          typename OutputGridDesc,
          typename OutputDataType,
          typename Block2ETileMap,
          typename GridwiseTensorRearrangeKernel>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_tensor_rearrange(const InputGridDesc in_grid_desc,
                                const InputDataType* __restrict__ p_in_global,
                                const OutputGridDesc out_grid_desc,
                                OutputDataType* __restrict__ p_out_global,
                                const Block2ETileMap block_2_tile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__) || defined(__gfx908__) ||             \
    defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx1030__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx941__) || defined(__gfx942__))
    GridwiseTensorRearrangeKernel::Run(
        in_grid_desc, p_in_global, out_grid_desc, p_out_global, block_2_tile_map);
#else
    ignore = in_grid_desc;
    ignore = p_in_global;
    ignore = out_grid_desc;
    ignore = p_out_global;
    ignore = block_2_tile_map;
#endif
}

template <typename InputGridDesc,
          typename InputDataType,
          typename OutputGridDesc,
          typename OutputDataType,
          index_t BlockSize,
          index_t MPerBlock,
          index_t KPerBlock,
          typename ThreadClusterLengths,
          index_t ScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          typename Block2ETileMap>
struct GridwiseTensorRearrange
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    __device__ static void Run(const InputGridDesc& in_grid_desc,
                               const InputDataType* __restrict__ p_in_global,
                               const OutputGridDesc& out_grid_desc,
                               OutputDataType* __restrict__ p_out_global,
                               const Block2ETileMap& block_2_tile_map)
    {
        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t k_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * KPerBlock);

        // Global Memory
        const auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_desc.GetElementSpaceSize());
        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, out_grid_desc.GetElementSpaceSize());

        auto copy_global_to_global =
            ThreadGroupTensorSliceTransfer_v7<ThisThreadBlock,
                                              Tuple<InputDataType>,
                                              Tuple<OutputDataType>,
                                              decltype(tie(in_grid_desc)),
                                              decltype(tie(out_grid_desc)),
                                              tensor_operation::element_wise::PassThrough,
                                              Sequence<static_cast<index_t>(DstInMemOp)>,
                                              Sequence<MPerBlock, KPerBlock>,
                                              ThreadClusterLengths,
                                              Sequence<0, 1>,
                                              Sequence<0, 1>,
                                              I1,
                                              ScalarPerVector,
                                              Sequence<true>,
                                              Sequence<true>>{
                in_grid_desc,
                make_tuple(make_multi_index(m_block_data_idx_on_grid, k_block_data_idx_on_grid)),
                out_grid_desc,
                make_tuple(make_multi_index(m_block_data_idx_on_grid, k_block_data_idx_on_grid)),
                tensor_operation::element_wise::PassThrough{}};

        copy_global_to_global.Run(
            tie(in_grid_desc), tie(in_global_buf), tie(out_grid_desc), tie(out_global_buf));
    }

    __host__ static constexpr bool CheckValidity(const InputGridDesc& in_grid_desc,
                                                 const OutputGridDesc& out_grid_desc)
    {
        if(in_grid_desc.GetLength(I0) % MPerBlock != 0 ||
           in_grid_desc.GetLength(I1) % KPerBlock != 0)
            return false;
        if(out_grid_desc.GetLength(I0) % MPerBlock != 0 ||
           out_grid_desc.GetLength(I1) % KPerBlock != 0)
            return false;
        return true;
    }
};

} // namespace ck
