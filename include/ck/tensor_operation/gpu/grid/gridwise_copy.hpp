// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <numeric>
#include <iterator>

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace detail {
template <typename TileDims, typename GridDescriptor>
struct Block2TileMap
{
    static constexpr auto I0 = Number<0>{};

    static constexpr index_t NumDim = TileDims::Size();
    static_assert(NumDim == GridDescriptor::GetNumOfDimension());

    Block2TileMap()                     = delete;
    Block2TileMap(const Block2TileMap&) = default;
    Block2TileMap(Block2TileMap&&)      = delete;

    ~Block2TileMap() = default;

    Block2TileMap& operator=(const Block2TileMap&) = delete;
    Block2TileMap& operator=(Block2TileMap&&) = delete;

    explicit Block2TileMap(const GridDescriptor& desc) : desc_(desc) {}

    __host__ constexpr index_t CalculateGridSize(const GridDescriptor& desc) const
    {
        return [&]() {
            std::array<index_t, NumDim> num_tiles_per_axis;
            static_for<0, NumDim, 1>{}([&](auto I) {
                num_tiles_per_axis[I] =
                    math::integer_divide_ceil(desc.GetLength(I), TileDims::At(I));
            });

            return std::accumulate(begin(num_tiles_per_axis),
                                   end(num_tiles_per_axis),
                                   index_t{1},
                                   std::multiplies<index_t>{});
        }();
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == 1);

        auto block_1d_id = idx_top[I0];

        std::array<index_t, NumDim> num_tiles_per_axis;
        static_for<0, NumDim, 1>{}([&](auto I) {
            num_tiles_per_axis[I] = math::integer_divide_ceil(desc_.GetLength(I), TileDims::At(I));
        });

        std::array<index_t, NumDim> divisors;
        std::partial_sum(rbegin(num_tiles_per_axis),
                         rend(num_tiles_per_axis),
                         rbegin(divisors),
                         std::multiplies<index_t>{});

        const index_t grid_size = divisors.front();
        block_1d_id             = block_1d_id % grid_size; // swallow batch index

        return generate_tuple(
            [&](auto I) {
                return (block_1d_id % divisors[I]) / (divisors[I] / num_tiles_per_axis[I]);
            },
            Number<NumDim>{});
    }

    private:
    const GridDescriptor desc_;
};
} // namespace detail

template <typename GridwiseCopyFunctor,
          typename InGrid1dDesc,
          typename OutGrid1dDesc,
          typename InDataTypePointer,
          typename OutDataTypePointer,
          typename ElementwiseOperation,
          typename Block2TileMap>
__global__ void kernel_nd_copy(const InGrid1dDesc in_grid_1d_desc,
                               const OutGrid1dDesc out_grid_1d_desc,
                               const InDataTypePointer p_in_global,
                               const OutDataTypePointer p_out_global,
                               const ElementwiseOperation elementwise_op,
                               const Block2TileMap block_2_tile_map)
{
    GridwiseCopyFunctor::Run(in_grid_1d_desc,
                             out_grid_1d_desc,
                             p_in_global,
                             p_out_global,
                             elementwise_op,
                             block_2_tile_map);
}

template <typename InGrid1dDesc,
          typename OutGrid1dDesc,
          typename InDataTypePointer,
          typename OutDataTypePointer,
          typename ElementwiseOperation,
          index_t BlockSize,
          index_t NPerBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t MPerThread,
          index_t InScalarPerVector,
          index_t OutScalarPerVector>
struct GridwiseCopy
{
    static_assert(InGrid1dDesc::GetNumOfDimension() == 3 &&
                  OutGrid1dDesc::GetNumOfDimension() == 3);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using DefaultBlock2TileMap =
        detail::Block2TileMap<Sequence<NPerBlock, HPerBlock, WPerBlock>, InGrid1dDesc>;

    __host__ __device__ static constexpr auto GetInBlockDescriptor()
    {
        constexpr index_t ABlockLdsExtraM = 0;

        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(Number<NPerBlock>{}, Number<HPerBlock>{}, Number<WPerBlock>{}),
            make_tuple(Number<NPerBlock + ABlockLdsExtraM>{} * Number<HPerBlock>{},
                       Number<HPerBlock>{},
                       I1));
    }

    __host__ __device__ static constexpr auto MakeDefaultBlock2TileMap(const InGrid1dDesc& desc)
    {
        return DefaultBlock2TileMap{desc};
    }

    template <typename Block2TileMap>
    __device__ static void Run(const InGrid1dDesc in_grid_1d_desc,
                               const OutGrid1dDesc out_grid_1d_desc,
                               const InDataTypePointer p_in_global,
                               const OutDataTypePointer p_out_global,
                               const ElementwiseOperation elementwise_op,
                               const Block2TileMap& block_2_tile_map)
    {
        const index_t thread_global_id = get_thread_global_1d_id();

        using InDataType   = remove_cv_t<remove_pointer_t<InDataTypePointer>>;
        auto in_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr, InDataType, MPerThread, true>{};

        using OutDataType   = remove_cv_t<remove_pointer_t<OutDataTypePointer>>;
        auto out_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MPerThread, true>{};

        auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_1d_desc.GetElementSpaceSize());

        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, out_grid_1d_desc.GetElementSpaceSize());

        const auto thread_global_offset = make_multi_index(thread_global_id * MPerThread);

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto M               = in_grid_1d_desc.GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * MPerThread;
        const auto loop_step_index = make_multi_index(loop_step);

#if 0
        auto in_global_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             InDataType,
                                             decltype(in_grid_1d_desc),
                                             decltype(thread_buffer_desc_m),
                                             Sequence<MPerThread>, // SliceLengths
                                             Sequence<0>,          // DimAccessOrder
                                             0,                    // SrcVectorDim
                                             InScalarPerVector,    // ScalarPerVector
                                             1,                    // SrcScalarStrideInVector
                                             false>{in_grid_1d_desc, thread_global_offset};
#else
        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * NPerBlock * HPerBlock);

        // const index_t n_block_data_idx_on_grid =
        //     __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetInBlockDescriptor();

        // // B matrix in LDS memory, dst of blockwise copy
        // constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        using SliceLengths = Sequence<NPerBlock, HPerBlock, WPerBlock>;
        using ABlockTransferThreadClusterLengths_AK0_M_AK1 = Sequence<4, 64, 1>;
        using ABlockTransferThreadClusterArrangeOrder      = Sequence<1, 0, 2>;
        using ABlockTransferSrcAccessOrder                 = Sequence<1, 0, 2>;
        constexpr index_t ABlockTransferSrcVectorDim       = 2;
        constexpr index_t ABlockTransferSrcScalarPerVector = 1;
        constexpr index_t ABlockTransferDstScalarPerVector = 1;

        auto in_global_load =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                ElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                SliceLengths,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InDataType,
                                                InDataType,
                                                decltype(in_grid_1d_desc),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector,
                                                1,
                                                1,
                                                true,
                                                true>(
                in_grid_1d_desc,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                elementwise_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#endif
        auto out_global_store =
            ThreadwiseTensorSliceTransfer_v1r3<OutDataType,
                                               OutDataType,
                                               decltype(thread_buffer_desc_m),
                                               decltype(out_grid_1d_desc),
                                               PassThroughOp,
                                               SliceLengths,      // SliceLengths
                                               Sequence<1, 0, 2>, // DimAccessOrder
                                               0,                 // SrcVectorDim
                                               OutScalarPerVector,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               false>(
                out_grid_1d_desc, thread_global_offset, PassThroughOp{});

        index_t num_iter = M / (loop_step);
        do
        {
            in_global_load.Run(in_grid_1d_desc,
                               in_global_buf,
                               thread_buffer_desc_m,
                               make_tuple(I0),
                               in_thread_buf);

            in_global_load.MoveSrcSliceWindow(in_grid_1d_desc, loop_step_index);

            static_for<0, MPerThread, 1>{}([&](auto iM) {
                // get reference to in data
                const auto& in_data_ref = in_thread_buf(iM);

                // get reference to dst data
                auto& out_data_ref = out_thread_buf(iM);

                elementwise_op(out_data_ref, in_data_ref);
            });

            out_global_store.Run(thread_buffer_desc_m,
                                 make_tuple(I0),
                                 out_thread_buf,
                                 out_grid_1d_desc,
                                 out_global_buf);

            out_global_store.MoveDstSliceWindow(out_grid_1d_desc, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
