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
    static_assert(NumDim == 2);
    static_assert(NumDim <= GridDescriptor::GetNumOfDimension());

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
            std::array<index_t, 2> num_tiles_per_axis;
            static_for<NumDim - 2, NumDim, 1>{}([&](auto I) {
                num_tiles_per_axis[I - (NumDim - 2)] =
                    math::integer_divide_ceil(desc.GetLength(I), TileDims::At(I - (NumDim - 2)));
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

        std::array<index_t, 2> num_tiles_per_axis;
        static_for<NumDim - 2, NumDim, 1>{}([&](auto I) {
            num_tiles_per_axis[I - (NumDim - 2)] =
                math::integer_divide_ceil(desc_.GetLength(I), TileDims::At(I - (NumDim - 2)));
        });

        std::array<index_t, 2> divisors;
        index_t product = 1;
        auto divisor    = rbegin(divisors);
        for(auto num_tiles = rbegin(num_tiles_per_axis); num_tiles != rend(num_tiles_per_axis);
            ++num_tiles)
        {
            product *= (*num_tiles);
            *(divisor++) = product;
        }

        const index_t grid_size = divisors.front();
        block_1d_id             = block_1d_id % grid_size; // swallow batch index

        return generate_tuple(
            [&](auto I) {
                return (block_1d_id % divisors[I]) / (divisors[I] / num_tiles_per_axis[I]);
            },
            Number<2>{});
    }

    private:
    const GridDescriptor desc_;
};
} // namespace detail

template <typename GridwisePermute,
          typename InGridDesc,
          typename OutGridDesc,
          typename InDataTypePointer,
          typename OutDataTypePointer,
          typename ElementwiseOperation,
          typename Block2TileMap>
__global__ void kernel_nd_permute(const InGridDesc in_grid_desc,
                                  const OutGridDesc out_grid_desc,
                                  const InDataTypePointer p_in_global,
                                  const OutDataTypePointer p_out_global,
                                  const ElementwiseOperation elementwise_op,
                                  const Block2TileMap block_2_tile_map)
{
    __shared__ char p_shared[GridwisePermute::GetSharedMemoryNumberOfByte()];

    GridwisePermute::Run(in_grid_desc,
                         out_grid_desc,
                         p_in_global,
                         p_out_global,
                         p_shared,
                         elementwise_op,
                         block_2_tile_map);
}

template <typename InGridDesc,
          typename OutGridDesc,
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
struct GridwisePermute
{
    static_assert(InGridDesc::GetNumOfDimension() == 3 && OutGridDesc::GetNumOfDimension() == 3);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using DefaultBlock2TileMap = detail::Block2TileMap<Sequence<HPerBlock, WPerBlock>, InGridDesc>;

    __host__ __device__ static constexpr auto GetInBlockDescriptor()
    {
        constexpr index_t ABlockLdsExtraM = 0;

        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(make_tuple(1, Number<HPerBlock>{}, Number<WPerBlock>{}),
                                            make_tuple(Number<WPerBlock + ABlockLdsExtraM>{},
                                                       Number<WPerBlock + ABlockLdsExtraM>{},
                                                       I1));
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetInBlockDescriptor();

        using InDataType = remove_cv_t<remove_pointer_t<InDataTypePointer>>;

        // lds max alignment
        constexpr auto max_lds_align = 1;

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        return a_block_space_size_aligned * sizeof(InDataType);
    }

    __host__ __device__ static constexpr auto MakeDefaultBlock2TileMap(const InGridDesc& desc)
    {
        return DefaultBlock2TileMap{desc};
    }

    template <typename Block2TileMap>
    __device__ static void Run(const InGridDesc in_grid_desc,
                               const OutGridDesc out_grid_desc,
                               const InDataTypePointer p_in_global,
                               const OutDataTypePointer p_out_global,
                               void* __restrict__ p_shared,
                               const ElementwiseOperation elementwise_op,
                               const Block2TileMap& block_2_tile_map)
    {
        using InDataType  = remove_cv_t<remove_pointer_t<InDataTypePointer>>;
        using OutDataType = remove_cv_t<remove_pointer_t<OutDataTypePointer>>;

        auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_desc.GetElementSpaceSize());

        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, out_grid_desc.GetElementSpaceSize());

        const auto loop_step_index = make_multi_index(1, 0, 0);

        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * HPerBlock);

        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * WPerBlock);

        // Input slice in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetInBlockDescriptor();

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<InDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        using SliceLengths                                 = Sequence<1, HPerBlock, WPerBlock>;
        using ABlockTransferThreadClusterLengths           = Sequence<1, 16, BlockSize / 16>;
        using ABlockTransferThreadClusterArrangeOrder      = Sequence<0, 1, 2>;
        using ABlockTransferSrcAccessOrder                 = Sequence<0, 1, 2>;
        using ABlockTransferDstAccessOrder                 = Sequence<0, 1, 2>;
        constexpr index_t ABlockTransferSrcVectorDim       = 2;
        constexpr index_t ABlockTransferDstVectorDim       = 2;
        constexpr index_t ABlockTransferSrcScalarPerVector = 1;
        constexpr index_t ABlockTransferDstScalarPerVector = 1;

        auto in_global_load =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                ElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                SliceLengths,
                                                ABlockTransferThreadClusterLengths,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InDataType,
                                                InDataType,
                                                decltype(in_grid_desc),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                ABlockTransferDstAccessOrder,
                                                ABlockTransferSrcVectorDim,
                                                ABlockTransferDstVectorDim,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector,
                                                1,
                                                1,
                                                true,
                                                true>(
                in_grid_desc,
                make_multi_index(0, h_block_data_idx_on_grid, w_block_data_idx_on_grid),
                ck::tensor_operation::element_wise::PassThrough{},
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        auto in_grid_desc_tranformed = transform_tensor_descriptor(
            in_grid_desc,
            make_tuple(make_pass_through_transform(in_grid_desc.GetLength(I0)),
                       make_pass_through_transform(in_grid_desc.GetLength(I1)),
                       make_pass_through_transform(in_grid_desc.GetLength(I2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}));

        auto out_global_store = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            ElementwiseOperation,
            ck::tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            Sequence<1, HPerBlock, WPerBlock>, // SliceLengths
            ABlockTransferThreadClusterLengths,
            Sequence<0, 1, 2>, // ABlockTransferThreadClusterArrangeOrder
            InDataType,
            OutDataType,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(in_grid_desc_tranformed),
            Sequence<0, 1, 2>, // ABlockTransferSrcAccessOrder
            Sequence<0, 1, 2>, // ABlockTransferDstAccessOrder
            1,                 // ABlockTransferSrcVectorDim
            1,                 // ABlockTransferDstVectorDim
            1,                 // ABlockTransferSrcScalarPerVector
            1,                 // ABlockTransferDstScalarPerVector
            1,
            1,
            true,
            true>(a_block_desc_ak0_m_ak1,
                  make_multi_index(0, 0, 0),
                  ck::tensor_operation::element_wise::PassThrough{},
                  in_grid_desc_tranformed,
                  make_multi_index(0, h_block_data_idx_on_grid, w_block_data_idx_on_grid),
                  elementwise_op);

        index_t num_iter = in_grid_desc.GetLength(I0);
        do
        {
            in_global_load.Run(
                in_grid_desc, in_global_buf, a_block_desc_ak0_m_ak1, a_block_buf, I0);

            in_global_load.MoveSrcSliceWindow(in_grid_desc, loop_step_index);

            out_global_store.Run(
                a_block_desc_ak0_m_ak1, a_block_buf, in_grid_desc_tranformed, out_global_buf, I0);

            out_global_store.MoveDstSliceWindow(in_grid_desc_tranformed, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
