// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "tensor_utils.hpp"
#include "layout_utils.hpp"

#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace {

/**
 * \brief Calculate shape for partition based on number of threads per each dim and
 * previous shape
 *
 * \param shape Base tensor shape.
 * \param thread_lengths Tuple of thread lengths.
 * \return Partition shape.
 */
template <typename... Ts, typename... Ls>
__host__ __device__ constexpr auto CalculateLocalPartitionShape(const Tuple<Ts...>& shape,
                                                                const Tuple<Ls...>& thread_lengths)
{
    static_assert(Tuple<Ts...>::Size() == Tuple<Ls...>::Size(), "Wrong thread_lengths shape.");
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            const auto slice_len = size<num_i>(shape) / thread_lengths.At(num_i);
            return slice_len;
        },
        Number<Tuple<Ls...>::Size()>{});
}

/**
 * \brief Calculate total number of blocks.
 *
 * \param shape Base tensor shape.
 * \param tile_shape Tile shape.
 * \return Tuple with blocks number.
 */
template <typename... Ts, typename... Ls>
__host__ __device__ constexpr auto CalculateGridSize(const Tuple<Ts...>& shape,
                                                     const Tuple<Ls...>& tile_shape)
{
    static_assert(Tuple<Ts...>::Size() == Tuple<Ls...>::Size(), "Wrong thread_lengths shape.");
    return generate_tuple([&](auto i) { return size<i>(shape) / size<i>(tile_shape); },
                          Number<Tuple<Ls...>::Size()>{});
}

/**
 * \brief Calculate scaled offset for new partition/tile.
 *
 * \param thread_idxs Thread 1d id.
 * \param partition_lengths_seq Sequence of partition shape.
 * \param old_offset_idxs Multi index offset from base tensor to shift values.
 * \return Partition shape.
 */
template <typename ThreadIdxs, typename PartitionLengthsSeq, typename OldOffsetIdxs>
__host__ __device__ constexpr auto
CalculateOffsetMultiIdxs(const ThreadIdxs& thread_idxs,
                         const PartitionLengthsSeq& partition_lengths_seq,
                         const OldOffsetIdxs& old_offset_idxs)
{
    return thread_idxs * partition_lengths_seq + old_offset_idxs;
}

} // namespace
/// @endcond

/**
 * \brief Create local partition for thread (At now only packed partition
 * is supported).
 *
 * \param tensor Tensor for partition.
 * \param thread_lengths Layout of threads (could not be nested).
 * \param thread_id Thread index represented as integer.
 * \return Partition tensor.
 */
template <typename TensorType, typename ThreadLengthsTuple>
__host__ __device__ constexpr auto
make_local_partition(TensorType& tensor,
                     [[maybe_unused]] const ThreadLengthsTuple& thread_lengths,
                     const index_t thread_id)
{
    static_assert(!IsNestedTuple(ThreadLengthsTuple{}));
    // Calculate new partition shape
    const auto& tensor_shape = shape(tensor);
    constexpr auto partition_shape =
        CalculateLocalPartitionShape(decltype(tensor_shape){}, ThreadLengthsTuple{});
    // Create Thread Cluster Descriptor
    constexpr auto partition_lengths_seq = generate_sequence_v2(
        [&](auto I) { return size<I>(partition_shape); }, Number<ThreadLengthsTuple::Size()>{});
    constexpr auto thread_lengths_seq =
        generate_sequence_v2([&](auto I) { return size<I>(ThreadLengthsTuple{}); },
                             Number<ThreadLengthsTuple::Size()>{});
    constexpr auto thread_cluster_desc_ = make_cluster_descriptor(thread_lengths_seq);
    // Calculate thread idxs and offsets
    const auto thread_idxs = thread_cluster_desc_.CalculateBottomIndex(make_multi_index(thread_id));
    const auto offset_multi_idxs =
        CalculateOffsetMultiIdxs(thread_idxs, partition_lengths_seq, tensor.GetMultiIdxOffsets());
    // Create new layout and tensor
    auto& flatten_desc = layout(tensor).GetUnrolledDescriptor();
    const auto partition_layout =
        Layout<remove_reference_t<decltype(partition_shape)>, decltype(flatten_desc)>(
            partition_shape, flatten_desc);
    auto partition_tensor =
        make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), partition_layout);
    // Apply offsets
    partition_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
    return partition_tensor;
}

/**
 * \brief Create local tile for thread block. (At now only packed tile
 * is supported).
 *
 * \note Temporary to gain the best performance use 2d
 * tile_shape.
 *
 *
 * \param tensor Tensor for partition.
 * \param tile_shape Shapes of requested tile.
 * \param block_id Block index represented as integer.

 * \return Tile tensor.
 */
template <typename TensorType, typename BlockShapeTuple>
__host__ __device__ constexpr auto
make_local_tile(const TensorType& tensor, const BlockShapeTuple& tile_shape, const index_t block_id)
{
    static_assert(!IsNestedTuple(BlockShapeTuple{}));

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    auto& aligned_desc = layout(tensor).GetMergedNestingDescriptor();

    if constexpr(BlockShapeTuple::Size() == I2)
    {
        // Optimized version for 2d tile shape [MxK]
        const auto block_2_tile_map =
            BlockToCTileMap_M00_N0_M01Adapt<BlockShapeTuple{}.At(I0),
                                            BlockShapeTuple{}.At(I1),
                                            remove_cvref_t<decltype(aligned_desc)>>(aligned_desc);
        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(block_id));
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * size<0>(tile_shape));
        const index_t k_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * size<1>(tile_shape));
        const auto offset_multi_idxs =
            make_tuple(m_block_data_idx_on_grid, k_block_data_idx_on_grid);
        // Create new layout and tensor
        const auto tile_layout =
            Layout<remove_reference_t<decltype(tile_shape)>, decltype(aligned_desc)>(tile_shape,
                                                                                     aligned_desc);
        auto tile_tensor =
            make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), tile_layout);
        // Apply offsets
        tile_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
        return tile_tensor;
    }
    else
    {
        // Calculate offsets
        // Sequence with data to process per block
        constexpr auto tile_shape_seq =
            generate_sequence_v2([](auto I) { return size(BlockShapeTuple{}.At(I)); },
                                 Number<BlockShapeTuple::Size()>{});
        // Tuple with number of blocks
        const auto block_lengths           = CalculateGridSize(shape(tensor), tile_shape);
        constexpr auto block_cluster_desc_ = make_cluster_descriptor(block_lengths);
        const auto block_idxs =
            block_cluster_desc_.CalculateBottomIndex(make_multi_index(block_id));
        const auto offset_multi_idxs =
            CalculateOffsetMultiIdxs(block_idxs, tile_shape_seq, tensor.GetMultiIdxOffsets());
        // Create new layout and tensor
        const auto tile_layout =
            Layout<remove_reference_t<decltype(tile_shape)>, decltype(aligned_desc)>(tile_shape,
                                                                                     aligned_desc);
        auto tile_tensor =
            make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), tile_layout);
        // Apply offsets
        tile_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
        return tile_tensor;
    }
}

} // namespace wrapper
} // namespace ck
