// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "tensor_utils.hpp"
#include "layout_utils.hpp"

#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"

namespace ck {
namespace wrapper {

namespace {

namespace detail {

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
            const auto slice_len =
                ck::math::integer_divide_ceil(size<num_i>(shape), thread_lengths.At(num_i));
            return slice_len;
        },
        Number<Tuple<Ls...>::Size()>{});
}

/**
 * \brief Apply projection.
 *
 * \param base_tuple Tuple to apply projection.
 * \param projection Projection to remove selected dim from partitioning.
 * slice(X) to remove, where X is dim size, Number<1>{} to keep.
 * \return Multi index after projection.
 */
template <typename MultiIndex, typename ProjectionTuple>
__host__ __device__ constexpr auto
ApplyProjection([[maybe_unused]] const MultiIndex& base_tuple,
                [[maybe_unused]] const ProjectionTuple& projection)
{
    if constexpr(is_same_v<ProjectionTuple, Tuple<>>)
    {
        return Tuple<>{};
    }
    else
    {
        auto base_tuple_after_projection = generate_tuple(
            [&](auto i) {
                const auto i_num = Number<i.value>{};
                static_assert(
                    is_detected<is_slice, tuple_element_t<i_num, ProjectionTuple>>::value ||
                    is_same_v<tuple_element_t<i_num, ProjectionTuple>, Number<1>>);
                if constexpr(is_detected<is_slice, tuple_element_t<i_num, ProjectionTuple>>::value)
                {
                    // When slice (to remove), then insert empty tuple (will be removed in next
                    // step).
                    return Tuple<>{};
                }
                else
                {
                    return base_tuple.At(i_num);
                }
            },
            Number<MultiIndex::Size()>{});
        // Remove empty tuples
        return UnrollNestedTuple<0, 1>(base_tuple_after_projection);
    }
}

/**
 * \brief Calculate shape with dims from projection.
 *
 * \param shape Base tensor shape.
 * \param projection Projection to remove selected dim from partitioning.
 * slice(X) to remove, where X is dim size, Number<1>{} to keep.
 * \return Shape with dims from projection
 */
template <typename... Ts, typename... Ps>
__host__ __device__ constexpr auto CalculateShapeWithProjection(const Tuple<Ts...>& shape,
                                                                const Tuple<Ps...>& projection)
{
    return generate_tuple(
        [&](auto i) {
            if constexpr(is_detected<is_slice, tuple_element_t<i, Tuple<Ps...>>>::value)
            {
                return size<i>(projection).to_;
            }
            else
            {
                // number of shape element in actual fragment of shape and projection (method to
                // calculate shape idx)
                constexpr index_t shape_i =
                    detail::ApplyProjection(TupleSlice<0, i>(Tuple<Ts...>{}),
                                            TupleSlice<0, i>(Tuple<Ps...>{}))
                        .Size();
                return size<shape_i>(shape);
            }
        },
        Number<Tuple<Ps...>::Size()>{});
}

/**
 * \brief Calculate total number of blocks.
 *
 * \param shape Base tensor shape.
 * \param tile_shape Tile shape.
 * \param projection Projection to remove selected dim from partitioning.
 * slice(X) to remove, where X is dim size, Number<1>{} to keep.
 * \return Tuple with blocks number.
 */
template <typename... Ts, typename... Ls, typename... Ps>
__host__ __device__ constexpr auto CalculateGridSize(const Tuple<Ts...>& shape,
                                                     const Tuple<Ls...>& tile_shape,
                                                     const Tuple<Ps...>& projection)
{
    auto shape_with_projection = CalculateShapeWithProjection(shape, projection);
    return generate_tuple(
        [&](auto i) {
            return ck::math::integer_divide_ceil(size<i>(shape_with_projection),
                                                 size<i>(tile_shape));
        },
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

/**
 * \brief Calculate default projection.
 *
 * \param tile_shape Tile shape.
 * \return Default projection (filled with Number<1>{}).
 */
template <typename TileShape>
__host__ __device__ constexpr auto
GenerateDefaultProjection([[maybe_unused]] const TileShape tile_shape)
{
    return generate_tuple([&](auto) { return Number<1>{}; }, Number<TileShape::Size()>{});
}

} // namespace detail
} // namespace

/**
 * \brief Create local partition for thread (At now only packed partition
 * is supported).
 *
 * \param tensor Tensor for partition.
 * \param thread_lengths Layout of threads (could not be nested).
 * \param thread_id Thread index represented as integer.
 * \param projection Projection to remove selected dim from partitioning.
 * slice(X) to remove, where X is dim size, Number<1>{} to keep.
 * \return Partition tensor.
 */
template <typename TensorType, typename ThreadLengthsTuple, typename ProjectionTuple>
__host__ __device__ constexpr auto
make_local_partition(TensorType& tensor,
                     [[maybe_unused]] const ThreadLengthsTuple& thread_lengths,
                     const index_t thread_id,
                     const ProjectionTuple& projection)
{
    static_assert(!IsNestedTuple(ThreadLengthsTuple{}));
    // Calculate new partition shape
    const auto& tensor_shape = shape(tensor);
    // Calculate projected thread lengths
    constexpr auto projected_thread_lengths =
        detail::ApplyProjection(ThreadLengthsTuple{}, ProjectionTuple{});
    constexpr auto partition_shape =
        detail::CalculateLocalPartitionShape(decltype(tensor_shape){}, projected_thread_lengths);
    // Create Thread Cluster Descriptor
    constexpr auto partition_shape_seq =
        generate_sequence_v2([&](auto I) { return size<I>(partition_shape); },
                             Number<decltype(partition_shape)::Size()>{});
    constexpr auto thread_lengths_seq =
        generate_sequence_v2([&](auto I) { return size<I>(ThreadLengthsTuple{}); },
                             Number<ThreadLengthsTuple::Size()>{});
    constexpr auto thread_cluster_desc_ = make_cluster_descriptor(thread_lengths_seq);
    // Calculate thread idxs and offsets
    const auto thread_idxs = thread_cluster_desc_.CalculateBottomIndex(make_multi_index(thread_id));
    // Apply projection on thread idxs to remove not needed idxs
    const auto projected_thread_idxs = detail::ApplyProjection(thread_idxs, projection);
    const auto offset_multi_idxs     = detail::CalculateOffsetMultiIdxs(
        projected_thread_idxs, partition_shape_seq, tensor.GetMultiIdxOffsets());
    // Create new layout and tensor
    auto& unrolled_desc = layout(tensor).GetUnrolledDescriptor();
    const auto partition_layout =
        Layout<remove_reference_t<decltype(partition_shape)>, decltype(unrolled_desc)>(
            partition_shape, unrolled_desc);
    auto partition_tensor =
        make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), partition_layout);
    // Apply offsets
    partition_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
    return partition_tensor;
}

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
__host__ __device__ constexpr auto make_local_partition(TensorType& tensor,
                                                        const ThreadLengthsTuple& thread_lengths,
                                                        const index_t thread_id)
{
    const auto projection = detail::GenerateDefaultProjection(ThreadLengthsTuple{});
    return make_local_partition(tensor, thread_lengths, thread_id, projection);
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
 * \param projection Projection to remove selected dim from partitioning.
 * slice(X) to remove, where X is dim size, Number<1>{} to keep.
 * \return Tile tensor.
 */
template <typename TensorType, typename BlockShapeTuple, typename ProjectionTuple>
__host__ __device__ constexpr auto make_local_tile(const TensorType& tensor,
                                                   const BlockShapeTuple& tile_shape,
                                                   const index_t block_id,
                                                   const ProjectionTuple& projection)
{
    static_assert(!IsNestedTuple(BlockShapeTuple{}));

    constexpr bool is_default_projection =
        is_same_v<ProjectionTuple, decltype(detail::GenerateDefaultProjection(BlockShapeTuple{}))>;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    auto& aligned_desc = layout(tensor).GetMergedNestingDescriptor();

    // TODO: Enable block_2_tile_map partitioning for non-default projection.
    if constexpr(BlockShapeTuple::Size() == I2 && is_default_projection)
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
        constexpr auto projected_tile_shape =
            detail::ApplyProjection(BlockShapeTuple{}, ProjectionTuple{});
        using ProjectedTileShapeTuple = decltype(projected_tile_shape);
        constexpr auto projected_tile_shape_seq =
            generate_sequence_v2([](auto I) { return ProjectedTileShapeTuple{}.At(I); },
                                 Number<ProjectedTileShapeTuple::Size()>{});
        // Tuple with number of blocks
        const auto block_lengths = detail::CalculateGridSize(shape(tensor), tile_shape, projection);
        const auto block_cluster_desc_ = make_cluster_descriptor(block_lengths);
        const auto block_idxs =
            block_cluster_desc_.CalculateBottomIndex(make_multi_index(block_id));
        const auto projected_block_idxs = detail::ApplyProjection(block_idxs, projection);
        const auto offset_multi_idxs    = detail::CalculateOffsetMultiIdxs(
            projected_block_idxs, projected_tile_shape_seq, tensor.GetMultiIdxOffsets());
        // Create new layout and tensor
        const auto tile_layout =
            Layout<remove_reference_t<ProjectedTileShapeTuple>, decltype(aligned_desc)>(
                projected_tile_shape, aligned_desc);
        auto tile_tensor =
            make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), tile_layout);
        // Apply offsets
        tile_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
        return tile_tensor;
    }
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
    const auto projection = detail::GenerateDefaultProjection(BlockShapeTuple{});
    return make_local_tile(tensor, tile_shape, block_id, projection);
}

/**
 * \brief Pad tensor shapes to be adjusted to tile lengths.
 *
 *
 * \param tensor Tensor to pad.
 * \param tile_lengths Tile lengths to align tensor shape.
 * \return Padded tensor.
 */
template <typename TensorType, typename TileLengths>
__host__ __device__ constexpr auto pad(const TensorType& tensor, const TileLengths& tile_lengths)
{
    const auto& tensor_shape = shape(tensor);
    using TensorShapeType    = remove_reference_t<decltype(tensor_shape)>;
    auto& unrolled_desc      = layout(tensor).GetUnrolledDescriptor();
    // Generate sequence with ones to mark that all dims will be padded
    constexpr auto do_pads_seq =
        generate_sequence_v2([](auto) { return Number<1>{}; }, Number<TensorShapeType::Size()>{});
    // Create descriptor with padding
    auto padded_desc =
        tensor_operation::device::PadTensorDescriptor(unrolled_desc, tile_lengths, do_pads_seq);
    // Generate padded shape
    const auto padded_shape = generate_tuple(
        [&](auto i) {
            const auto& dim         = size<i>(tensor_shape);
            const auto& tile_length = size<i>(tile_lengths);
            return ck::math::integer_divide_ceil(dim, tile_length) * tile_length;
        },
        Number<TileLengths::Size()>{});
    // Create layout and tensor
    const auto padded_layout =
        Layout<decltype(padded_shape), decltype(padded_desc)>(padded_shape, padded_desc);
    auto partition_tensor =
        make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), padded_layout);
    partition_tensor.SetMultiIdxOffset(tensor.GetMultiIdxOffsets());
    return partition_tensor;
}

} // namespace wrapper
} // namespace ck
