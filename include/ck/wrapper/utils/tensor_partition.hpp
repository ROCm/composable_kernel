// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "tensor_utils.hpp"
#include "layout_utils.hpp"

namespace ck {
namespace wrapper {

namespace {
// Calculate shape for partition based on number of threads per each dim and
// previous shape
template <typename... Ts, typename... Ls>
__host__ __device__ constexpr auto CalculateLocalPartitionShape(const Tuple<Ts...>& shape,
                                                                const Tuple<Ls...>& thread_lengths)
{
    static_assert(Tuple<Ts...>::Size() == Tuple<Ls...>::Size(), "Wrong thread_lengths shape.");
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {
                // if tuple then recurrence
                return CalculateLocalPartitionShape(shape.At(num_i), thread_lengths.At(num_i));
            }
            else
            {
                const auto slice_len = shape.At(num_i) / thread_lengths.At(num_i);
                return slice_len;
            }
        },
        Number<Tuple<Ts...>::Size()>{});
}

// Calculate shape for partition based on number of threads per each dim,
// previous strides and steps
template <typename... Ts, typename... Ls, typename... Steps, typename FlattenDescType>
__host__ __device__ constexpr auto
CalculateLocalPartitionDescriptor(const Tuple<Ts...>& shape,
                                  const Tuple<Ls...>& thread_lengths,
                                  const Tuple<Steps...>& steps,
                                  const FlattenDescType& flatten_desc)
{

    static_assert(Tuple<Ts...>::Size() == Tuple<Ls...>::Size(), "Wrong thread_lengths shape.");
    const auto unrolled_thread_lengths = UnrollNestedTuple(thread_lengths);
    const auto unrolled_shape          = UnrollNestedTuple(shape);
    constexpr auto dims                = decltype(unrolled_thread_lengths)::Size();

    using UnrolledStepsType = decltype(UnrollNestedTuple(steps));

    using I1 = Number<1>;

    const auto transforms = generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_same_v<Tuple<Steps...>, Tuple<>>)
            {
                // By default raked partition
                const auto partition_stride = unrolled_thread_lengths.At(num_i);
                return make_embed_transform(make_tuple(unrolled_shape.At(num_i)),
                                            make_tuple(partition_stride));
            }
            else if constexpr(!is_same_v<tuple_element_t<i.value, UnrolledStepsType>, index_t>)
            {
                // Compiletime partition
                if constexpr(is_same_v<tuple_element_t<i.value, UnrolledStepsType>, I1>)
                {
                    // raked
                    const auto partition_stride = unrolled_thread_lengths.At(num_i);
                    return make_embed_transform(make_tuple(unrolled_shape.At(num_i)),
                                                make_tuple(partition_stride));
                }
                else
                {
                    // packed
                    return make_embed_transform(make_tuple(unrolled_shape.At(num_i)),
                                                make_tuple(I1{}));
                }
            }
            else
            {
                // Runtime partition
                if(steps.At(num_i) == 1)
                {
                    // raked
                    const auto partition_stride = unrolled_thread_lengths.At(num_i);
                    return make_embed_transform(make_tuple(unrolled_shape.At(num_i)),
                                                make_tuple(partition_stride));
                }
                else
                {
                    // packed
                    return make_embed_transform(make_tuple(unrolled_shape.At(num_i)),
                                                make_tuple(I1{}));
                }
            }
        },
        Number<dims>{});

    const auto lower_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<dims>{});
    const auto upper_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<dims>{});
    return transform_tensor_descriptor(flatten_desc, transforms, lower_dims, upper_dims);
}

template <typename... Ls, typename... Steps>
__host__ __device__ constexpr auto CalculateLayoutOffsetIdxImpl(const Tuple<Ls...>& thread_lengths,
                                                                const Tuple<Steps...>& steps,
                                                                index_t& thread_id)
{
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ls...>>>::value)
            {
                // if tuple then recurrence
                if constexpr(is_same_v<Tuple<Steps...>, Tuple<>>)
                {
                    return CalculateLayoutOffsetIdxImpl(
                        thread_lengths.At(num_i), Tuple<>{}, thread_id);
                }
                else
                {
                    return CalculateLayoutOffsetIdxImpl(
                        thread_lengths.At(num_i), steps.At(num_i), thread_id);
                }
            }
            else
            {
                // Update thread_id after each dim
                const auto dim_thread_id = thread_id % thread_lengths.At(num_i);
                thread_id /= thread_lengths.At(num_i);
                if constexpr(is_same_v<Tuple<Steps...>, Tuple<>>)
                {
                    return dim_thread_id;
                }
                else
                {
                    // Apply step
                    return steps.At(num_i) * dim_thread_id;
                }
            }
        },
        Number<Tuple<Ls...>::Size()>{});
}

// Convert integer thread_idx to tuple index with steps applied
template <typename... Ls, typename... Steps>
__host__ __device__ constexpr auto CalculateLayoutOffsetIdx(const Tuple<Ls...>& thread_lengths,
                                                            const Tuple<Steps...>& steps,
                                                            const index_t thread_id)
{
    // Create tmp thread_id copy for CalculateLayoutOffsetIdxImpl updates
    index_t thread_id_copy = thread_id;
    return CalculateLayoutOffsetIdxImpl(thread_lengths, steps, thread_id_copy);
}

// Apply steps to index represented as tuple
template <typename... Steps, typename... Idxs>
__host__ __device__ constexpr auto CalculateLayoutOffsetIdx(const Tuple<Steps...>& steps,
                                                            const Tuple<Idxs...>& block_idxs)
{
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Idxs...>>>::value)
            {
                // if tuple then recurrence
                if constexpr(is_same_v<Tuple<Steps...>, Tuple<>>)
                {
                    return CalculateLayoutOffsetIdx(Tuple<>{}, block_idxs.At(num_i));
                }
                else
                {
                    return CalculateLayoutOffsetIdx(steps.At(num_i), block_idxs.At(num_i));
                }
            }
            else
            {
                if constexpr(is_same_v<Tuple<Steps...>, Tuple<>>)
                {
                    return block_idxs.At(num_i);
                }
                else
                {
                    // apply step
                    return steps.At(num_i) * block_idxs.At(num_i);
                }
            }
        },
        Number<Tuple<Idxs...>::Size()>{});
}

// User passes only shape per block to the make_local_tile function. This function calculates
// block layout based on the shape.
template <typename... Ts, typename... BlockDims>
__host__ __device__ constexpr auto CalculateBlockLengths(const Tuple<Ts...>& shape,
                                                         const Tuple<BlockDims...>& tile_shape)
{
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {
                // if tuple then recurrence
                return CalculateBlockLengths(shape.At(num_i), tile_shape.At(num_i));
            }
            else
            {
                return shape.At(num_i) / tile_shape.At(num_i);
            }
        },
        Number<Tuple<Ts...>::Size()>{});
}
} // namespace

/**
 * \brief Create local partition for thread.
 *
 * \param tensor Tensor for partition.
 * \param thread_lengths Layout of threads.
 * \param thread_id Thread index represented as integer.
 * \param steps Thread step (default=1, raked partition)
 * \return Partition tensor.
 */
template <typename TensorType, typename ThreadLengthsTuple, typename StepsTuple = Tuple<>>
__host__ __device__ constexpr auto make_local_partition(const TensorType& tensor,
                                                        const ThreadLengthsTuple& thread_lengths,
                                                        const index_t thread_id,
                                                        const StepsTuple steps = StepsTuple{})
{
    // Create shape, strides and layout for new partition tensor
    const auto partition_shape = CalculateLocalPartitionShape(shape(tensor), thread_lengths);
    // Create new descriptor and layout
    const auto& flatten_desc = layout(tensor).GetUnnestedDescriptor();
    auto partition_desc =
        CalculateLocalPartitionDescriptor(shape(tensor), thread_lengths, steps, flatten_desc);
    const auto partition_layout = Layout<decltype(partition_shape), decltype(partition_desc)>(
        partition_shape, partition_desc);
    // Calculate offset for new partition tensor
    const auto offset_idx       = CalculateLayoutOffsetIdx(thread_lengths, steps, thread_id);
    const auto partition_offset = layout(tensor)(offset_idx);
    return make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer() + partition_offset,
                                                             partition_layout);
}

/**
 * \brief Create local tile for thread block.
 *
 * \param tensor Tensor for partition.
 * \param tile_shape Shapes of requested tile.
 * \param block_idx Block index represented as tuple.
 * \param steps Block step (default=1, raked partition)
 * \return Tile tensor.
 */
template <typename TensorType,
          typename BlockShapeTuple,
          typename BlockIdxTuple,
          typename StepsTuple = Tuple<>>
__host__ __device__ constexpr auto make_local_tile(const TensorType& tensor,
                                                   const BlockShapeTuple& tile_shape,
                                                   const BlockIdxTuple& block_idx,
                                                   const StepsTuple steps = StepsTuple{})
{
    // Create block lengths, strides and layout for new tile tensor
    const auto block_lengths = CalculateBlockLengths(shape(tensor), tile_shape);
    // Create new descriptor and layout
    const auto& flatten_desc = layout(tensor).GetUnnestedDescriptor();
    auto tile_desc =
        CalculateLocalPartitionDescriptor(tile_shape, block_lengths, steps, flatten_desc);
    const auto tile_layout = Layout<remove_reference_t<decltype(tile_shape)>, decltype(tile_desc)>(
        tile_shape, tile_desc);
    // Calculate offset for new partition tensor
    const auto offset_idx  = CalculateLayoutOffsetIdx(steps, block_idx);
    const auto tile_offset = layout(tensor)(offset_idx);
    return make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer() + tile_offset,
                                                             tile_layout);
}

} // namespace wrapper
} // namespace ck
