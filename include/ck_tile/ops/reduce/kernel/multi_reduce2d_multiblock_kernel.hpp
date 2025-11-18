// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"
#include "ck_tile/core/arch/generic_memory_space_atomic.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp"

// Multi Reduce2d MultiBlock Kernel:
// =======================================
// This kernel implements multiple 2D reduction operation that reduces data along the specified
// dimensions of a matrix. Reductions happen across multiple thread blocks based on intermediate
// reductions at the thread level.

namespace ck_tile {

template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
struct MultiReduceMultiblock
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    static constexpr auto thread_cluster_desc = make_cluster_descriptor(
        ck_tile::sequence<Problem::BlockShape::Block_M / Problem::BlockShape::ThreadTile_M,
                          Problem::BlockShape::Block_N / Problem::BlockShape::ThreadTile_N>{});

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST_DEVICE static void CalculateBlockGroupParams(const int reduce_total_length,
                                                              [[maybe_unused]] int K_BlockTileSize,
                                                              int& num_block_tile_iterations,
                                                              int& block_group_size)
    {

        constexpr int max_block_group_size =
            128; // Maximum 128, as in CK. It balances between latency (i.e. limiting stalls when
                 // performing the atomic operation) and block parallelism.

        num_block_tile_iterations =
            (reduce_total_length + (Problem::BlockShape::Block_N * max_block_group_size) - 1) /
            (Problem::BlockShape::Block_N * max_block_group_size);

        if(num_block_tile_iterations == 0)
        {
            num_block_tile_iterations = 1;
        }

        block_group_size =
            (reduce_total_length + (Problem::BlockShape::Block_N * num_block_tile_iterations) - 1) /
            (Problem::BlockShape::Block_N * num_block_tile_iterations);
    }

    private:
    // Helper function to calculate optimal vector size for input tensor
    template <typename InputShape, typename ReduceDims>
    static constexpr index_t CalculateInputVectorSize()
    {
        using S                              = typename Problem::BlockShape;
        constexpr index_t memory_vector_size = 16 / sizeof(XDataType); // Vectorization
        constexpr index_t thread_tile_vector_size =
            S::ThreadTile_N; // In the continuous dimension, within the tile

        constexpr auto innermost_reduce_dim    = ReduceDims{}.at(number<ReduceDims{}.size() - 1>{});
        constexpr bool is_innermost_contiguous = (innermost_reduce_dim == InputShape{}.size() - 1);

        constexpr index_t stride_based_vector_size =
            is_innermost_contiguous
                ? ck_tile::min(memory_vector_size, thread_tile_vector_size)
                : 1; // Move at "vectorization" steps if continuous otherwise 1 step

        return stride_based_vector_size;
    }

    static constexpr index_t CalculateOutputVectorSize()
    {
        using S                                   = typename Problem::BlockShape;
        constexpr index_t memory_vector_size      = 16 / sizeof(YDataType);
        constexpr index_t thread_tile_vector_size = S::ThreadTile_M;
        constexpr index_t vector_size = ck_tile::min(memory_vector_size, thread_tile_vector_size);

        return vector_size;
    }

    public:
    template <typename InputShape,
              typename InputStrides,
              typename KeptDim,
              typename ReduceDims,
              typename ElementwiseOps,
              typename AccumulatorOps,
              typename InterblockReduceOps>
    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   YDataType* p_y_tuple,
                                   InputShape input_shape,
                                   InputStrides input_strides,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims,
                                   index_t output_tensor_offset,
                                   ElementwiseOps elementwise_ops,
                                   AccumulatorOps accumulator_ops,
                                   InterblockReduceOps interblock_reduce_ops) const
    {

        static_assert(
            ElementwiseOps::size() == Problem::ReduceOp::size() &&
                AccumulatorOps::size() == Problem::ReduceOp::size() &&
                InterblockReduceOps::size() == Problem::ReduceOp::size(),
            "Error: All operations tuple size must match the number of reduction operations");

        using S         = typename Problem::BlockShape;
        auto reduce_ops = typename Problem::ReduceOp{};

        const auto number_operations = reduce_ops.size();

        static_assert(number_operations > 0,
                      "Error: At least one reduction operation must be specified!");

        static_assert(kept_dim.size() + reduce_dims.size() == InputShape::size(),
                      "Size of kept dimensions + reduced dimensions must equal input tensor rank");

        const auto kept_lens = [&]() {
            return generate_tuple([&](auto I) { return input_shape.at(number<kept_dim.at(I)>{}); },
                                  number<kept_dim.size()>{});
        }();
        const auto reduce_lens = [&]() {
            return generate_tuple(
                [&](auto I) { return input_shape.at(number<reduce_dims.at(I)>{}); },
                number<reduce_dims.size()>{});
        }();

        int block_group_size     = 0;
        int num_n_tile_iteration = 0;
        int total_reduce_len     = 1;
        static_for<0, reduce_lens.size(), 1>{}(
            [&](auto i) { total_reduce_len *= reduce_lens.at(i); });

        CalculateBlockGroupParams(
            total_reduce_len, S::Block_N, num_n_tile_iteration, block_group_size);

        constexpr index_t output_vector_size = CalculateOutputVectorSize();

        const auto thread_local_id = get_thread_id();
        const auto block_global_id = get_block_id();                     // Hardware block id
        const auto block_group_id  = block_global_id / block_group_size; // Logical block group id
        const auto block_local_id =
            block_global_id % block_group_size; // Logical block id within the block group

        const auto thread_cluster_idx =
            thread_cluster_desc.calculate_bottom_index(make_tuple(thread_local_id));
        const auto thread_n_cluster_id =
            thread_cluster_idx[number<1>{}]; // cluster index in N dimension

        const auto kept_merge_transform =
            make_merge_transform(kept_lens); // Dimension(s) not reduced are being flattened
        const auto reduce_merge_transform =
            make_merge_transform(reduce_lens); // Dimension(s) to reduce are being flattened

        const auto custom_padding_values = ck_tile::apply(
            [](auto... args) {
                return ck_tile::make_tuple(args.template GetIdentityValue<XDataType>()...);
            },
            reduce_ops); // Get the identity element for each operation

        constexpr auto x_tensor_vector_size =
            CalculateInputVectorSize<InputShape, ReduceDims>(); // Move at "vectorization" steps if
                                                                // continuous otherwise 1 step

        auto desc = make_naive_tensor_descriptor(
            input_shape, input_strides, number<x_tensor_vector_size>{}, number<1>{});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()]; // shared memory reused by
                                                                       // the different operations

        auto block_reduce2d =
            Policy::template GetBlockReduce2d<Problem>(); // Get the block reduction , at thread
                                                          // level, function
        auto block_reduce2d_sync =
            Policy::template GetBlockReduce2dSync<Problem>(); // Get the block sync, at warp level,
                                                              // function
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>(); // Get the block for the
                                                                       // cross warp level sync

        index_t m_offset = S::Block_M * block_group_id;
        index_t n_offset = S::Block_N * num_n_tile_iteration * block_local_id;

        static_for<0, number_operations, 1>{}([&](auto i) {
            auto buffer_view = make_buffer_view<address_space_enum::global>(
                p_x,
                desc.get_element_space_size(),
                custom_padding_values.get(number<i>{})); // Input tensor buffer view

            const auto x_tensor = tensor_view<decltype(buffer_view), decltype(desc)>{
                buffer_view, desc}; // Tensor view over the buffer view and tensor descriptor
            const auto transformed_x_tensor = pad_tensor_view(
                transform_tensor_view(x_tensor,
                                      make_tuple(kept_merge_transform, reduce_merge_transform),
                                      make_tuple(kept_dim, reduce_dims),
                                      make_tuple(sequence<0>{}, sequence<1>{})),
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                sequence<0, 1>{});

            auto x_window = make_tile_window(
                transformed_x_tensor,
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                {m_offset, n_offset},
                Policy::template MakeXBlockTileDistribution<Problem>()); // Input tile windows
                                                                         // and prep the block

            using ComputeDataTensorType = decltype(cast_tile<ComputeDataType>(load_tile(x_window)));

            auto y_compute = block_reduce2d.template MakeYBlockTile<ComputeDataTensorType>();

            set_tile(y_compute,
                     reduce_ops.get(number<i>{}).template GetIdentityValue<ComputeDataType>());

            for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
            {
                auto x = load_tile(x_window);

                // Apply the elementwise operation before the reduction
                auto x_compute = cast_tile<ComputeDataType>(x);

                tile_elementwise_inout(elementwise_ops.get(number<i>{}), x_compute, x_compute);

                block_reduce2d(x_compute, y_compute, reduce_ops.get(number<i>{}));

                move_tile_window(x_window, {0, S::Block_N});
            }

            block_reduce2d_sync(y_compute, reduce_ops.get(number<i>{}));
            block_reduce2d_cross_warp_sync(
                y_compute, static_cast<void*>(smem), reduce_ops.get(number<i>{}));

            if(thread_n_cluster_id == 0)
            {
                tile_elementwise_inout(accumulator_ops.get(number<i>{}), y_compute, y_compute);

                constexpr auto mem_op = interblock_reduce_ops.get(number<i>{}).GetAtomic();

                // Create output tensor view with the specific atomic operation
                auto y_tensor_view = make_naive_tensor_view<address_space_enum::global, mem_op>(
                    p_y_tuple + (i * output_tensor_offset) + (S::Block_M * block_group_id),
                    make_tuple(S::Block_M), // output shape (full block size)
                    make_tuple(1),          // output strides (contiguous)
                    number<output_vector_size>{},
                    number<1>{});

                // Create tile window using y_compute's tile distribution
                // The window origin is 0 because we're starting from the block origin
                // The tile distribution will handle the per-thread positioning
                auto y_window = make_tile_window(
                    y_tensor_view,
                    make_tuple(number<S::ThreadTile_M>{}),
                    {0},
                    y_compute.get_tile_distribution() // Use the distribution from y_compute
                );

                // Cast and update using the tile window (using the atomic op)
                auto y_output = cast_tile<YDataType>(y_compute);
                update_tile(y_window, y_output);
            }
        });
    }

    /// @brief Validates if the given arguments are supported by the 2D multi reduction kernel.
    ///
    /// @param y_continous_dim Size of the continuous dimension of the output tensor.
    ///                        Must be a multiple of ThreadTile_N for proper thread mapping.
    ///
    /// @param input_strides   The stride configuration of the input tensor.
    ///                        The last stride must be 1 to ensure contiguous memory access
    ///                        and enable efficient vectorized loads.
    ///
    /// @return true if the arguments are supported, false otherwise.
    ///         Error messages are logged when CK_TILE_LOGGING is enabled.
    ///
    /// @note Requirements:
    ///       - y_continous_dim % ThreadTile_N == 0 (for proper thread distribution)
    ///       - input_strides[-1] == 1 (for contiguous memory access)
    template <typename InputStrides>
    CK_TILE_HOST static bool IsSupportedArgument(index_t y_continous_dim,
                                                 InputStrides input_strides)
    {
        using S = typename Problem::BlockShape;

        if(y_continous_dim % S::ThreadTile_N != 0)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Total reduction size should be a multiple of ThreadTile_N!");
            }
            return false;
        }

        if(input_strides.at(number<input_strides.size() - 1>{}) != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "Input tensor's last stride must be 1 to support correct vector access!");
            }
            return false;
        }

        return true;
    }
};

} // namespace ck_tile
