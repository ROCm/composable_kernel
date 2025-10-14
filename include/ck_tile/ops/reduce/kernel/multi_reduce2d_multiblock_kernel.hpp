// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"

// Multi Reduce2d MultiBlock Kernel:
// =======================================
// This kernel implements multiple 2D reduction operation that reduces data along the specified
// dimensions of a matrix.

namespace ck_tile {

template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
struct MultiReduce
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;
    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
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
        constexpr index_t thread_tile_vector_size = S::ThreadTile_M; // ?
        constexpr index_t vector_size = ck_tile::min(memory_vector_size, thread_tile_vector_size);

        return vector_size;
    }

    public:
    template <typename InputShape, typename InputStrides, typename KeptDim, typename ReduceDims>
    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   YDataType* p_y_tuple,
                                   InputShape input_shape,
                                   InputStrides input_strides,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims,
                                   index_t output_tensor_offset) const
    {
        using S                      = typename Problem::BlockShape;
        const auto iM                = get_block_id() * S::Block_M;
        auto reduce_funcs            = typename Problem::ReduceOp{};
        const auto number_operations = reduce_funcs.size();

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

        const auto kept_merge_transform =
            make_merge_transform(kept_lens); // Dimension(s) not reduced are being flattened
        const auto reduce_merge_transform =
            make_merge_transform(reduce_lens); // Dimension(s) to reduce are being flattened

        const auto custom_padding_values = ck_tile::apply(
            [](auto... args) {
                return ck_tile::make_tuple(
                    type_convert<XDataType>(args.template GetIdentityValue<ComputeDataType>())...);
            },
            reduce_funcs); // Get the identity element for each operation

        constexpr auto x_tensor_vector_size =
            CalculateInputVectorSize<InputShape, ReduceDims>(); // Move at "vectorization" steps if
                                                                // continuous otherwise 1 step

        auto desc = make_naive_tensor_descriptor(input_shape,
                                                 input_strides,
                                                 number<x_tensor_vector_size>{},
                                                 number<1>{}); // Create the tensor descriptor

        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_x,
            desc.get_element_space_size(),
            custom_padding_values.get(
                number<0>{})); // Input tensor buffer view, currently using the first operation
                               // identity element as padding?? TODO: check this

        const auto x_tensor = tensor_view<decltype(buffer_view), decltype(desc)>{
            buffer_view, desc}; // Tensor view over the buffer view and tensor descriptor
        const auto transformed_x_tensor = pad_tensor_view(
            transform_tensor_view(x_tensor,
                                  make_tuple(kept_merge_transform, reduce_merge_transform),
                                  make_tuple(kept_dim, reduce_dims),
                                  make_tuple(sequence<0>{}, sequence<1>{})),
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<0, 1>{}); // Effective transform the input tensor to 2D (kept, reduced) and pad
                               // to block size

        const auto kept_strides = [&]() {
            return generate_tuple(
                [&](auto I) {
                    index_t stride = 1;
                    static_for<I + 1, kept_dim.size(), 1>{}(
                        [&](auto J) { stride *= kept_lens.at(number<J>{}); });
                    return stride;
                },
                number<kept_dim.size()>{});
        }(); // Compute the strides, for a dimensions (a,b,c), the strides are (b*c, c, 1)

        constexpr auto y_tensor_vector_size = CalculateOutputVectorSize();

        auto y_tile_windows = generate_tuple(
            [&]([[maybe_unused]] auto i) {
                const auto y_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                    p_y_tuple + (i * output_tensor_offset),
                    kept_lens,
                    kept_strides,
                    number<y_tensor_vector_size>{},
                    number<1>{});

                const auto y_merge = transform_tensor_view(
                    y_tensor_view,
                    make_tuple(kept_merge_transform),
                    make_tuple(typename arithmetic_sequence_gen<0, kept_dim.size(), 1>::type{}),
                    make_tuple(sequence<0>{}));

                return make_tile_window(y_merge, make_tuple(number<S::Block_M>{}), {iM});
            },
            number<number_operations>{});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()]; // shared memory is reused
                                                                       // for each operation

        const auto merged_reduce_len =
            transformed_x_tensor.get_tensor_descriptor().get_lengths().at(
                number<1>{}); // Get the last dimension size (reduced dimension)
        index_t num_n_tile_iteration = __builtin_amdgcn_readfirstlane(integer_divide_ceil(
            merged_reduce_len, S::Block_N)); // Figure out the number of iterations needed to cover
                                             // the reduced dimension with Block size N

        auto block_reduce2d =
            Policy::template GetBlockReduce2d<Problem>(); // Get the block reduction , at thread
                                                          // level, function
        auto block_reduce2d_sync =
            Policy::template GetBlockReduce2dSync<Problem>(); // Get the block sync, at warp level,
                                                              // function
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>(); // Get the block for the
                                                                       // cross warp level sync

        static_for<0, number_operations, 1>{}([&](auto i) {
            auto x_window = make_tile_window(
                transformed_x_tensor,
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                {iM, 0},
                Policy::template MakeXBlockTileDistribution<Problem>()); // Input tile windows and
                                                                         // prep the block

            using XTensorType = decltype(load_tile(x_window)); // Load input tile

            auto y_compute = block_reduce2d.template MakeYBlockTile<XTensorType>();

            set_tile(y_compute,
                     reduce_funcs.get(number<i>{}).template GetIdentityValue<ComputeDataType>());

            for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
            {
                const auto x = load_tile(x_window);

                block_reduce2d(x, y_compute, reduce_funcs.get(number<i>{}));

                move_tile_window(x_window, {0, S::Block_N});
            }

            block_reduce2d_sync(y_compute, reduce_funcs.get(number<i>{}));
            block_reduce2d_cross_warp_sync(
                y_compute, static_cast<void*>(smem), reduce_funcs.get(number<i>{}));

            // Store the result back in each element of the tuple
            store_tile(y_tile_windows.get(number<i>{}), cast_tile<YDataType>(y_compute));
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
