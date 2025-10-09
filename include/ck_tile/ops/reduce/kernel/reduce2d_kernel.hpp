// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"

// Reduce2d Kernel:
// =======================================
// This kernel implements a 2D reduction operation that reduces data along the second dimension
// of a matrix. The reduction is performed in multiple hierarchical stages.

namespace ck_tile {

template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
struct Reduce
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
        using S                                   = typename Problem::BlockShape;
        constexpr index_t memory_vector_size      = 16 / sizeof(XDataType);
        constexpr index_t thread_tile_vector_size = S::ThreadTile_N;

        // Check if innermost reduce dimension is the last dimension (stride 1).
        constexpr auto innermost_reduce_dim    = ReduceDims{}.at(number<ReduceDims{}.size() - 1>{});
        constexpr bool is_innermost_contiguous = (innermost_reduce_dim == InputShape{}.size() - 1);

        // If innermost reduce dimension is not the last dim (not contiguous), limit vectorization
        constexpr index_t stride_based_vector_size =
            is_innermost_contiguous ? ck_tile::min(memory_vector_size, thread_tile_vector_size) : 1;

        return stride_based_vector_size;
    }

    // Helper function to calculate optimal vector size for output tensor
    static constexpr index_t CalculateOutputVectorSize()
    {
        using S                                   = typename Problem::BlockShape;
        constexpr index_t memory_vector_size      = 16 / sizeof(YDataType);
        constexpr index_t thread_tile_vector_size = S::ThreadTile_M;
        constexpr index_t vector_size = ck_tile::min(memory_vector_size, thread_tile_vector_size);

        return vector_size;
    }

    public:
    template <typename InputShape, typename InputStrides, typename KeptDim, typename ReduceDims>
    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   YDataType* p_y,
                                   InputShape input_shape,
                                   InputStrides input_strides,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims) const
    {
        using S       = typename Problem::BlockShape;
        const auto iM = get_block_id() * S::Block_M;

        static_assert(kept_dim.size() + reduce_dims.size() == InputShape::size(),
                      "Size of kept dimensions + reduced dimensions must equal input tensor rank");

        // Extract lengths based on kept and reduced dimensions
        const auto kept_lens = [&]() {
            return generate_tuple([&](auto I) { return input_shape.at(number<kept_dim.at(I)>{}); },
                                  number<kept_dim.size()>{});
        }();
        const auto reduce_lens = [&]() {
            return generate_tuple(
                [&](auto I) { return input_shape.at(number<reduce_dims.at(I)>{}); },
                number<reduce_dims.size()>{});
        }();

        const auto kept_merge_transform   = make_merge_transform(kept_lens);
        const auto reduce_merge_transform = make_merge_transform(reduce_lens);

        auto reduce_func = typename Problem::ReduceOp{};
        const XDataType custom_padding_value =
            type_convert<XDataType>(reduce_func.template GetIdentityValue<ComputeDataType>());

        // Calculate optimal vector size for input tensor
        constexpr auto x_tensor_vector_size = CalculateInputVectorSize<InputShape, ReduceDims>();

        // Create input tensor view with custom padding value
        auto desc = make_naive_tensor_descriptor(
            input_shape, input_strides, number<x_tensor_vector_size>{}, number<1>{});

        // Create buffer view with custom padding value
        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_x, desc.get_element_space_size(), custom_padding_value);

        // Create tensor view with custom padding
        const auto x_tensor = tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
        const auto transformed_x_tensor = pad_tensor_view(
            transform_tensor_view(x_tensor,
                                  make_tuple(kept_merge_transform, reduce_merge_transform),
                                  make_tuple(kept_dim, reduce_dims),
                                  make_tuple(sequence<0>{}, sequence<1>{})),
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<0, 1>{});

        // Calculate strides for output tensor based on its own dimensions
        const auto kept_strides = [&]() {
            return generate_tuple(
                [&](auto I) {
                    // Calculate stride for dimension I as product of all following dimensions
                    index_t stride = 1;
                    static_for<I + 1, kept_dim.size(), 1>{}(
                        [&](auto J) { stride *= kept_lens.at(number<J>{}); });
                    return stride;
                },
                number<kept_dim.size()>{});
        }();

        // Calculate optimal vector size for output tensor
        constexpr auto y_tensor_vector_size = CalculateOutputVectorSize();

        const auto y_m = make_naive_tensor_view<address_space_enum::global>(
            p_y, kept_lens, kept_strides, number<y_tensor_vector_size>{}, number<1>{});

        // Transform output tensor to 1D merged view
        // This creates a view compatible with the 2D reduction pattern
        const auto y_merged = transform_tensor_view(
            y_m,
            make_tuple(kept_merge_transform),
            make_tuple(typename arithmetic_sequence_gen<0, kept_dim.size(), 1>::type{}),
            make_tuple(sequence<0>{}));

        auto x_window = make_tile_window(transformed_x_tensor,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_merged, make_tuple(number<S::Block_M>{}), {iM});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()];

        // Get the merged dimension size from the transformed tensor
        const auto merged_reduce_len =
            transformed_x_tensor.get_tensor_descriptor().get_lengths().at(number<1>{});
        index_t num_n_tile_iteration =
            amd_wave_read_first_lane(integer_divide_ceil(merged_reduce_len, S::Block_N));

        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto y_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(y_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        for(int iN = amd_wave_read_first_lane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_reduce2d(x, y_compute, reduce_func);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_compute, reduce_func);
        block_reduce2d_cross_warp_sync(y_compute, smem, reduce_func);

        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }

    /// @brief Validates if the given arguments are supported by the 2D reduction kernel.
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
