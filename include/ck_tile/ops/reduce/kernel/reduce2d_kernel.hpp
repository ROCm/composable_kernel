// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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

#if 0
    CK_TILE_DEVICE void operator()(const XDataType* p_x, YDataType* p_y, index_t M, index_t N)
    const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M), number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});

        const auto f_reduce = [](const auto& v0, const auto& v1) { return v0 + v1; };

        const XDataType reduce_init_value = 0;

        constexpr auto reduce_dims = sequence<1>{};

        auto y_compute = decltype(block_tile_reduce<ComputeDataType>(
            load_tile(x_window), reduce_dims, f_reduce, reduce_init_value)){};

        set_tile(y_compute, reduce_init_value);

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_tile_reduce(y_compute, x, reduce_dims, f_reduce);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_tile_reduce_sync(y_compute, f_reduce);

        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }
#else
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

        // Extract lengths based on kept and reduced dimensions
        const auto kept_len    = input_shape.at(number<kept_dim.at(0)>{});
        const auto reduce_lens = [&]() {
            return generate_tuple(
                [&](auto I) { return input_shape.at(number<reduce_dims.at(I)>{}); },
                number<reduce_dims.size()>{});
        }();

        // Create transforms
        const auto pass_through_transform = make_pass_through_transform(kept_len);
        const auto merge_transform        = make_merge_transform(reduce_lens);

        auto reduce_func = typename Problem::ReduceOp{};
        const XDataType custom_padding_value =
            type_convert<XDataType>(reduce_func.template GetIdentityValue<ComputeDataType>());

        // Create input tensor view with custom padding value
        // First create the descriptor
        auto desc = make_naive_tensor_descriptor(
            input_shape, input_strides, number<S::Vector_N>{}, number<1>{});

        // Create buffer view with custom padding value
        auto buffer_view = make_buffer_view<address_space_enum::global>(
            p_x, desc.get_element_space_size(), custom_padding_value);

        // Create tensor view with custom padding
        const auto x_tensor = tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
        const auto transformed_x_tensor = pad_tensor_view(
            transform_tensor_view(x_tensor,
                                  ck_tile::make_tuple(pass_through_transform, merge_transform),
                                  ck_tile::make_tuple(kept_dim, reduce_dims),
                                  ck_tile::make_tuple(sequence<0>{}, sequence<1>{})),
            make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
            sequence<0, 1>{});

        const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(kept_len), number<1>{});

        auto x_window = make_tile_window(transformed_x_tensor,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()];

        // Get the merged dimension size from the transformed tensor
        const auto merged_reduce_len =
            transformed_x_tensor.get_tensor_descriptor().get_lengths().at(number<1>{});
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(merged_reduce_len, S::Block_N));

        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto y_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(y_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_reduce2d(x, y_compute, reduce_func);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_compute, reduce_func);
        block_reduce2d_cross_warp_sync(y_compute, smem, reduce_func);

        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }

    template <typename ArgParser>
    CK_TILE_HOST static bool IsSupportedArgument(const ArgParser& arg_parser)
    {
        using S = typename Problem::BlockShape;
        if(arg_parser.get_int("n") % S::Vector_N != 0)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Size of n dimension should be a multiple of Vector_N !");
            }
            return false;
        }
        return true;
    }

#endif
};

} // namespace ck_tile
