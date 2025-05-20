// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"

namespace ck_tile {

template<typename T> struct is_std_tuple : std::false_type {};
template<typename... Args> struct is_std_tuple<std::tuple<Args...>> : std::true_type {};

template <typename F, std::size_t... Is>
constexpr auto generate_tuple_impl(F&& f, std::index_sequence<Is...>) {
    return ck_tile::make_tuple(f(std::integral_constant<std::size_t, Is>{})...);
}

template <typename F, std::size_t N>
constexpr auto generate_tuple(F&& f, std::integral_constant<std::size_t, N>) {
    return generate_tuple_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}

// transform tuple
template <typename F, typename Tuple, std::size_t... Is>
constexpr auto transform_tuple_impl(F&& f, Tuple&& t, std::index_sequence<Is...>) {
        return ck_tile::make_tuple(f(std::forward<Tuple>(t).get(number<Is>{}))...);
}

template <typename F, typename Tuple>
constexpr auto transform_tuple(F&& f, Tuple&& t) {
    return transform_tuple_impl(
        std::forward<F>(f),
        std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

template <typename F, typename Tuple, std::size_t... Is>
constexpr void for_each_in_tuple_impl(F&& f, Tuple&& t, std::index_sequence<Is...>) {
        (f(std::forward<Tuple>(t).get(number<Is>{})), ...);
}

template <typename F, typename Tuple>
constexpr void for_each_in_tuple(F&& f, Tuple&& t) {
    for_each_in_tuple_impl(
        std::forward<F>(f),
        std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}


// Helper to unpack a tuple and call a function with its elements
// TODO: use CK TILE unpack
template <typename F, typename Tuple, std::size_t... Is>
CK_TILE_DEVICE auto apply_tuple_impl(F&& f, Tuple&& t, std::index_sequence<Is...>)
{
    // return f(t.get(number<Is>{})...);
    if constexpr (is_std_tuple<std::decay_t<Tuple>>::value) {
        return std::forward<F>(f)(std::get<Is>(std::forward<Tuple>(t))...);
    } else {
        // Assuming ck_tile::tuple or compatible with .get(number<Is>{})
        return std::forward<F>(f)(std::forward<Tuple>(t).get(number<Is>{})...);
    }
}

template <typename F, typename Tuple>
CK_TILE_DEVICE auto apply_tuple(F&& f, Tuple&& t)
{
    constexpr std::size_t N = std::tuple_size<std::decay_t<Tuple>>::value;
    return apply_tuple_impl(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<N>{});
}

template <std::size_t... Is>
constexpr auto make_dim_seq_impl(std::index_sequence<Is...>)
{
    return ck_tile::sequence<Is...>{};
}

template <std::size_t N>
constexpr auto make_dim_seq()
{
    return make_dim_seq_impl(std::make_index_sequence<N>{});
}

template <typename Problem_, typename Policy_>
struct ElementWiseKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType            = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType      = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = ck_tile::remove_cvref_t<typename Problem::ElementWiseOperation>;

    template <typename... InputTensorType, typename Dims>
    CK_TILE_DEVICE void operator()(
        Dims lens, Dims strides, const tuple<InputTensorType...>& input_tensors, YDataType* p_y) const
    {
        using S = typename Problem::BlockShape;

        const auto iM = get_block_id() * S::Block_M;
        auto merge_tuple = ck_tile::generate_tuple(
                                [&](auto idx) { return lens[idx]; },
                                number<lens.size()>{}
                            );
        auto dim_seq = make_dim_seq<Dims::size()>();

        auto make_tile_windows = [&](const auto& tensors, const auto& iM_) {
            return generate_tuple(
                [&](auto idx) {
                    return [&] {
                        auto tensor_view = make_naive_tensor_view<address_space_enum::global,
                            memory_operation_enum::set,
                            amd_buffer_coherence_enum::slc>(
                            tensors.get(idx), lens, strides, number<S::Vector_M>{});
                        auto transformed_tensor = transform_tensor_view(tensor_view,
                            ck_tile::make_tuple(make_merge_transform(merge_tuple)),
                            ck_tile::make_tuple(dim_seq),
                            ck_tile::make_tuple(sequence<0>{}));
                        return make_tile_window(
                            transformed_tensor,
                            ck_tile::make_tuple(number<S::Block_M>{}),
                            {iM_},
                            Policy::template MakeXBlockTileDistribution<Problem>());
                    }();
                },
                number<sizeof...(InputTensorType)>{}); // Generate for all input tensors
        };

        auto x_windows = make_tile_windows(input_tensors, iM);

        // Load tiles for all input tensors
        auto load_tiles = [&](const auto& tile_windows) {
            return transform_tuple(
                [&](const auto& window) { return load_tile(window); },
                tile_windows);
        };

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, lens, strides, number<S::Vector_M>{});

        // Transform the tensor view if needed
        auto transformed_y_m_n = transform_tensor_view(y_m_n,
            ck_tile::make_tuple(make_merge_transform(merge_tuple)),
                            ck_tile::make_tuple(dim_seq),
                            ck_tile::make_tuple(sequence<0>{}));

        auto y_window = make_tile_window(transformed_y_m_n,
                                         make_tuple(number<S::Block_M>{}),
                                         {iM},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        
        // Compute values for all input tensors
        auto compute_values = [&](const auto& tiles, const auto& tile_idx) {
            return transform_tuple(
                [&](const auto& tile) { return type_convert<ComputeDataType>(tile[tile_idx]); },
                tiles);
        };

        // Move tile windows for all input tensors
        auto move_tile_windows = [&](auto& tile_windows) {
            for_each_in_tuple(
                [&](auto& window) { move_tile_window(window, {S::Block_M}); },
                tile_windows);
        };

        const auto x_tiles = load_tiles(x_windows);

        auto y_tile = load_tile(y_window);

        // Process the vector add
        const auto& x_tile0 = x_tiles.get(number<0>{});
        const auto spans = x_tile0.get_distributed_spans();

        sweep_tile_span(spans[number<0>{}], [&](auto idx) {
        
            const auto tile_idx = make_tuple(idx);
            const auto x_values = compute_values(x_tiles, tile_idx);

            auto y = y_tile(tile_idx);

            apply_tuple([&](auto&&... xs) {
                binary_operation(ElementWiseOperation{}, y, xs...);
            }, x_values);

            y_tile(tile_idx) = y; // to avoid temporary object to be use when calling n_ary_operation
        });

        // Store results
        store_tile(y_window, cast_tile<YDataType>(y_tile));
        
        // Move windows to next block
        move_tile_windows(x_windows);
        move_tile_window(y_window, {S::Block_M});
    }
};

} // namespace ck_tile
