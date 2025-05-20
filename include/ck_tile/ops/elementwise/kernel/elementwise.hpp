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
    // return std::make_tuple(f(std::integral_constant<std::size_t, Is>{})...);
    return ck_tile::make_tuple(f(std::integral_constant<std::size_t, Is>{})...);
}

template <typename F, std::size_t N>
constexpr auto generate_tuple(F&& f, std::integral_constant<std::size_t, N>) {
    return generate_tuple_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}

// transform tuple
template <typename F, typename Tuple, std::size_t... Is>
constexpr auto transform_tuple_impl(F&& f, Tuple&& t, std::index_sequence<Is...>) {
    // return std::make_tuple(f(std::get<Is>(std::forward<Tuple>(t)))...);
    // return std::make_tuple(f(t.get(number<Is>{}))...);
    // if constexpr (is_std_tuple<std::decay_t<Tuple>>::value) {
    //     return std::make_tuple(f(std::get<Is>(std::forward<Tuple>(t)))...);
    //     // return ck_tile::make_tuple(f(std::get<Is>(std::forward<Tuple>(t)))...);
    // } else {
        // Assuming ck_tile::tuple or compatible with .get(number<Is>{})
        // return std::make_tuple(f(std::forward<Tuple>(t).get(number<Is>{}))...);
        return ck_tile::make_tuple(f(std::forward<Tuple>(t).get(number<Is>{}))...);
    // }
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
    // (f(std::get<Is>(std::forward<Tuple>(t))), ...);
    // if constexpr (is_std_tuple<std::decay_t<Tuple>>::value) {
    //     (f(std::get<Is>(std::forward<Tuple>(t))), ...);
    // } else {
    //     // Assuming ck_tile::tuple or compatible with .get(number<Is>{})
        (f(std::forward<Tuple>(t).get(number<Is>{})), ...);
    // }
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


// Create a tuple of tile windows for all input tensors
// auto make_tile_windows = [&](const auto& input_tensors, const auto& iM) {
//     return generate_tuple(
//         [&](auto idx) {
//             return make_tile_window(
//                 make_naive_tensor_view<address_space_enum::global,
//                                        memory_operation_enum::set,
//                                        amd_buffer_coherence_enum::slc>(
//                     input_tensors.get(idx), make_tuple(M0), make_tuple(1), number<S::Vector_M>{}),
//                 make_tuple(number<S::Block_M>{}),
//                 {iM},
//                 Policy::template MakeXBlockTileDistribution<Problem>());
//         },
//         number<sizeof...(InputTensorType)>{}); // Generate for all input tensors
// };
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
        // const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, const index_t M0) const
        // const index_t M0, const tuple<InputTensorType...>& input_tensors, YDataType* p_y) const
        Dims lens, Dims strides, const tuple<InputTensorType...>& input_tensors, YDataType* p_y) const
    {
        using S = typename Problem::BlockShape;
        // auto inputs_as_tuple = std::make_tuple(input_tensors...);

        // const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
        //                                             memory_operation_enum::set,
        //                                             amd_buffer_coherence_enum::slc>(
        //     input_tensors.get(number<0>()), make_tuple(M0), make_tuple(1), number<S::Vector_M>{});

        // const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
        //                                             memory_operation_enum::set,
        //                                             amd_buffer_coherence_enum::slc>(
        //     input_tensors.get(number<1>()), make_tuple(M0), make_tuple(1), number<S::Vector_M>{});

        const auto iM = get_block_id() * S::Block_M;
        auto merge_tuple = ck_tile::generate_tuple(
                                [&](auto idx) { return lens[idx]; },
                                number<lens.size()>{}
                            );
        auto dim_seq = make_dim_seq<Dims::size()>();

        // auto x_window_a = make_tile_window(x_m_n_a,
        //                                    make_tuple(number<S::Block_M>{}),
        //                                    {iM},
        //                                    Policy::template MakeXBlockTileDistribution<Problem>());

        // auto x_window_b = make_tile_window(x_m_n_b,
        //                                    make_tuple(number<S::Block_M>{}),
        //                                    {iM},
        //                                    Policy::template MakeXBlockTileDistribution<Problem>());
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
        // auto x_window_a = x_windows.get(number<0>{});
        // auto x_window_b = x_windows.get(number<1>{});

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

        // index_t num_m_tile_iteration = 
        //     __builtin_amdgcn_readfirstlane(integer_divide_ceil(lens.template at<lens.size() - 1>(), S::Block_M));

        // for(int i = __builtin_amdgcn_readfirstlane(0); i < num_m_tile_iteration; ++i)
        // {
            // Load tile data
            // const auto xa = load_tile(x_window_a);
            // const auto xb = load_tile(x_window_b);
        const auto x_tiles = load_tiles(x_windows);

        auto y_compute = load_tile(y_window);

        // Process the vector add
        // constexpr auto spans = decltype(x_windows.get(number<0>{}))::get_distributed_spans();
        const auto& x_tile0 = x_tiles.get(number<0>{});
        // constexpr auto spans = decltype(x_tile0)::get_distributed_spans();
        const auto spans = x_tile0.get_distributed_spans();

        sweep_tile_span(spans[number<0>{}], [&](auto idx) {
        
            const auto tile_idx = make_tuple(idx);
            // const auto a_val = type_convert<ComputeDataType>(xa[tile_idx]);
            // const auto b_val = type_convert<ComputeDataType>(xb[tile_idx]);
            const auto x_values = compute_values(x_tiles, tile_idx);

            auto temp = y_compute(tile_idx);
            // ElementWiseOperation{}(y_compute(tile_idx), a_val, b_val);
            // n_ary_operation2(ElementWiseOperation{}, temp,  std::get<0>(x_values), std::get<1>(x_values));
            // 
            apply_tuple([&](auto&&... xs) {
                n_ary_operation2(ElementWiseOperation{}, temp, xs...);
            }, x_values);

            y_compute(tile_idx) = temp; // to avoid temporary object to be use when calling n_ary_operation
        });

        // Store results
        store_tile(y_window, cast_tile<YDataType>(y_compute));
        
        // Move windows to next block (corrected)
        // For 1D operation, we only need to move along the M dimension by Block_M elements
        // move_tile_window(x_window_a, {S::Block_M});
        // move_tile_window(x_window_b, {S::Block_M});
        move_tile_windows(x_windows);
        move_tile_window(y_window, {S::Block_M});
        // }
    }


    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa  = load_tile(x_window_a);
            const auto xb  = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y           = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    ElementWiseOperation{}(y_compute(i_j_idx), x, y);
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }

    template <typename Dims>
    CK_TILE_DEVICE void operator()(
        const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, Dims lens, Dims strides) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_a, lens, strides, number<S::Vector_N>{}, number<1>{});

        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    amd_buffer_coherence_enum::slc>(
            p_x_b, lens, strides, number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, lens, strides, number<S::Vector_N>{}, number<1>{});


        const auto x_m_n_a_t = Policy::template MakeXTransformation<Problem>(x_m_n_a);
        const auto x_m_n_b_t = Policy::template MakeXTransformation<Problem>(x_m_n_b);
        const auto y_m_n_t = Policy::template MakeXTransformation<Problem>(y_m_n);

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a_t,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto x_window_b = make_tile_window(x_m_n_b_t,
                                           make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                           {iM, 0},
                                           Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n_t,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(lens.template at<lens.size() - 1>(), S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa  = load_tile(x_window_a);
            const auto xb  = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y           = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    ElementWiseOperation{}(y_compute(i_j_idx), x, y);
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
