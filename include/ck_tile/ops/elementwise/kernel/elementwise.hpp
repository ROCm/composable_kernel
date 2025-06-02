// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct ElementWiseKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType            = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType      = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = ck_tile::remove_cvref_t<typename Problem::ElementWiseOperation>;

    template <typename... XDataType, typename Dims>
    CK_TILE_DEVICE void operator()(Dims lens,
                                   Dims input_strides,
                                   Dims output_strides,
                                   const tuple<XDataType...>& input_tensors,
                                   YDataType* p_y) const
    {
        using S = typename Problem::BlockShape;

        const auto iM = get_block_id() * S::kBlockM;
        auto merge_tuple =
            ck_tile::generate_tuple([&](auto idx) { return lens[idx]; }, number<lens.size()>{});

        auto dim_seq = make_index_sequence<Dims::size()>{};

        auto make_tile_windows = [&](const auto& tensors, const auto& iM_) {
            return generate_tuple(
                [&](auto idx) {
                    auto tensor_view = make_naive_tensor_view<address_space_enum::global,
                                                              memory_operation_enum::set,
                                                              amd_buffer_coherence_enum::slc>(
                        tensors.get(idx), lens, input_strides, number<S::kVectorM>{});

                    auto transformed_tensor = transform_tensor_view(
                        tensor_view,
                        ck_tile::make_tuple(make_merge_transform(merge_tuple)),
                        ck_tile::make_tuple(dim_seq),
                        ck_tile::make_tuple(sequence<0>{}));

                    return make_tile_window(transformed_tensor,
                                            ck_tile::make_tuple(number<S::kBlockM>{}),
                                            {iM_},
                                            Policy::template MakeXBlockTileDistribution<Problem>());
                },
                number<sizeof...(XDataType)>{}); // Generate for all input tensors
        };

        auto x_windows = make_tile_windows(input_tensors, iM);

        // Load tiles for all input tensors
        auto load_tiles = [&](const auto& tile_windows) {
            return transform_tuples([&](const auto& window) { return load_tile(window); },
                                    tile_windows);
        };

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  amd_buffer_coherence_enum::slc>(
            p_y, lens, output_strides, number<S::kVectorM>{});

        // Transform the tensor view if needed
        auto transformed_y_m_n =
            transform_tensor_view(y_m_n,
                                  ck_tile::make_tuple(make_merge_transform(merge_tuple)),
                                  ck_tile::make_tuple(dim_seq),
                                  ck_tile::make_tuple(sequence<0>{}));

        auto y_window = make_tile_window(transformed_y_m_n,
                                         make_tuple(number<S::kBlockM>{}),
                                         {iM},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        // Compute values for all input tensors
        auto compute_values = [&](const auto& tiles, const auto& tile_idx) {
            return transform_tuples(
                [&](const auto& tile) { return type_convert<ComputeDataType>(tile[tile_idx]); },
                tiles);
        };

        const auto x_tiles = load_tiles(x_windows);

        // Process the vector operation
        const auto& x_tile0 = x_tiles.get(number<0>{});
        const auto spans    = x_tile0.get_distributed_spans();
        auto y_tile = make_static_distributed_tensor<YDataType>(x_tile0.get_tile_distribution());

        sweep_tile_span(spans[number<0>{}], [&](auto idx) {
            const auto tile_idx = make_tuple(idx);
            const auto x_values = compute_values(x_tiles, tile_idx);

            apply([&](auto&&... vals) { ElementWiseOperation{}(y_tile(tile_idx), vals...); }, x_values);
        });

        // Store results
        store_tile(y_window, cast_tile<YDataType>(y_tile));
    }
};

} // namespace ck_tile
