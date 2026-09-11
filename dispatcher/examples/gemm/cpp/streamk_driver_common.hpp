// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Shared helpers for the Stream-K GEMM example drivers (03 standalone and 04
 * registry). Kept in one place so the two drivers do not duplicate CLI parsing,
 * layout/dtype tags, and the Stream-K verification tolerance.
 */

#include <algorithm>
#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "ck_tile/dispatcher/kernel_key.hpp"

template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

inline std::string get_opt(int argc, char** argv, const std::string& key, const std::string& def)
{
    for(int i = 1; i < argc - 1; ++i)
        if(key == argv[i])
            return argv[i + 1];
    return def;
}

// Map a ck_tile element type to the dispatcher's DataType enum so the registry
// key reflects the kernel that was actually generated (fp16/bf16/fp8/bf8/...),
// instead of assuming fp16. Keeps the registry identifier and selection correct
// across every datatype the codegen emits.
template <typename T>
constexpr ck_tile::dispatcher::DataType dtype_enum_of()
{
    using U = ck_tile::remove_cvref_t<T>;
    if constexpr(std::is_same_v<U, ck_tile::fp16_t>)
        return ck_tile::dispatcher::DataType::FP16;
    else if constexpr(std::is_same_v<U, ck_tile::bf16_t>)
        return ck_tile::dispatcher::DataType::BF16;
    else if constexpr(std::is_same_v<U, ck_tile::fp8_t>)
        return ck_tile::dispatcher::DataType::FP8;
    else if constexpr(std::is_same_v<U, ck_tile::bf8_t>)
        return ck_tile::dispatcher::DataType::BF8;
    else if constexpr(std::is_same_v<U, ck_tile::int8_t>)
        return ck_tile::dispatcher::DataType::INT8;
    else if constexpr(std::is_same_v<U, float>)
        return ck_tile::dispatcher::DataType::FP32;
    else
        return ck_tile::dispatcher::DataType::UNKNOWN;
}

template <typename Layout>
constexpr ck_tile::dispatcher::LayoutTag layout_tag_of()
{
    return std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>
               ? ck_tile::dispatcher::LayoutTag::RowMajor
               : ck_tile::dispatcher::LayoutTag::ColMajor;
}

struct StreamKTolerance
{
    double rtol;
    double atol;
};

// Stream-K verification tolerance. Stream-K splits K across CUs and reduces the
// partials; atomic reduction accumulates them directly into low-precision C, so
// the tolerance must account for the split-K accumulation error -- exactly as
// tile_engine's calculate_rtol_atol does. The plain single-pass
// get_relative/absolute_threshold(K) under-estimates the error and would
// spuriously FAIL correct atomic results on small-M/N, large-K shapes.
//
// `num_wgs_per_tile` is the number of workgroups reducing into a single output
// tile (Stream-K has no fixed split-k), taken from the kernel's own tile
// partitioner so the driver and tile_engine agree on the split factor.
template <typename ComputeType, typename CDataType, typename AccDataType>
inline StreamKTolerance
streamk_tolerance(ck_tile::index_t K, ck_tile::index_t num_wgs_per_tile, float maxv)
{
    const ck_tile::index_t k_per_split = ck_tile::integer_divide_ceil(K, num_wgs_per_tile);
    // single-pass (per-split) tolerance
    const double rtol_base =
        ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(k_per_split);
    const double atol_base = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        maxv / num_wgs_per_tile, k_per_split);
    // error contributed by reducing num_wgs_per_tile partials in low-precision C
    const double rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(num_wgs_per_tile);
    const double atol_split_k =
        ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(maxv, num_wgs_per_tile);
    return {std::max(rtol_base, rtol_split_k), std::max(atol_base, atol_split_k)};
}
