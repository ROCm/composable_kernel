// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // The logic below may need to become more advanced once bugs in Stream-K Tile Partitioner are
    // resolved. Because the number of WGs contributing to a macro tile in C may not be the same for
    // all macro tiles in C.

    // Calculate error due to more than 1 WG contributing to the same macro tile in C
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

enum struct GemmPipelineType
{
    Mem,
    CompV3,
    CompV4
};

template <GemmPipelineType PT, typename Problem>
struct GemmPipelineTypeSelector;

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::Mem, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrMem<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrMem<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrMem"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV3, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV3"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV4, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV4"; }
};

namespace detail {
template <typename Lhs, typename Rhs>
struct combine;

template <typename... Lhs, typename... Rhs>
struct combine<::testing::Types<Lhs...>, ::testing::Types<Rhs...>>
{
    using type = ::testing::Types<Lhs..., Rhs...>;
};

template <typename Lhs, typename Rhs>
using combine_t = typename combine<Lhs, Rhs>::type;
} // namespace detail

// This is the base class for all stream-k tests
#define STREAM_K_TEST_CLASS_BASE TestCkTileStreamK

// Macros to help generate test suite names from the parameters given
#define CONCATENATE_TEST_SUITE_NAME(PREFIX, TEST_PARAMS) PREFIX##_##TEST_PARAMS
// Helper macro to expand the arguments before passing them to CONCATENATE_TEST_SUITE_NAME
#define MAKE_TEST_SUITE_NAME_INTERNAL(TEST_BASE_NAME, TEST_PARAMS) \
    CONCATENATE_TEST_SUITE_NAME(TEST_BASE_NAME, TEST_PARAMS)

// Final macro to be used to create the test suite name from the base class name and the test
// parameters
#define MAKE_TEST_SUITE_NAME(TEST_PARAMS) \
    MAKE_TEST_SUITE_NAME_INTERNAL(STREAM_K_TEST_CLASS_BASE, TEST_PARAMS)

// Macro to declare a test suite with the given name and parameters, based on the base test class
#define DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS)  \
    template <typename Tuple>                                      \
    class TEST_SUITE_NAME : public STREAM_K_TEST_CLASS_BASE<Tuple> \
    {                                                              \
    };                                                             \
    TYPED_TEST_SUITE(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
