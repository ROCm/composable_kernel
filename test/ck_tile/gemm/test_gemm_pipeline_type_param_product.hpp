// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <tuple>
#include "gtest/gtest.h"

// Helper to create flattened cartesian product of GemmConfig × PrecTypes
template <typename GemmConfigs, typename PrecTypes>
struct CartesianProduct;

// Specialization for the actual cartesian product implementation
template <typename... GemmConfigs, typename... PrecTypes>
struct CartesianProduct<::testing::Types<GemmConfigs...>, ::testing::Types<PrecTypes...>>
{
    private:
    // Helper to flatten a single PrecType tuple with GemmConfig
    template <typename GemmConfig, typename PrecType>
    struct FlattenHelper;

    template <typename GemmConfig, typename APrecType, typename BPrecType, typename CPrecType>
    struct FlattenHelper<GemmConfig, std::tuple<APrecType, BPrecType, CPrecType>>
    {
        using type = std::tuple<GemmConfig, APrecType, BPrecType, CPrecType>;
    };

    // Helper to generate all flattened combinations of one GemmConfig with all PrecTypes
    template <typename GemmConfig>
    using MakeCombinations =
        ::testing::Types<typename FlattenHelper<GemmConfig, PrecTypes>::type...>;

    // Concatenate all type lists
    template <typename... TypeLists>
    struct Concatenate;

    // Base case: single type list
    template <typename... Types>
    struct Concatenate<::testing::Types<Types...>>
    {
        using type = ::testing::Types<Types...>;
    };

    // Two type lists
    template <typename... Types1, typename... Types2>
    struct Concatenate<::testing::Types<Types1...>, ::testing::Types<Types2...>>
    {
        using type = ::testing::Types<Types1..., Types2...>;
    };

    // Three or more type lists - recursive case
    template <typename TypeList1, typename TypeList2, typename... Rest>
    struct Concatenate<TypeList1, TypeList2, Rest...>
    {
        using type =
            typename Concatenate<typename Concatenate<TypeList1, TypeList2>::type, Rest...>::type;
    };

    public:
    using type = typename Concatenate<MakeCombinations<GemmConfigs>...>::type;
};

template <typename GemmConfigs, typename PrecTypes>
using CartesianProduct_t = typename CartesianProduct<GemmConfigs, PrecTypes>::type;
