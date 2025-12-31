// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_diagnostics.hpp"

namespace ck_tile::builder::factory {

using namespace ck_tile::builder::diagnostics;

template <typename T>
struct FwdXdlV3Algorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseFwdXdlGemm)
    CHECK_CONCEPT(T, SpecifiesFwdConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesGemmSpecialization)
    CHECK_CONCEPT(T, SpecifiesBlockGemm)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseFwdXdlGemm;
    static constexpr bool c8 = c_SpecifiesFwdConvSpecialization;
    static constexpr bool c9 = c_SpecifiesGemmSpecialization;
    static constexpr bool c10 = c_SpecifiesBlockGemm;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Forward XDL V3 Algorithm Diagnostic (closest match) ===\n" 
               "Concepts for FwdXdlV3 Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseFwdXdlGemm) +
               DIAGNOSTIC_LINE(SpecifiesFwdConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesGemmSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesBlockGemm);
    }
};

template <typename T>
struct FwdXdlAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseFwdXdlGemm)
    CHECK_CONCEPT(T, SpecifiesFwdConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesGemmSpecialization)
    CHECK_CONCEPT(T, SpecifiesNumPrefetchStages)
    CHECK_CONCEPT(T, SpecifiesNumGroupsToMerge)
    CHECK_CONCEPT(T, SpecifiesLoopScheduler)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseFwdXdlGemm;
    static constexpr bool c8 = c_SpecifiesFwdConvSpecialization;
    static constexpr bool c9 = c_SpecifiesGemmSpecialization;
    static constexpr bool c10 = c_SpecifiesNumPrefetchStages;
    static constexpr bool c11 = c_SpecifiesNumGroupsToMerge;
    static constexpr bool c12 = c_SpecifiesLoopScheduler;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12;
    }

    static consteval auto message() -> std::string {
        return  std::string("\n=== Forward XDL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdXdl Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseFwdXdlGemm) +
               DIAGNOSTIC_LINE(SpecifiesFwdConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesGemmSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesNumPrefetchStages) +
               DIAGNOSTIC_LINE(SpecifiesNumGroupsToMerge) +
               DIAGNOSTIC_LINE(SpecifiesLoopScheduler);
    }
};

template <typename T>
struct FwdWmmaAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseWmmaGemm)
    CHECK_CONCEPT(T, SpecifiesFwdConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesGemmSpecialization)
    CHECK_CONCEPT(T, SpecifiesNumPrefetchStages)
    CHECK_CONCEPT(T, SpecifiesLoopScheduler)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseWmmaGemm;
    static constexpr bool c8 = c_SpecifiesFwdConvSpecialization;
    static constexpr bool c9 = c_SpecifiesGemmSpecialization;
    static constexpr bool c10 = c_SpecifiesNumPrefetchStages;
    static constexpr bool c11 = c_SpecifiesLoopScheduler;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Forward WMMA Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdWmma Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseWmmaGemm) +
               DIAGNOSTIC_LINE(SpecifiesFwdConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesGemmSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesNumPrefetchStages) +
               DIAGNOSTIC_LINE(SpecifiesLoopScheduler);
    }
};

template <typename T>
struct FwdDlAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesFwdConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesGemmSpecialization)
    CHECK_CONCEPT(T, SpecifiesDlThreadConfig)
    CHECK_CONCEPT(T, SpecifiesDlThreadCluster)
    CHECK_CONCEPT(T, SpecifiesDlBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesDlEpilogue)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesFwdConvSpecialization;
    static constexpr bool c4 = c_SpecifiesGemmSpecialization;
    static constexpr bool c5 = c_SpecifiesDlThreadConfig;
    static constexpr bool c6 = c_SpecifiesDlThreadCluster;
    static constexpr bool c7 = c_SpecifiesDlBlockTransfer;
    static constexpr bool c8 = c_SpecifiesDlEpilogue;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Forward DL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdDl Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesFwdConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesGemmSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesDlThreadConfig) +
               DIAGNOSTIC_LINE(SpecifiesDlThreadCluster) +
               DIAGNOSTIC_LINE(SpecifiesDlBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesDlEpilogue);
    }
};

template <typename T>
struct TileAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesTileThreadBlock)
    CHECK_CONCEPT(T, SpecifiesTileTransfer)
    CHECK_CONCEPT(T, SpecifiesTileConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesTileBlockGemm)
    CHECK_CONCEPT(T, SpecifiesTileOptimizations)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesTileThreadBlock;
    static constexpr bool c3 = c_SpecifiesTileTransfer;
    static constexpr bool c4 = c_SpecifiesTileConvSpecialization;
    static constexpr bool c5 = c_SpecifiesTileBlockGemm;
    static constexpr bool c6 = c_SpecifiesTileOptimizations;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== CK Tile Algorithm Diagnostic (closest match) ===\n" 
               "Concepts for CK Tile Conv Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesTileThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesTileTransfer) +
               DIAGNOSTIC_LINE(SpecifiesTileConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesTileBlockGemm) +
               DIAGNOSTIC_LINE(SpecifiesTileOptimizations);
    }
};

template <typename T>
struct LargeTensorAlgorithm : public FwdXdlAlgorithm<decltype(T::base_algorithm)>
{
    using BaseAlgorithmType = decltype(T::base_algorithm);
    CHECK_CONCEPT(T, SpecifiesLargeTensorSupport)

    static constexpr bool c13 = c_SpecifiesLargeTensorSupport;

    static consteval bool is_valid() {
        return FwdXdlAlgorithm<BaseAlgorithmType>::is_valid() && c13;
    }

    static consteval auto message() -> std::string {
        return FwdXdlAlgorithm<BaseAlgorithmType>::message() +
               DIAGNOSTIC_LINE(SpecifiesLargeTensorSupport);
    }
};

template <typename T>
struct BwdXdlAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer4D)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseBwdXdlGemm)
    CHECK_CONCEPT(T, SpecifiesBwdWeightConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesTransposeTransfer)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer4D;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseBwdXdlGemm;
    static constexpr bool c8 = c_SpecifiesBwdWeightConvSpecialization;
    static constexpr bool c9 = c_SpecifiesTransposeTransfer;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Backward XDL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for BwdXdl Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer4D) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseBwdXdlGemm) +
               DIAGNOSTIC_LINE(SpecifiesBwdWeightConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesTransposeTransfer);
    }
};

template <typename T>
struct BwdXdlV3Algorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseBwdXdlGemm)
    CHECK_CONCEPT(T, SpecifiesBwdWeightConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesBlockGemm)
    CHECK_CONCEPT(T, SpecifiesGenericInstance)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseBwdXdlGemm;
    static constexpr bool c8 = c_SpecifiesBwdWeightConvSpecialization;
    static constexpr bool c9 = c_SpecifiesBlockGemm;
    static constexpr bool c10 = c_SpecifiesGenericInstance;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Backward XDL V3 Algorithm Diagnostic (closest match) ===\n"
               "Concepts for BwdXdlV3 Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseBwdXdlGemm) +
               DIAGNOSTIC_LINE(SpecifiesBwdWeightConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesBlockGemm) +
               DIAGNOSTIC_LINE(SpecifiesGenericInstance);
    }
};

template <typename T>
struct BwdTwoStageXdlAlgorithm {
    CHECK_CONCEPT(T, ConvAlgorithmDescriptor)
    CHECK_CONCEPT(T, SpecifiesThreadBlock)
    CHECK_CONCEPT(T, SpecifiesBlockTransfer)
    CHECK_CONCEPT(T, SpecifiesLdsTransfer)
    CHECK_CONCEPT(T, SpecifiesThreadClusterAccessOrder)
    CHECK_CONCEPT(T, SpecifiesSourceAccessOrder)
    CHECK_CONCEPT(T, SpecifiesGridwiseBwdXdlGemm)
    CHECK_CONCEPT(T, SpecifiesBwdWeightConvSpecialization)
    CHECK_CONCEPT(T, SpecifiesBlockGemm)
    CHECK_CONCEPT(T, SpecifiesTransposeTransfer)
    CHECK_CONCEPT(T, SpecifiesGemmBatchOptions)
    CHECK_CONCEPT(T, SpecifiesTwoStageSupport)

    static constexpr bool c1 = c_ConvAlgorithmDescriptor;
    static constexpr bool c2 = c_SpecifiesThreadBlock;
    static constexpr bool c3 = c_SpecifiesBlockTransfer;
    static constexpr bool c4 = c_SpecifiesLdsTransfer;
    static constexpr bool c5 = c_SpecifiesThreadClusterAccessOrder;
    static constexpr bool c6 = c_SpecifiesSourceAccessOrder;
    static constexpr bool c7 = c_SpecifiesGridwiseBwdXdlGemm;
    static constexpr bool c8 = c_SpecifiesBwdWeightConvSpecialization;
    static constexpr bool c9 = c_SpecifiesBlockGemm;
    static constexpr bool c10 = c_SpecifiesTransposeTransfer;
    static constexpr bool c11 = c_SpecifiesGemmBatchOptions;
    static constexpr bool c12 = c_SpecifiesTwoStageSupport;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && & c10 && c11 && c12;
    }

    static consteval auto message() -> std::string {
        return std::string("\n=== Backward two stage XDL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for BwdXdlV3 Algorithm:\n") +
               DIAGNOSTIC_LINE(ConvAlgorithmDescriptor) +
               DIAGNOSTIC_LINE(SpecifiesThreadBlock) +
               DIAGNOSTIC_LINE(SpecifiesBlockTransfer) +
               DIAGNOSTIC_LINE(SpecifiesLdsTransfer) +
               DIAGNOSTIC_LINE(SpecifiesThreadClusterAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesSourceAccessOrder) +
               DIAGNOSTIC_LINE(SpecifiesGridwiseBwdXdlGemm) +
               DIAGNOSTIC_LINE(SpecifiesBwdWeightConvSpecialization) +
               DIAGNOSTIC_LINE(SpecifiesBlockGemm) +
               DIAGNOSTIC_LINE(SpecifiesTransposeTransfer) +
               DIAGNOSTIC_LINE(SpecifiesGemmBatchOptions) +
               DIAGNOSTIC_LINE(SpecifiesTwoStageSupport);
    }
};

template <typename T>
consteval int count_matches_fwd_xdl_v3() {
    using Alg = FwdXdlV3Algorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9 + Alg::c10;
}

template <typename T>
consteval int count_matches_fwd_xdl() {
    using Alg = FwdXdlAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9 + Alg::c10 + Alg::c11 + Alg::c12;
}

template <typename T>
consteval int count_matches_fwd_wmma() {
    using Alg = FwdWmmaAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9 + Alg::c10 + Alg::c11;
}

template <typename T>
consteval int count_matches_fwd_dl() {
    using Alg = FwdDlAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8;
}

template <typename T>
consteval int count_matches_bwd_xdl() {
    using Alg = BwdXdlAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9;
}

template <typename T>
consteval int count_matches_bwd_xdl_v3() {
    using Alg = BwdXdlV3Algorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9;
}

template <typename T>
consteval int count_matches_bwd_two_stage_xdl() {
    using Alg = BwdTwoStageXdlAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9 + Alg::c10 + Alg::c11 + Alg::c12;
}

template <typename T>
consteval int count_matches_large_tensor() {
    using Alg = LargeTensorAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6 + Alg::c7 + Alg::c8 + Alg::c9 + Alg::c10 + Alg::c11 + Alg::c12 + Alg::c13;
}

template <typename T>
consteval int count_matches_tile() {
    using Alg = TileAlgorithm<T>;
    return Alg::c1 + Alg::c2 + Alg::c3 + Alg::c4 + Alg::c5 + Alg::c6;
}

template <typename AlgoType>
consteval void diagnose_fwd_algorithm_signature() 
{
    // Find closest matching variant
    constexpr int xdl_v3_matches = count_matches_fwd_xdl_v3<AlgoType>();
    constexpr int xdl_matches = count_matches_fwd_xdl<AlgoType>();
    constexpr int wmma_matches = count_matches_fwd_wmma<AlgoType>();
    constexpr int dl_matches = count_matches_fwd_dl<AlgoType>();
    constexpr int large_tensor_matches = count_matches_large_tensor<AlgoType>();
    constexpr int tile_matches = count_matches_tile<AlgoType>();
    
    // Find maximum matches across all variants
    constexpr int max_1 = xdl_v3_matches > xdl_matches ? xdl_v3_matches : xdl_matches;
    constexpr int max_2 = wmma_matches > dl_matches ? wmma_matches : dl_matches;
    constexpr int max_3 = max_1 > max_2 ? max_1 : max_2;
    constexpr int max_4 = max_3 > large_tensor_matches ? max_3 : large_tensor_matches;
    constexpr int max_matches = max_4 > tile_matches ? max_4 : tile_matches;
    
    // Generate detailed diagnostic for the closest match
    if constexpr(max_matches == xdl_v3_matches) {
        using Alg = FwdXdlV3Algorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else if constexpr(max_matches == xdl_matches) {
        using Alg = FwdXdlAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else if constexpr(max_matches == wmma_matches) {
        using Alg = FwdWmmaAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else if constexpr(max_matches == dl_matches) {
        using Alg = FwdDlAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else if constexpr (max_matches == large_tensor_matches) {
        using Alg = LargeTensorAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else if constexpr (max_matches == tile_matches) {
        using Alg = TileAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    }
    else {
        // This should never happen
        static_assert(false, "Internal Error: No matching algorithm variant found for diagnostics.");
    }
}

template <typename AlgoType>
consteval void diagnose_bwd_weight_algorithm_signature() 
{
    constexpr int xdl_matches = count_matches_bwd_xdl<AlgoType>();
    constexpr int xdl_v3_matches = count_matches_fwd_xdl_v3<AlgoType>();
    constexpr int two_stage_xdl_matches = count_matches_bwd_two_stage_xdl<AlgoType>();

    constexpr int max1 = xdl_v3_matches > xdl_matches ? xdl_v3_matches : xdl_matches;
    constexpr int max_matches = max1 > two_stage_xdl_matches ? max1 : two_stage_xdl_matches;

    if constexpr (max_matches == xdl_matches) {
        using Alg = BwdXdlAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } 
    else if constexpr (max_matches == xdl_v3_matches) {
        using Alg = BwdXdlV3Algorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    }
    else if constexpr (max_matches == two_stage_xdl_matches) {
        using Alg = BwdTwoStageXdlAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    }
    else {
        // This should never happen
        static_assert(false, "Internal Error: No matching algorithm variant found for diagnostics.");
    }
}

}
