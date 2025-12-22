// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory {

#define CHECK_MARK(cond) (cond ? "[✓]" : "[✗]")

template <typename T>
struct FwdXdlV3Algorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesThreadBlock<T>;
    static constexpr bool c3 = SpecifiesBlockTransfer<T>;
    static constexpr bool c4 = SpecifiesLdsTransfer<T>;
    static constexpr bool c5 = SpecifiesThreadClusterAccessOrder<T>;
    static constexpr bool c6 = SpecifiesSourceAccessOrder<T>;
    static constexpr bool c7 = SpecifiesGridwiseFwdXdlGemm<T>;
    static constexpr bool c8 = SpecifiesFwdConvSpecialization<T>;
    static constexpr bool c9 = SpecifiesGemmSpecialization<T>;
    static constexpr bool c10 = SpecifiesBlockGemm<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10;
    }

    static consteval const std::string message() {
        return "\n=== Forward XDL V3 Algorithm Diagnostic (closest match) ===\n" 
               "Concepts for FwdXdlV3 Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesBlockTransfer: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesLdsTransfer: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesThreadClusterAccessOrder: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesSourceAccessOrder: " + std::string(CHECK_MARK(c6)) + "\n"
               "  SpecifiesGridwiseFwdXdlGemm: " + std::string(CHECK_MARK(c7)) + "\n"
               "  SpecifiesFwdConvSpecialization: " + std::string(CHECK_MARK(c8)) + "\n"
               "  SpecifiesGemmSpecialization: " + std::string(CHECK_MARK(c9)) + "\n"
               "  SpecifiesBlockGemm: " + std::string(CHECK_MARK(c10)) + "\n";
    }
};

template <typename T>
struct FwdXdlAlgorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesThreadBlock<T>;
    static constexpr bool c3 = SpecifiesBlockTransfer<T>;
    static constexpr bool c4 = SpecifiesLdsTransfer<T>;
    static constexpr bool c5 = SpecifiesThreadClusterAccessOrder<T>;
    static constexpr bool c6 = SpecifiesSourceAccessOrder<T>;
    static constexpr bool c7 = SpecifiesGridwiseFwdXdlGemm<T>;
    static constexpr bool c8 = SpecifiesFwdConvSpecialization<T>;
    static constexpr bool c9 = SpecifiesGemmSpecialization<T>;
    static constexpr bool c10 = SpecifiesNumPrefetchStages<T>;
    static constexpr bool c11 = SpecifiesNumGroupsToMerge<T>;
    static constexpr bool c12 = SpecifiesLoopScheduler<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12;
    }

    static consteval const std::string message() {
        return  "\n=== Forward XDL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdXdl Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesBlockTransfer: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesLdsTransfer: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesThreadClusterAccessOrder: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesSourceAccessOrder: " + std::string(CHECK_MARK(c6)) + "\n"
               "  SpecifiesGridwiseFwdXdlGemm: " + std::string(CHECK_MARK(c7)) + "\n"
               "  SpecifiesFwdConvSpecialization: " + std::string(CHECK_MARK(c8)) + "\n"
               "  SpecifiesGemmSpecialization: " + std::string(CHECK_MARK(c9)) + "\n"
               "  SpecifiesNumPrefetchStages: " + std::string(CHECK_MARK(c10)) + "\n"
               "  SpecifiesNumGroupsToMerge: " + std::string(CHECK_MARK(c11)) + "\n"
               "  SpecifiesLoopScheduler: " + std::string(CHECK_MARK(c12)) + "\n";
    }
};

template <typename T>
struct FwdWmmaAlgorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesThreadBlock<T>;
    static constexpr bool c3 = SpecifiesBlockTransfer<T>;
    static constexpr bool c4 = SpecifiesLdsTransfer<T>;
    static constexpr bool c5 = SpecifiesThreadClusterAccessOrder<T>;
    static constexpr bool c6 = SpecifiesSourceAccessOrder<T>;
    static constexpr bool c7 = SpecifiesGridwiseWmmaGemm<T>;
    static constexpr bool c8 = SpecifiesFwdConvSpecialization<T>;
    static constexpr bool c9 = SpecifiesGemmSpecialization<T>;
    static constexpr bool c10 = SpecifiesNumPrefetchStages<T>;
    static constexpr bool c11 = SpecifiesLoopScheduler<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11;
    }

    static consteval const std::string message() {
        return "\n=== Forward WMMA Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdWmma Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesBlockTransfer: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesLdsTransfer: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesThreadClusterAccessOrder: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesSourceAccessOrder: " + std::string(CHECK_MARK(c6)) + "\n"
               "  SpecifiesGridwiseWmmaGemm: " + std::string(CHECK_MARK(c7)) + "\n"
               "  SpecifiesFwdConvSpecialization: " + std::string(CHECK_MARK(c8)) + "\n"
               "  SpecifiesGemmSpecialization: " + std::string(CHECK_MARK(c9)) + "\n"
               "  SpecifiesNumPrefetchStages: " + std::string(CHECK_MARK(c10)) + "\n"
               "  SpecifiesLoopScheduler: " + std::string(CHECK_MARK(c11)) + "\n";
    }
};

template <typename T>
struct FwdDlAlgorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesThreadBlock<T>;
    static constexpr bool c3 = SpecifiesFwdConvSpecialization<T>;
    static constexpr bool c4 = SpecifiesGemmSpecialization<T>;
    static constexpr bool c5 = SpecifiesDlThreadConfig<T>;
    static constexpr bool c6 = SpecifiesDlThreadCluster<T>;
    static constexpr bool c7 = SpecifiesDlBlockTransfer<T>;
    static constexpr bool c8 = SpecifiesDlEpilogue<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8;
    }

    static consteval const std::string message() {
        return "\n=== Forward DL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for FwdDl Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesFwdConvSpecialization: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesGemmSpecialization: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesDlThreadConfig: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesDlThreadCluster: " + std::string(CHECK_MARK(c6)) + "\n"
               "  SpecifiesDlBlockTransfer: " + std::string(CHECK_MARK(c7)) + "\n"
               "  SpecifiesDlEpilogue: " + std::string(CHECK_MARK(c8)) + "\n";
    }
};

template <typename T>
struct TileAlgorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesTileThreadBlock<T>;
    static constexpr bool c3 = SpecifiesTileTransfer<T>;
    static constexpr bool c4 = SpecifiesTileConvSpecialization<T>;
    static constexpr bool c5 = SpecifiesTileBlockGemm<T>;
    static constexpr bool c6 = SpecifiesTileOptimizations<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6;
    }

    static consteval const std::string message() {
        return "\n=== CK Tile Algorithm Diagnostic (closest match) ===\n" 
               "Concepts for CK Tile Conv Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesTileThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesTileTransfer: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesTileConvSpecialization: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesTileBlockGemm: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesTileOptimizations: " + std::string(CHECK_MARK(c6)) + "\n";
    }
};

template <typename T>
struct LargeTensorAlgorithm : public FwdXdlAlgorithm<decltype(T::base_algorithm)>
{
    using BaseAlgorithmType = decltype(T::base_algorithm);
    static constexpr bool c13 = SpecifiesLargeTensorSupport<T>;

    static consteval bool is_valid() {
        return FwdXdlAlgorithm<BaseAlgorithmType>::is_valid() && c13;
    }

    static consteval const std::string message() {
        return FwdXdlAlgorithm<BaseAlgorithmType>::message() +
               "  SpecifiesLargeTensorSupport: " + std::string(CHECK_MARK(c13)) + "\n";
    }
};

template <typename T>
struct BwdXdlAlgorithm {
    static constexpr bool c1 = ConvAlgorithmDescriptor<T>;
    static constexpr bool c2 = SpecifiesThreadBlock<T>;
    static constexpr bool c3 = SpecifiesBlockTransfer<T>;
    static constexpr bool c4 = SpecifiesLdsTransfer<T>;
    static constexpr bool c5 = SpecifiesThreadClusterAccessOrder<T>;
    static constexpr bool c6 = SpecifiesSourceAccessOrder<T>;
    static constexpr bool c7 = SpecifiesGridwiseBwdXdlGemm<T>;
    static constexpr bool c8 = SpecifiesBwdWeightConvSpecialization<T>;
    static constexpr bool c9 = SpecifiesTransposeTransfer<T>;

    static consteval bool is_valid() {
        return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9;
    }

    static consteval const std::string message() {
        return "\n=== Backward XDL Algorithm Diagnostic (closest match) ===\n"
               "Concepts for BwdXdl Algorithm:\n"
               "  ConvAlgorithmDescriptor: " + std::string(CHECK_MARK(c1)) + "\n"
               "  SpecifiesThreadBlock: " + std::string(CHECK_MARK(c2)) + "\n"
               "  SpecifiesBlockTransfer: " + std::string(CHECK_MARK(c3)) + "\n"
               "  SpecifiesLdsTransfer: " + std::string(CHECK_MARK(c4)) + "\n"
               "  SpecifiesThreadClusterAccessOrder: " + std::string(CHECK_MARK(c5)) + "\n"
               "  SpecifiesSourceAccessOrder: " + std::string(CHECK_MARK(c6)) + "\n"
               "  SpecifiesGridwiseBwdXdlGemm: " + std::string(CHECK_MARK(c7)) + "\n"
               "  SpecifiesBwdWeightConvSpecialization: " + std::string(CHECK_MARK(c8)) + "\n"
               "  SpecifiesTransposeTransfer: " + std::string(CHECK_MARK(c9)) + "\n";
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
    constexpr int xdl_matches = count_matches_fwd_xdl<AlgoType>();
    constexpr int max_matches = xdl_matches;
    if constexpr (max_matches == xdl_matches) {
        using Alg = BwdXdlAlgorithm<AlgoType>;
        static_assert(Alg::is_valid(), Alg::message());
    } else {
        // This should never happen
        static_assert(false, "Internal Error: No matching algorithm variant found for diagnostics.");
    }
}

} 
