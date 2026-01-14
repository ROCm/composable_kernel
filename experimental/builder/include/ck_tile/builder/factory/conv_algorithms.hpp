// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory {

template <typename T, size_t ThreadClusterRank = 3>
concept SpecifiesTileTransferParameters =
    SpecifiesThreadClusters<T, ThreadClusterRank> && SpecifiesLdsTransfer<T> &&
    SpecifiesThreadClusterAccessOrder<T> && SpecifiesSourceAccessOrder<T>;

// Base algorithm concepts
template <typename T, size_t ThreadClusterRank = 3>
concept ConvAlgorithm = 
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && 
    SpecifiesTileTransferParameters<T, ThreadClusterRank> && 
    SpecifiesWarpGemm<T> && SpecifiesGemmPipeline<T>;;

template <typename T>
concept FwdAlgorithm = ConvAlgorithm<T, 3> && SpecifiesFwdConvSpecialization<T>; 

template <typename T>
concept FwdAlgorithmV3 = FwdAlgorithm<T> && SpecifiesPipelineV3<T>;

template <typename T, size_t ThreadClusterRank = 3>
concept BwdAlgorithm = ConvAlgorithm<T, ThreadClusterRank> &&  SpecifiesBwdWeightConvSpecialization<T>;

template <typename T>
concept BwdAlgorithmV3 = BwdAlgorithm<T, 3> && SpecifiesPipelineV3<T>;

template <typename T>
concept DlAlgorithm = 
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && 
    SpecifiesDlThreadConfig<T> && SpecifiesDlThreadCluster<T> && SpecifiesDlEpilogue<T>;

template <typename T>
concept FwdDlAlgorithmBase = DlAlgorithm<T> && SpecifiesFwdConvSpecialization<T> && 
    SpecifiesDlFwdBlockTransfer<T> && SpecifiesGemmSpecialization<T>;

template <auto Value>
concept FwdXdlAlgorithmBase = FwdAlgorithm<decltype(Value)> && SpecifiesXdl<Value>;

template <auto Value>
concept BwdXdlAlgorithmBase = BwdAlgorithm<decltype(Value), 4> && SpecifiesXdl<Value>;

template <auto Value>
concept BwdXdlV3AlgorithmBase = BwdAlgorithmV3<decltype(Value)> && SpecifiesXdl<Value>;

template <auto Value>
concept BwdWmmaAlgorithmBase = BwdAlgorithm<decltype(Value), 3> && SpecifiesWmma<Value>;

template <auto Value>
concept BwdWmmaV3AlgorithmBase = BwdAlgorithmV3<decltype(Value)> && SpecifiesWmma<Value>;

// Reference algorithm concept
template <auto Value>
concept ReferenceAlgorithm = ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesReferenceAlgorithm<decltype(Value)>;

// Tile-based algorithm concept
template <auto Value>
concept TileAlgorithm = ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesTileThreadBlock<decltype(Value)> &&
                        SpecifiesTileTransfer<decltype(Value)> && SpecifiesTileConvSpecialization<decltype(Value)> &&
                        SpecifiesTileBlockGemm<decltype(Value)> && SpecifiesTileOptimizations<decltype(Value)>;

// FWD XDL algorithm concepts
template <auto Value>
concept FwdXdlAlgorithm = FwdXdlAlgorithmBase<Value> && SpecifiesGenericInstance<decltype(Value)>;

template <auto Value>
concept LargeTensorAlgorithm = FwdXdlAlgorithmBase<Value> && SpecifiesLargeTensorSupport<decltype(Value)>;

template <auto Value>
concept FwdXdlV3Algorithm = FwdAlgorithmV3<decltype(Value)> && SpecifiesXdl<Value>;

// FWD WMMA algorithm concepts
template <auto Value>
concept FwdWmmaAlgorithm = FwdAlgorithm<decltype(Value)> && SpecifiesWmma<Value>;

// FWD DL algorithms
template <auto Value>
concept FwdDlAlgorithm = FwdDlAlgorithmBase<decltype(Value)>;
   
// BWD weight XDL algorithm concepts
template <auto Value>
concept BwdXdlAlgorithm =
    BwdXdlAlgorithmBase<Value> && SpecifiesTransposeTransfer<decltype(Value)> && 
    SpecifiesGenericInstance<decltype(Value)>;

template <auto Value>
concept BwdMultiDXdlAlgorithm = BwdXdlAlgorithmBase<Value> && SpecifiesMultipleDSupport<decltype(Value)>;

template <auto Value>
concept BwdXdlV3Algorithm = BwdXdlV3AlgorithmBase<Value>;

template <auto Value>
concept BwdTwoStageXdlAlgorithm = BwdXdlV3AlgorithmBase<Value> && SpecifiesTransposeTransfer<decltype(Value)> && 
                                   SpecifiesTwoStageSupport<decltype(Value)>;

// BWD weight WMMA algorithm concepts
template <auto Value>
concept BwdWmmaAlgorithm =
    BwdWmmaAlgorithmBase<Value> && SpecifiesNumPrefetchStages<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesGenericInstance<decltype(Value)>;

template <auto Value>
concept BwdMultiDWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<Value> && SpecifiesMultipleDSupport<decltype(Value)>;

template <auto Value>
concept BwdWmmaV3Algorithm =
    BwdWmmaV3AlgorithmBase<Value> && SpecifiesTransposeTransfer<decltype(Value)> && 
    SpecifiesGenericInstance<decltype(Value)>;

template <auto Value>
concept BwdTwoStageWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<Value> && SpecifiesTransposeTransfer<decltype(Value)> &&
                                     SpecifiesTwoStageSupport<decltype(Value)>;

// BWD weight DL algorithms
template <auto Value>
concept BwdDlAlgorithm = DlAlgorithm<decltype(Value)> && SpecifiesBwdWeightConvSpecialization<decltype(Value)> && 
                        SpecifiesDlBwdBlockTransfer<decltype(Value)>;

// Concepts for valid XDL/WMMA algorithms
template <auto Value>
concept SpecifiesValidFwdXdlAlgorithm = 
FwdXdlAlgorithm<Value> || FwdXdlV3Algorithm<Value> || LargeTensorAlgorithm<Value>;

template <auto Value>
concept SpecifiesValidFwdWmmaAlgorithm = FwdWmmaAlgorithm<Value>;

template <auto Value>
concept SpecifiesValidBwdXdlAlgorithm = 
    BwdXdlAlgorithm<Value> || BwdXdlV3Algorithm<Value> || 
    BwdTwoStageXdlAlgorithm<Value> || BwdMultiDXdlAlgorithm<Value>;

template <auto Value>
concept SpecifiesValidBwdWmmaAlgorithm = 
    BwdWmmaAlgorithm<Value> || BwdWmmaV3Algorithm<Value> || 
    BwdTwoStageWmmaV3Algorithm<Value> || BwdMultiDWmmaV3Algorithm<Value>;

template <auto Value>
concept FwdWarpGemmOrDL = SpecifiesValidWarpGemm<Value> || FwdDlAlgorithm<Value>;

template <auto Value>
concept BwdWarpGemmOrDL = SpecifiesValidWarpGemm<Value> || BwdDlAlgorithm<Value>;

} // namespace ck_tile::builder::factory
