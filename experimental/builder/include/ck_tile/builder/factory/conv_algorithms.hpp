// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory {

// Base algorithm concepts
template <auto Value>
concept FwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesFwdConvSpecialization<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesXdl<Value>;

template <auto Value>
concept BwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution4D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesBwdWeightConvSpecialization<decltype(Value)> && 
    SpecifiesXdl<Value>;

template <auto Value>
concept BwdXdlV3AlgorithmBase =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesBwdWeightConvSpecialization<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesXdl<Value> && SpecifiesPipelineV3<decltype(Value)>;

template <auto Value>
concept BwdWmmaAlgorithmBase =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesBwdWeightConvSpecialization<decltype(Value)> && 
    SpecifiesWmma<Value>;

template <auto Value>
concept BwdWmmaV3AlgorithmBase =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesBwdWeightConvSpecialization<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesWmma<Value> && SpecifiesPipelineV3<decltype(Value)>;

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
concept FwdXdlV3Algorithm =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesFwdConvSpecialization<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesXdl<Value> && SpecifiesPipelineV3<decltype(Value)>;

// FWD WMMA algorithm concepts
template <auto Value>
concept FwdWmmaAlgorithm =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesThreadDistribution3D<decltype(Value)> && SpecifiesLdsTransfer<decltype(Value)> && 
    SpecifiesThreadClusterAccessOrder<decltype(Value)> && SpecifiesSourceAccessOrder<decltype(Value)> && 
    SpecifiesWarpGemm<decltype(Value)> && SpecifiesFwdConvSpecialization<decltype(Value)> && 
    SpecifiesGemmPipeline<decltype(Value)> && SpecifiesWmma<Value>;

// FWD DL algorithms
template <auto Value>
concept FwdDlAlgorithm =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> && 
    SpecifiesFwdConvSpecialization<decltype(Value)> && SpecifiesGemmSpecialization<decltype(Value)> && 
    SpecifiesDlThreadConfig<decltype(Value)> && SpecifiesDlThreadCluster<decltype(Value)> &&
    SpecifiesDlFwdBlockTransfer<decltype(Value)> && SpecifiesDlEpilogue<decltype(Value)>;

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
concept BwdDlAlgorithm =
    ConvAlgorithmDescriptor<decltype(Value)> && SpecifiesThreadBlock<decltype(Value)> &&
    SpecifiesBwdWeightConvSpecialization<decltype(Value)> && SpecifiesDlThreadConfig<decltype(Value)> &&
    SpecifiesDlThreadCluster<decltype(Value)> && SpecifiesDlBwdBlockTransfer<decltype(Value)> && 
    SpecifiesDlEpilogue<decltype(Value)>;

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
