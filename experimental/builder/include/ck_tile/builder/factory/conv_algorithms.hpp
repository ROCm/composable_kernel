// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory {

// Base algorithm concepts
template <typename T>
concept FwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesFwdConvSpecialization<T> && SpecifiesGemmSpecialization<T> &&
    SpecifiesNumPrefetchStages<T> && SpecifiesNumGroupsToMerge<T> && SpecifiesLoopScheduler<T> &&
    SpecifiesXdl<T>;

template <typename T>
concept BwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution4D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesXdl<T>;

template <typename T>
concept BwdXdlV3AlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesGemmPipeline<T> && SpecifiesXdl<T>;

template <typename T>
concept BwdWmmaAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesWmma<T>;

template <typename T>
concept BwdWmmaV3AlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesGemmPipeline<T> && SpecifiesWmma<T>;

// Reference algorithm concept
template <typename T>
concept ReferenceAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesReferenceAlgorithm<T>;

// Tile-based algorithm concept
template <typename T>
concept TileAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesTileThreadBlock<T> &&
                        SpecifiesTileTransfer<T> && SpecifiesTileConvSpecialization<T> &&
                        SpecifiesTileBlockGemm<T> && SpecifiesTileOptimizations<T>;

// FWD XDL algorithm concepts
template <typename T>
concept FwdXdlAlgorithm = FwdXdlAlgorithmBase<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept LargeTensorAlgorithm = FwdXdlAlgorithmBase<T> && SpecifiesLargeTensorSupport<T>;

template <typename T>
concept FwdXdlV3Algorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesFwdConvSpecialization<T> && SpecifiesGemmSpecialization<T> && SpecifiesGemmPipeline<T> && SpecifiesXdl<T>;

// FWD WMMA algorithm concepts
template <typename T>
concept FwdWmmaAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesThreadDistribution3D<T> &&
    SpecifiesLdsTransfer<T> && SpecifiesThreadClusterAccessOrder<T> &&
    SpecifiesSourceAccessOrder<T> && SpecifiesWarpGemm<T> &&
    SpecifiesFwdConvSpecialization<T> && SpecifiesGemmSpecialization<T> &&
    SpecifiesNumPrefetchStages<T> && SpecifiesLoopScheduler<T> && SpecifiesGemmPipeline<T> && SpecifiesWmma<T>;

// FWD DL algorithms
template <typename T>
concept FwdDlAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesDlThreadConfig<T> && SpecifiesDlThreadCluster<T> &&
    SpecifiesDlFwdBlockTransfer<T> && SpecifiesDlEpilogue<T>;

// BWD weight XDL algorithm concepts
template <typename T>
concept BwdXdlAlgorithm =
    BwdXdlAlgorithmBase<T> && SpecifiesTransposeTransfer<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdMultiDXdlAlgorithm = BwdXdlAlgorithmBase<T> && SpecifiesMultipleDSupport<T>;

template <typename T>
concept BwdXdlV3Algorithm = BwdXdlV3AlgorithmBase<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdTwoStageXdlAlgorithm = BwdXdlV3AlgorithmBase<T> && SpecifiesTransposeTransfer<T> &&
                                  SpecifiesGemmBatchOptions<T> && SpecifiesTwoStageSupport<T>;

// BWD weight WMMA algorithm concepts
template <typename T>
concept BwdWmmaAlgorithm =
    BwdWmmaAlgorithmBase<T> && SpecifiesNumPrefetchStages<T> && SpecifiesLoopScheduler<T> &&
    SpecifiesGemmPipeline<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdMultiDWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<T> && SpecifiesMultipleDSupport<T>;

template <typename T>
concept BwdWmmaV3Algorithm =
    BwdWmmaV3AlgorithmBase<T> && SpecifiesTransposeTransfer<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdTwoStageWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<T> && SpecifiesTransposeTransfer<T> &&
                                     SpecifiesGemmBatchOptions<T> && SpecifiesTwoStageSupport<T>;

// BWD weight DL algorithms
template <typename T>
concept BwdDlAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesDlThreadConfig<T> &&
    SpecifiesDlThreadCluster<T> && SpecifiesDlBwdBlockTransfer<T> && SpecifiesDlEpilogue<T>;

} // namespace ck_tile::builder::factory
