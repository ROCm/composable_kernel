// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory {

// Base algorithm concepts
template <typename T, size_t ThreadClusterRank = 3>
concept TileTransferParameters =
    SpecifiesBlockTransfer<T, ThreadClusterRank> && SpecifiesLdsTransfer<T> &&
    SpecifiesThreadClusterArrangeOrder<T> && SpecifiesSourceAccessOrder<T>;

template <typename T>
concept SpecifiesTileTransferParameters3D = TileTransferParameters<T, 3>;

template <typename T>
concept SpecifiesTileTransferParameters4D = TileTransferParameters<T, 4>;

template <typename T>
concept FwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseFwdXdlGemm<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesNumPrefetchStages<T> &&
    SpecifiesNumGroupsToMerge<T> && SpecifiesLoopScheduler<T>;

template <typename T>
concept BwdXdlAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> &&
    (SpecifiesTileTransferParameters4D<T> || SpecifiesTileTransferParameters3D<T>) &&
    (SpecifiesGridwiseBwdXdlGemm<T> || SpecifiesGridwiseBwdDataXdlGemm<T>) &&
    (SpecifiesBwdWeightConvSpecialization<T> || SpecifiesBwdDataConvSpecialization<T>);

template <typename T>
concept BwdXdlV3AlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    (SpecifiesGridwiseBwdXdlGemm<T> || SpecifiesGridwiseBwdDataXdlGemm<T>) &&
    (SpecifiesBwdWeightConvSpecialization<T> || SpecifiesBwdDataConvSpecialization<T>) &&
    SpecifiesBlockGemm<T> && SpecifiesNumGroupsToMerge<T>;

template <typename T>
concept BwdWmmaAlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseWmmaGemm<T> &&
    (SpecifiesBwdWeightConvSpecialization<T> || SpecifiesBwdDataConvSpecialization<T>);

template <typename T>
concept BwdWmmaV3AlgorithmBase =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseWmmaGemm<T> &&
    (SpecifiesBwdWeightConvSpecialization<T> || SpecifiesBwdDataConvSpecialization<T>) &&
    SpecifiesBlockGemm<T>;

// Reference algorithm concept
template <typename T>
concept ReferenceAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesReferenceAlgorithm<T>;

// Tile-based algorithm concept
template <typename T>
concept TileAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesTileThreadBlock<T> &&
                        SpecifiesTileTransfer<T> && SpecifiesTileConvSpecialization<T> &&
                        SpecifiesTileBlockGemm<T> && SpecifiesTileOptimizations<T>;

// Depthwise tile-based algorithm concept (no GEMM — direct spatial pipeline)
template <typename T>
concept DepthwiseAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesDepthwiseConvParams<T>;

// FWD XDL algorithm concepts
template <typename T>
concept FwdXdlAlgorithm = FwdXdlAlgorithmBase<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept LargeTensorAlgorithm = FwdXdlAlgorithmBase<T> && SpecifiesLargeTensorSupport<T>;

template <typename T>
concept FwdXdlV3Algorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseFwdXdlGemm<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesBlockGemm<T> && SpecifiesNumGroupsToMerge<T>;

// FWD WMMA V3 algorithm concept
template <typename T>
concept FwdWmmaV3Algorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseWmmaGemm<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesBlockGemm<T> && SpecifiesNumGroupsToMerge<T>;

// FWD WMMA algorithm concepts
template <typename T>
concept FwdWmmaAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesTileTransferParameters3D<T> &&
    SpecifiesGridwiseWmmaGemm<T> && SpecifiesFwdConvSpecialization<T> &&
    SpecifiesGemmSpecialization<T> && SpecifiesNumPrefetchStages<T> && SpecifiesLoopScheduler<T> &&
    SpecifiesGridwiseGemmPipeline<T>;

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
    SpecifiesGridwiseGemmPipeline<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdMultiDWmmaAlgorithm = BwdWmmaAlgorithmBase<T> && SpecifiesMultipleDSupport<T>;

template <typename T>
concept BwdMultiDWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<T> && SpecifiesMultipleDSupport<T>;

template <typename T>
concept BwdWmmaV3Algorithm =
    BwdWmmaV3AlgorithmBase<T> && SpecifiesTransposeTransfer<T> && SpecifiesGenericInstance<T>;

template <typename T>
concept BwdTwoStageWmmaV3Algorithm = BwdWmmaV3AlgorithmBase<T> && SpecifiesTransposeTransfer<T> &&
                                     SpecifiesGemmBatchOptions<T> && SpecifiesTwoStageSupport<T>;

// BWD weigth DL algorithms
template <typename T>
concept BwdDlAlgorithm =
    ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> &&
    SpecifiesBwdWeightConvSpecialization<T> && SpecifiesDlThreadConfig<T> &&
    SpecifiesDlThreadCluster<T> && SpecifiesDlBwdBlockTransfer<T> && SpecifiesDlEpilogue<T>;

} // namespace ck_tile::builder::factory
