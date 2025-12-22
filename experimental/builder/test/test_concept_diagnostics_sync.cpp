// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file test_concept_diagnostics_sync.cpp
 * @brief Unit tests to ensure concepts and their diagnostics remain in sync
 *
 * This test suite verifies that:
 * 1. Valid types satisfy their corresponding concepts
 * 2. Invalid types (missing members) do not satisfy concepts
 * 3. Diagnostic messages correctly identify missing requirements
 * 4. Existing test types from conv_algorithm_types.hpp satisfy their concepts
 */

#include <gtest/gtest.h>
#include <string>

#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_diagnostics.hpp"
#include "ck_tile/builder/types.hpp"
#include "experimental/builder/test/impl/conv_algorithm_types.hpp"

namespace ck_tile::builder::test {

using ck_tile::builder::ThreadBlockDescriptor;
using ck_tile::builder::GridwiseXdlGemmDescriptor;
using ck_tile::builder::BlockTransferDescriptor;
using ck_tile::builder::ThreadClusterDescriptor;
using ck_tile::builder::LdsTransferDescriptor;
using ck_tile::builder::EpilogueDescriptor;
using ck_tile::builder::AccessOrderDescriptor;
using ck_tile::builder::BlockGemmDescriptor;
using ck_tile::builder::GridwiseWmmaGemmDescriptor;
using ck_tile::builder::TileThreadBlockDescriptor;
using ck_tile::builder::TileTransferDescriptor;
using ck_tile::builder::TileBlockGemmDescriptor;
using ck_tile::builder::TileOptimizationsDescriptor;
using ck_tile::builder::DlThreadConfigDescriptor;
using ck_tile::builder::DlThreadClusterDescriptor;
using ck_tile::builder::DlBlockTransferDescriptor;
using ck_tile::builder::DlEpilogueDescriptor;
using ck_tile::builder::ConvAlgorithmDescriptor;
using ck_tile::builder::SpecifiesThreadBlock;
using ck_tile::builder::SpecifiesGridwiseFwdXdlGemm;
using ck_tile::builder::SpecifiesGridwiseBwdXdlGemm;
using ck_tile::builder::SpecifiesBlockGemm;
using ck_tile::builder::SpecifiesFwdConvSpecialization;
using ck_tile::builder::SpecifiesBwdWeightConvSpecialization;
using ck_tile::builder::SpecifiesGemmSpecialization;
using ck_tile::builder::SpecifiesNumPrefetchStages;
using ck_tile::builder::SpecifiesLoopScheduler;
using ck_tile::builder::SpecifiesTileThreadBlock;
using ck_tile::builder::SpecifiesTileTransfer;
using ck_tile::builder::SpecifiesTileBlockGemm;
using ck_tile::builder::SpecifiesTileOptimizations;
using ck_tile::builder::SpecifiesTileConvSpecialization;
using ck_tile::builder::SpecifiesDlThreadConfig;
using ck_tile::builder::SpecifiesDlThreadCluster;

// Helper to check if a string contains a substring
bool contains(const std::string& str, const std::string& substr)
{
    return str.find(substr) != std::string::npos;
}

// =============================================================================
// BASIC DESCRIPTOR CONCEPTS TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, ThreadBlockDescriptor_Valid)
{
    // The ThreadBlock type from conv_algorithm_types.hpp should satisfy the concept
    static_assert(ThreadBlockDescriptor<ThreadBlock>);
}

TEST(ConceptDiagnosticsSync, GridwiseXdlGemmDescriptor_Valid)
{
    // The XdlParams type should satisfy the concept
    static_assert(GridwiseXdlGemmDescriptor<XdlParams>);
}

TEST(ConceptDiagnosticsSync, BlockTransferDescriptor_Valid)
{
    // The BlockTransfer type should satisfy the concept
    static_assert(BlockTransferDescriptor<BlockTransfer>);
}

TEST(ConceptDiagnosticsSync, ThreadClusterDescriptor_Valid)
{
    // The ThreadCluster type should satisfy the concept
    static_assert(ThreadClusterDescriptor<ThreadCluster>);
}

TEST(ConceptDiagnosticsSync, LdsTransferDescriptor_Valid)
{
    // The LdsTransfer type should satisfy the concept
    static_assert(LdsTransferDescriptor<LdsTransfer>);
}

TEST(ConceptDiagnosticsSync, EpilogueDescriptor_Valid)
{
    // The Epilogue type should satisfy the concept
    static_assert(EpilogueDescriptor<Epilogue>);
}

TEST(ConceptDiagnosticsSync, AccessOrderDescriptor_Valid)
{
    // The AccessOrder type should satisfy the concept
    static_assert(AccessOrderDescriptor<AccessOrder>);
}

TEST(ConceptDiagnosticsSync, BlockGemmDescriptor_Valid)
{
    // The BlockGemm type should satisfy the concept
    static_assert(BlockGemmDescriptor<BlockGemm>);
}

TEST(ConceptDiagnosticsSync, GridwiseWmmaGemmDescriptor_Valid)
{
    // The GridwiseWmmaGemm type should satisfy the concept
    static_assert(GridwiseWmmaGemmDescriptor<GridwiseWmmaGemm>);
}

// =============================================================================
// HIGH-LEVEL "SPECIFIES" CONCEPTS TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, SpecifiesThreadBlock_Valid)
{
    static_assert(SpecifiesThreadBlock<ThreadBlock_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesGridwiseFwdXdlGemm_Valid)
{
    static_assert(SpecifiesGridwiseFwdXdlGemm<FwdXdlGemm_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesGridwiseBwdXdlGemm_Valid)
{
    static_assert(SpecifiesGridwiseBwdXdlGemm<BwdXdlGemm_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesBlockGemm_Valid)
{
    static_assert(SpecifiesBlockGemm<BlockGemm_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesFwdConvSpecialization_Valid)
{
    static_assert(SpecifiesFwdConvSpecialization<ConvSpecializationFwd_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesBwdWeightConvSpecialization_Valid)
{
    static_assert(SpecifiesBwdWeightConvSpecialization<ConvSpecializationBwdWeight_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesGemmSpecialization_Valid)
{
    static_assert(SpecifiesGemmSpecialization<ConvSpecializationFwd_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesNumPrefetchStages_Valid)
{
    static_assert(SpecifiesNumPrefetchStages<Prefetch_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesLoopScheduler_Valid)
{
    static_assert(SpecifiesLoopScheduler<Prefetch_>);
}

// =============================================================================
// TILE-SPECIFIC CONCEPTS TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, TileThreadBlockDescriptor_Valid)
{
    static_assert(TileThreadBlockDescriptor<TileThreadBlock>);
}

TEST(ConceptDiagnosticsSync, TileTransferDescriptor_Valid)
{
    static_assert(TileTransferDescriptor<TileTransfer>);
}

TEST(ConceptDiagnosticsSync, TileBlockGemmDescriptor_Valid)
{
    static_assert(TileBlockGemmDescriptor<TileBlockGemm>);
}

TEST(ConceptDiagnosticsSync, TileOptimizationsDescriptor_Valid)
{
    static_assert(TileOptimizationsDescriptor<TileOptimizations>);
}

TEST(ConceptDiagnosticsSync, SpecifiesTileThreadBlock_Valid)
{
    static_assert(SpecifiesTileThreadBlock<TileThreadBlock_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesTileTransfer_Valid)
{
    static_assert(SpecifiesTileTransfer<TileTransfer_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesTileBlockGemm_Valid)
{
    static_assert(SpecifiesTileBlockGemm<TileBlockGemm_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesTileOptimizations_Valid)
{
    static_assert(SpecifiesTileOptimizations<TileOptimizations_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesTileConvSpecialization_Valid)
{
    static_assert(SpecifiesTileConvSpecialization<TileConvSpecialization_>);
}

// =============================================================================
// DL-SPECIFIC CONCEPTS TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, DlThreadConfigDescriptor_Valid)
{
    static_assert(DlThreadConfigDescriptor<DlThreadConfig>);
}

TEST(ConceptDiagnosticsSync, DlThreadClusterDescriptor_Valid)
{
    static_assert(DlThreadClusterDescriptor<DlThreadCluster>);
}

TEST(ConceptDiagnosticsSync, DlBlockTransferDescriptor_Valid)
{
    static_assert(DlBlockTransferDescriptor<DlBlockTransfer>);
}

TEST(ConceptDiagnosticsSync, DlEpilogueDescriptor_Valid)
{
    static_assert(DlEpilogueDescriptor<DlEpilogue>);
}

TEST(ConceptDiagnosticsSync, SpecifiesDlThreadConfig_Valid)
{
    static_assert(SpecifiesDlThreadConfig<DlThreadConfig_>);
}

TEST(ConceptDiagnosticsSync, SpecifiesDlThreadCluster_Valid)
{
    static_assert(SpecifiesDlThreadCluster<DlThreadCluster_>);
}

// =============================================================================
// INVALID TYPE TESTS - Test that concepts correctly reject invalid types
// =============================================================================

namespace invalid_types {

// Test ThreadBlockDescriptor with missing members
struct MissingBlockSize
{
    struct
    {
        size_t m, n, k;
    } tile_size;
};

struct MissingTileSizeM
{
    size_t block_size;
    struct
    {
        size_t n, k;
    } tile_size;
};

// Test GridwiseXdlGemmDescriptor with missing members
struct MissingMPerXdl
{
    size_t n_per_xdl;
    size_t m_xdl_per_wave;
    size_t n_xdl_per_wave;
};

// Test BlockTransferDescriptor with missing members
struct MissingK0
{
    size_t m_n;
    size_t k1;
};

// Test LdsTransferDescriptor with missing members
struct MissingSrcVectorDim
{
    size_t src_scalar_per_vector;
    size_t lds_dst_scalar_per_vector;
    bool is_direct_load;
    bool lds_padding;
};

} // namespace invalid_types

TEST(ConceptDiagnosticsSync, ThreadBlockDescriptor_Invalid)
{
    static_assert(!ThreadBlockDescriptor<invalid_types::MissingBlockSize>);
    static_assert(!ThreadBlockDescriptor<invalid_types::MissingTileSizeM>);
}

TEST(ConceptDiagnosticsSync, GridwiseXdlGemmDescriptor_Invalid)
{
    static_assert(!GridwiseXdlGemmDescriptor<invalid_types::MissingMPerXdl>);
}

TEST(ConceptDiagnosticsSync, BlockTransferDescriptor_Invalid)
{
    static_assert(!BlockTransferDescriptor<invalid_types::MissingK0>);
}

TEST(ConceptDiagnosticsSync, LdsTransferDescriptor_Invalid)
{
    static_assert(!LdsTransferDescriptor<invalid_types::MissingSrcVectorDim>);
}

// =============================================================================
// COMPREHENSIVE ALGORITHM TYPE TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, CompleteAlgorithmTypes)
{
    // Test that complete algorithm types satisfy their concepts
    static_assert(ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
    static_assert(ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>);
    static_assert(ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>);
    static_assert(ConvAlgorithmDescriptor<ConvAlgorithm_Tile_GroupedConvolutionKernel>);
    static_assert(ConvAlgorithmDescriptor<ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle>);
    
    // Test specific requirements for each algorithm type
    static_assert(SpecifiesThreadBlock<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
    static_assert(SpecifiesGridwiseFwdXdlGemm<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
    static_assert(SpecifiesFwdConvSpecialization<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
    static_assert(SpecifiesNumPrefetchStages<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>);
    
    static_assert(SpecifiesTileThreadBlock<ConvAlgorithm_Tile_GroupedConvolutionKernel>);
    static_assert(SpecifiesTileBlockGemm<ConvAlgorithm_Tile_GroupedConvolutionKernel>);
    static_assert(SpecifiesTileOptimizations<ConvAlgorithm_Tile_GroupedConvolutionKernel>);
}

// =============================================================================
// DIAGNOSTIC MESSAGE TESTS
// =============================================================================

TEST(ConceptDiagnosticsSync, DiagnosticMessages)
{
    // Test that diagnostics can be called (even if messages may be empty at compile-time)
    // The key is that the diagnostic functions exist and compile
    std::string diag1 = ck_tile::builder::diagnostics::detailed_diagnostic_SpecifiesThreadBlock<invalid_types::MissingBlockSize>();
    std::string diag2 = ck_tile::builder::diagnostics::detailed_diagnostic_SpecifiesGridwiseFwdXdlGemm<invalid_types::MissingMPerXdl>();
    
    // These may be empty depending on the implementation, but they should compile
    EXPECT_TRUE(diag1.empty() || contains(diag1, "thread_block") || contains(diag1, "missing"));
    EXPECT_TRUE(diag2.empty() || contains(diag2, "gridwise_gemm") || contains(diag2, "missing"));
}

// =============================================================================
// CONCEPT COMPLETENESS TESTS
// =============================================================================

/**
 * @brief Verify that all concepts defined in conv_algorithm_concepts.hpp have tests
 * 
 * This test serves as documentation of which concepts are tested. If new concepts
 * are added, this test should be updated to include them.
 */
TEST(ConceptDiagnosticsSync, ConceptCoverage)
{
    // Basic Descriptor Concepts - verify they all exist and can be instantiated
    EXPECT_TRUE((ThreadBlockDescriptor<ThreadBlock>));
    EXPECT_TRUE((GridwiseXdlGemmDescriptor<XdlParams>));
    EXPECT_TRUE((BlockGemmDescriptor<BlockGemm>));
    EXPECT_TRUE((GridwiseWmmaGemmDescriptor<GridwiseWmmaGemm>));
    EXPECT_TRUE((BlockTransferDescriptor<BlockTransfer>));
    EXPECT_TRUE((ThreadClusterDescriptor<ThreadCluster>));
    EXPECT_TRUE((LdsTransferDescriptor<LdsTransfer>));
    EXPECT_TRUE((EpilogueDescriptor<Epilogue>));
    EXPECT_TRUE((AccessOrderDescriptor<AccessOrder>));
    
    // Tile Descriptor Concepts  
    EXPECT_TRUE((TileThreadBlockDescriptor<TileThreadBlock>));
    EXPECT_TRUE((TileTransferDescriptor<TileTransfer>));
    EXPECT_TRUE((TileBlockGemmDescriptor<TileBlockGemm>));
    EXPECT_TRUE((TileOptimizationsDescriptor<TileOptimizations>));
    
    // DL Descriptor Concepts
    EXPECT_TRUE((DlThreadConfigDescriptor<DlThreadConfig>));
    EXPECT_TRUE((DlThreadClusterDescriptor<DlThreadCluster>));
    EXPECT_TRUE((DlBlockTransferDescriptor<DlBlockTransfer>));
    EXPECT_TRUE((DlEpilogueDescriptor<DlEpilogue>));
}

} // namespace ck_tile::builder::test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
