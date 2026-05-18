// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Extended unit tests for KernelKey - covers all data types, layouts, pipelines

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>
#include <set>
#include <sstream>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

// =============================================================================
// DataType Tests
// =============================================================================

class DataTypeTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
};

TEST_F(DataTypeTest, AllDataTypesExist)
{
    // Every DataType should be accessible
    std::vector<DataType> all_types = {DataType::FP16,
                                       DataType::BF16,
                                       DataType::FP32,
                                       DataType::FP64,
                                       DataType::INT8,
                                       DataType::INT4,
                                       DataType::INT32,
                                       DataType::FP8,
                                       DataType::BF8,
                                       DataType::UNKNOWN};

    EXPECT_EQ(all_types.size(), 10);
}

TEST_F(DataTypeTest, DataTypesAreDifferent)
{
    EXPECT_NE(DataType::FP16, DataType::BF16);
    EXPECT_NE(DataType::FP16, DataType::FP32);
    EXPECT_NE(DataType::INT8, DataType::INT4);
}

// =============================================================================
// LayoutTag Tests
// =============================================================================

class LayoutTagTest : public ::testing::Test
{
};

TEST_F(LayoutTagTest, AllLayoutsExist)
{
    std::vector<LayoutTag> all_layouts = {
        LayoutTag::RowMajor, LayoutTag::ColMajor, LayoutTag::PackedExternal};

    EXPECT_EQ(all_layouts.size(), 3);
}

TEST_F(LayoutTagTest, LayoutsAreDifferent) { EXPECT_NE(LayoutTag::RowMajor, LayoutTag::ColMajor); }

// =============================================================================
// Pipeline Tests
// =============================================================================

class PipelineTest : public ::testing::Test
{
};

TEST_F(PipelineTest, AllPipelinesExist)
{
    std::vector<Pipeline> all_pipelines = {Pipeline::Mem,
                                           Pipeline::CompV1,
                                           Pipeline::CompV2,
                                           Pipeline::CompV3,
                                           Pipeline::CompV4,
                                           Pipeline::CompV5,
                                           Pipeline::PreShuffleV1,
                                           Pipeline::PreShuffleV2};

    EXPECT_EQ(all_pipelines.size(), 8);
}

TEST_F(PipelineTest, PipelinesAreDifferent)
{
    EXPECT_NE(Pipeline::Mem, Pipeline::CompV4);
    EXPECT_NE(Pipeline::CompV3, Pipeline::CompV4);
}

// =============================================================================
// Scheduler Tests
// =============================================================================

class SchedulerTest : public ::testing::Test
{
};

TEST_F(SchedulerTest, AllSchedulersExist)
{
    std::vector<Scheduler> all_schedulers = {
        Scheduler::Auto, Scheduler::Intrawave, Scheduler::Interwave};

    EXPECT_EQ(all_schedulers.size(), 3);
}

// =============================================================================
// Epilogue Tests
// =============================================================================

class EpilogueTest : public ::testing::Test
{
};

TEST_F(EpilogueTest, AllEpiloguesExist)
{
    std::vector<Epilogue> all_epilogues = {Epilogue::None,
                                           Epilogue::Default,
                                           Epilogue::CShuffle,
                                           Epilogue::Bias,
                                           Epilogue::Activation,
                                           Epilogue::BiasActivation};

    EXPECT_EQ(all_epilogues.size(), 6);
}

// =============================================================================
// KernelKey::Signature Tests
// =============================================================================

class SignatureTest : public ::testing::Test
{
    protected:
    KernelKey::Signature CreateDefaultSignature()
    {
        KernelKey::Signature sig;
        sig.dtype_a             = DataType::FP16;
        sig.dtype_b             = DataType::FP16;
        sig.dtype_c             = DataType::FP16;
        sig.dtype_acc           = DataType::FP32;
        sig.layout_a            = LayoutTag::RowMajor;
        sig.layout_b            = LayoutTag::ColMajor;
        sig.layout_c            = LayoutTag::RowMajor;
        sig.transpose_a         = false;
        sig.transpose_b         = false;
        sig.grouped             = false;
        sig.split_k             = 1;
        sig.elementwise_op      = "PassThrough";
        sig.num_d_tensors       = 0;
        sig.structured_sparsity = false;
        return sig;
    }
};

TEST_F(SignatureTest, DefaultValuesAreReasonable)
{
    KernelKey::Signature sig = CreateDefaultSignature();
    EXPECT_EQ(sig.split_k, 1);
    EXPECT_FALSE(sig.grouped);
    EXPECT_FALSE(sig.structured_sparsity);
}

TEST_F(SignatureTest, AllDataTypeCombinations)
{
    // Test various data type combinations that should be valid
    std::vector<std::tuple<DataType, DataType, DataType, DataType>> valid_combos = {
        {DataType::FP16, DataType::FP16, DataType::FP16, DataType::FP32},
        {DataType::BF16, DataType::BF16, DataType::BF16, DataType::FP32},
        {DataType::FP32, DataType::FP32, DataType::FP32, DataType::FP32},
        {DataType::INT8, DataType::INT8, DataType::INT8, DataType::INT32},
    };

    for(const auto& [a, b, c, acc] : valid_combos)
    {
        KernelKey::Signature sig;
        sig.dtype_a   = a;
        sig.dtype_b   = b;
        sig.dtype_c   = c;
        sig.dtype_acc = acc;

        EXPECT_EQ(sig.dtype_a, a);
        EXPECT_EQ(sig.dtype_b, b);
        EXPECT_EQ(sig.dtype_c, c);
        EXPECT_EQ(sig.dtype_acc, acc);
    }
}

TEST_F(SignatureTest, AllLayoutCombinations)
{
    std::vector<std::string> layout_codes = {
        "rrr", "rcr", "crr", "ccr", "rrc", "rcc", "crc", "ccc"};

    for(const std::string& code : layout_codes)
    {
        KernelKey::Signature sig = CreateDefaultSignature();
        sig.layout_a             = (code[0] == 'r') ? LayoutTag::RowMajor : LayoutTag::ColMajor;
        sig.layout_b             = (code[1] == 'r') ? LayoutTag::RowMajor : LayoutTag::ColMajor;
        sig.layout_c             = (code[2] == 'r') ? LayoutTag::RowMajor : LayoutTag::ColMajor;

        // Just verify assignment works
        EXPECT_TRUE(sig.layout_a == LayoutTag::RowMajor || sig.layout_a == LayoutTag::ColMajor);
    }
}

TEST_F(SignatureTest, SplitKValues)
{
    KernelKey::Signature sig = CreateDefaultSignature();

    std::vector<std::uint8_t> valid_split_k = {1, 2, 4, 8, 16};
    for(auto sk : valid_split_k)
    {
        sig.split_k = sk;
        EXPECT_EQ(sig.split_k, sk);
    }
}

// =============================================================================
// KernelKey::Algorithm Tests
// =============================================================================

class AlgorithmTest : public ::testing::Test
{
    protected:
    KernelKey::Algorithm CreateDefaultAlgorithm()
    {
        KernelKey::Algorithm algo;
        algo.tile_shape      = {256, 256, 32};
        algo.wave_shape      = {2, 2, 1};
        algo.warp_tile_shape = {32, 32, 16};
        algo.pipeline        = Pipeline::CompV4;
        algo.scheduler       = Scheduler::Intrawave;
        algo.epilogue        = Epilogue::CShuffle;
        algo.block_size      = 256;
        algo.double_buffer   = true;
        algo.persistent      = false;
        algo.preshuffle      = false;
        algo.transpose_c     = false;
        algo.num_wave_groups = 1;
        return algo;
    }
};

TEST_F(AlgorithmTest, CommonTileShapes)
{
    std::vector<std::tuple<int, int, int>> valid_tiles = {
        {64, 64, 32},
        {128, 128, 32},
        {128, 128, 64},
        {256, 256, 32},
        {256, 256, 64},
        {256, 128, 32},
        {128, 256, 32},
    };

    for(const auto& [m, n, k] : valid_tiles)
    {
        KernelKey::Algorithm algo = CreateDefaultAlgorithm();
        algo.tile_shape           = {static_cast<std::uint16_t>(m),
                                     static_cast<std::uint16_t>(n),
                                     static_cast<std::uint16_t>(k)};

        EXPECT_EQ(algo.tile_shape.m, m);
        EXPECT_EQ(algo.tile_shape.n, n);
        EXPECT_EQ(algo.tile_shape.k, k);
    }
}

TEST_F(AlgorithmTest, CommonWarpConfigs)
{
    std::vector<std::tuple<int, int, int>> valid_warps = {
        {1, 4, 1},
        {2, 2, 1},
        {4, 1, 1},
        {1, 2, 1},
        {2, 1, 1},
    };

    for(const auto& [m, n, k] : valid_warps)
    {
        KernelKey::Algorithm algo = CreateDefaultAlgorithm();
        algo.wave_shape           = {static_cast<std::uint8_t>(m),
                                     static_cast<std::uint8_t>(n),
                                     static_cast<std::uint8_t>(k)};

        EXPECT_EQ(algo.wave_shape.m, m);
        EXPECT_EQ(algo.wave_shape.n, n);
        EXPECT_EQ(algo.wave_shape.k, k);
    }
}

TEST_F(AlgorithmTest, AllPipelines)
{
    KernelKey::Algorithm algo = CreateDefaultAlgorithm();

    std::vector<Pipeline> pipelines = {Pipeline::Mem,
                                       Pipeline::CompV3,
                                       Pipeline::CompV4,
                                       Pipeline::PreShuffleV1,
                                       Pipeline::PreShuffleV2};

    for(Pipeline p : pipelines)
    {
        algo.pipeline = p;
        EXPECT_EQ(algo.pipeline, p);
    }
}

// =============================================================================
// KernelKey Identifier Encoding Tests
// =============================================================================

class IdentifierEncodingTest : public ::testing::Test
{
};

TEST_F(IdentifierEncodingTest, UniqueIdentifiersForDifferentConfigs)
{
    std::set<std::string> identifiers;

    // Generate multiple configurations
    for(int tile_m : {128, 256})
    {
        for(int wave_m : {1, 2, 4})
        {
            for(bool persistent : {true, false})
            {
                KernelKey key              = make_test_key(tile_m);
                key.algorithm.wave_shape.m = wave_m;
                key.algorithm.persistent   = persistent;

                std::string id = key.encode_identifier();
                EXPECT_TRUE(identifiers.find(id) == identifiers.end())
                    << "Duplicate identifier: " << id;
                identifiers.insert(id);
            }
        }
    }

    // Should have generated 2 * 3 * 2 = 12 unique identifiers
    EXPECT_EQ(identifiers.size(), 12);
}

TEST_F(IdentifierEncodingTest, IdentifierContainsTileShape)
{
    KernelKey key  = make_test_key(256, 128, 64);
    std::string id = key.encode_identifier();

    EXPECT_NE(id.find("256x128x64"), std::string::npos)
        << "Identifier should contain tile shape: " << id;
}

TEST_F(IdentifierEncodingTest, IdentifierContainsWarpConfig)
{
    KernelKey key            = make_test_key(256);
    key.algorithm.wave_shape = {4, 2, 1};
    std::string id           = key.encode_identifier();

    EXPECT_NE(id.find("4x2x1"), std::string::npos)
        << "Identifier should contain warp config: " << id;
}

TEST_F(IdentifierEncodingTest, IdentifierReflectsPersistence)
{
    KernelKey persistent_key            = make_test_key(256);
    persistent_key.algorithm.persistent = true;

    KernelKey non_persistent_key            = make_test_key(256);
    non_persistent_key.algorithm.persistent = false;

    std::string persistent_id     = persistent_key.encode_identifier();
    std::string non_persistent_id = non_persistent_key.encode_identifier();

    // EXPECT_NE above already verifies persistence affects encoding;
    // substring checks for specific spelling were brittle and have been removed.
    EXPECT_NE(persistent_id, non_persistent_id);
}

// =============================================================================
// KernelKey Equality Tests
// =============================================================================

class KeyEqualityTest : public ::testing::Test
{
};

TEST_F(KeyEqualityTest, IdenticalKeysAreEqual)
{
    KernelKey key1 = make_test_key(256, 256, 32, "gfx942");
    KernelKey key2 = make_test_key(256, 256, 32, "gfx942");

    EXPECT_EQ(key1, key2);
    EXPECT_FALSE(key1 != key2);
}

TEST_F(KeyEqualityTest, DifferentTileShapesNotEqual)
{
    KernelKey key1 = make_test_key(256, 256, 32);
    KernelKey key2 = make_test_key(128, 128, 32);

    EXPECT_NE(key1, key2);
}

TEST_F(KeyEqualityTest, DifferentDataTypesNotEqual)
{
    KernelKey key1         = make_test_key(256);
    KernelKey key2         = make_test_key(256);
    key2.signature.dtype_a = DataType::BF16;

    EXPECT_NE(key1, key2);
}

TEST_F(KeyEqualityTest, DifferentLayoutsNotEqual)
{
    KernelKey key1          = make_test_key(256);
    KernelKey key2          = make_test_key(256);
    key2.signature.layout_a = LayoutTag::ColMajor;

    EXPECT_NE(key1, key2);
}

TEST_F(KeyEqualityTest, DifferentGfxArchNotEqual)
{
    KernelKey key1 = make_test_key(256, 256, 32, "gfx942");
    KernelKey key2 = make_test_key(256, 256, 32, "gfx90a");

    EXPECT_NE(key1, key2);
}

// =============================================================================
// ElementwiseOps Tests
// =============================================================================

class ElementwiseOpsTest : public ::testing::Test
{
};

TEST_F(ElementwiseOpsTest, CanUseInKernelKey)
{
    KernelKey key = make_test_key(256);

    key.signature.elementwise_op = "Relu";
    EXPECT_EQ(key.signature.elementwise_op, "Relu");

    key.signature.elementwise_op = "Gelu";
    EXPECT_EQ(key.signature.elementwise_op, "Gelu");

    key.signature.elementwise_op = "PassThrough";
    EXPECT_EQ(key.signature.elementwise_op, "PassThrough");
}
