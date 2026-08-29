// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/gemm_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::AddOp;
using ::rocm_ck::DataType;
using ::rocm_ck::EpilogueOp;
using ::rocm_ck::FastGeluOp;
using ::rocm_ck::GeluOp;
using ::rocm_ck::GemmAlgorithm;
using ::rocm_ck::GemmOp;
using ::rocm_ck::GemmSpec;
using ::rocm_ck::GpuTarget;
using ::rocm_ck::isValidWaveTile;
using ::rocm_ck::Layout;
using ::rocm_ck::makeSpec;
using ::rocm_ck::MulOp;
using ::rocm_ck::Pipeline;
using ::rocm_ck::PipelineScheduler;
using ::rocm_ck::Quantization;
using ::rocm_ck::ReluOp;
using ::rocm_ck::SigmoidOp;
using ::rocm_ck::Signature;
using ::rocm_ck::SiluOp;
using ::rocm_ck::StoreStrategy;
using ::rocm_ck::TargetSet;
using ::rocm_ck::Tensor;
using ::rocm_ck::TilePartitioner;

// ============================================================================
// isValidWaveTile
// ============================================================================

TEST(WaveTileValidation, AcceptsFP32With16x16Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 4, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 8, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 16, TargetSet::cdna()));
}

TEST(WaveTileValidation, AcceptsFP32With32x32OnlyForSmallK)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 4, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 8, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(
        DataType::FP32, 32, 32, 16, TargetSet::cdna())); // k=16 invalid at 32x32 for fp32
}

TEST(WaveTileValidation, AcceptsFP16With16x16Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 32, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 16, 16, 4, TargetSet::cdna()));
}

TEST(WaveTileValidation, AcceptsFP16With32x32Tile)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 8, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 16, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(
        DataType::FP16, 32, 32, 4, TargetSet::cdna())); // k=4 invalid at 32x32 for fp16
}

TEST(WaveTileValidation, AcceptsSameTilesForBF16AsFP16)
{
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16, TargetSet::cdna()));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 32, 32, 16, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::BF16, 32, 32, 4, TargetSet::cdna()));
}

TEST(WaveTileValidation, RejectsAsymmetricAndIntegerConfigs)
{
    // Asymmetric tiles not supported
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 16, 32, 8, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 32, 16, 16, TargetSet::cdna()));

    // Integer types not yet in wave tile validation table
    EXPECT_FALSE(isValidWaveTile(DataType::I32, 16, 16, 4, TargetSet::cdna()));
}

// ============================================================================
// makeSpec: plain GEMM
// ============================================================================

TEST(MakeSpec, ProducesThreePhysicalTensorsForPlainGemm)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
}

TEST(MakeSpec, MapsGemmTensorsToSequentialSlots)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
}

TEST(MakeSpec, PropagatesDtypeToAllGemmTensors)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "C"), DataType::FP16);
}

TEST(MakeSpec, ComputesThreadBlockSizeFromWaves)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    // 2 * 2 * 1 * 64 = 256
    EXPECT_EQ(k.workgroup_size, 256);
}

TEST(MakeSpec, ReportsZeroEpilogueOpsForPlainGemm)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 0);
}

// ============================================================================
// makeSpec: GEMM + Add
// ============================================================================

TEST(MakeSpec, RegistersAddAsEpilogueOp)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Add);
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, D(output), bias(D0)
}

TEST(MakeSpec, PlacesBiasInD0SlotForGemmAdd)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(slot(k, "D"), 2);    // output slot
    EXPECT_EQ(slot(k, "bias"), 3); // D0 slot
}

TEST(MakeSpec, PropagatesDtypeToBiasTensor)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
}

// ============================================================================
// makeSpec: GEMM + Add + Relu
// ============================================================================

TEST(MakeSpec, RegistersAddAndReluAsEpilogueOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Relu));
}

TEST(MakeSpec, UsesFinalOutputSlotForGemmAddRelu)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(slot(k, "E"), 2);           // final output in slot 2
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, E(output), bias(D0)
}

// ============================================================================
// makeSpec: 32x32 wave tile
// ============================================================================

TEST(MakeSpec, Accepts32x32WaveTileWithCorrectBlockSize)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 32);
    EXPECT_EQ(k.wave_tile.n, 32);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// makeSpec: layout defaults
// ============================================================================

TEST(MakeSpec, AssignsRowColRowLayoutByDefault)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesBLayoutToRowForRR)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Row);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesBothLayoutsForCC)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                                      Tensor{.name = "B", .layout = Layout::Col}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(layout(k, "A"), Layout::Col);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, OverridesALayoutForCR)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                                      Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(layout(k, "A"), Layout::Col);
    EXPECT_EQ(layout(k, "B"), Layout::Row);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
}

TEST(MakeSpec, LayoutOverrideFlowsToPhysicalTensorTable)
{
    constexpr auto k = makeSpec(Signature{.dtype   = DataType::FP16,
                                          .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    // Verify the physical tensor table (what the device code sees)
    EXPECT_EQ(k.lhs().layout, Layout::Row);
    EXPECT_EQ(k.rhs().layout, Layout::Row);
    EXPECT_EQ(k.output().layout, Layout::Row);
}

// ============================================================================
// GemmSpec named accessors
// ============================================================================

TEST(GemmSpec, ProvidesLhsRhsOutputNamedAccessors)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.lhs().args_slot, 0);
    EXPECT_EQ(k.rhs().args_slot, 1);
    EXPECT_EQ(k.output().args_slot, 2);
    EXPECT_EQ(k.lhs().dtype, DataType::FP16);
}

// ============================================================================
// Accumulator dtype
// ============================================================================

TEST(MakeSpec, DefaultsAccDtypeToFP32)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.acc_dtype, DataType::FP32); // GemmOp default acc_dtype
}

// ============================================================================
// Multiple data types
// ============================================================================

TEST(MakeSpec, ProducesFP32GemmWithMatchingAccDtype)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(dtype(k, "A"), DataType::FP32);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
}

TEST(MakeSpec, ProducesBF16GemmWithCorrectDtype)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(dtype(k, "A"), DataType::BF16);
}

// ============================================================================
// Split-K (k_batch)
// ============================================================================

TEST(MakeSpec, DefaultsKBatchToOne)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.k_batch, 1);
}

TEST(MakeSpec, AcceptsExplicitKBatch)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .k_batch     = 4},
        TargetSet::cdna());

    EXPECT_EQ(k.k_batch, 4);
}

TEST(MakeSpec, KBatchPreservesOtherFields)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .k_batch     = 4},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.block_tile.k, 32);
}

TEST(MakeSpec, KBatchWorksWithEpilogueOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{.block_tile  = {128, 128, 32},
                                              .block_waves = {2, 2, 1},
                                              .wave_tile   = {16, 16, 16},
                                              .k_batch     = 2},
                                TargetSet::cdna());

    EXPECT_EQ(k.k_batch, 2);
    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
}

// ============================================================================
// isValidWaveTile: GpuTarget-specific validation
// ============================================================================

TEST(WaveTileValidation, AcceptsFP8TilesForGfx942)
{
    // gfx942 base MFMA: 32x32x16, 16x16x32
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::gfx942));
    // IterateK compositions available on gfx942+
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx942));
}

TEST(WaveTileValidation, AcceptsFP8TilesForGfx950)
{
    // gfx950 MFMA: 32x32x{16,32,64}, 16x16x{32,64}
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx950));
}

TEST(WaveTileValidation, TargetSetAllMeansIntersectionAcrossAllTargets)
{
    // all() = intersection across ALL targets (CDNA + RDNA).
    // Only 16x16x16 FP16/BF16 pass (valid on both MFMA and WMMA).
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, TargetSet::all()));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16, TargetSet::all()));
    // I8 16x16x16 fails -- CDNA MFMA I8 tiles are 32x32x16 and 16x16x32, not 16x16x16
    EXPECT_FALSE(isValidWaveTile(DataType::I8, 16, 16, 16, TargetSet::all()));

    // 32x32 tiles fail -- WMMA only has 16x16x16
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 32, 32, 16, TargetSet::all()));

    // FP8 fails -- gfx90a has no FP8, gfx1151 has no FP8
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, TargetSet::all()));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, TargetSet::all()));

    // FP32 fails -- WMMA doesn't support FP32
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 16, 16, 4, TargetSet::all()));
}

TEST(WaveTileValidation, TargetSetCdnaRejectsFP8BecauseGfx90a)
{
    // cdna() includes gfx90a which has no FP8 -- intersection rejects all FP8 tiles
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, TargetSet::cdna()));
}

TEST(WaveTileValidation, TargetSetGfx94AcceptsFP8)
{
    // family_gfx94() = gfx942 + gfx950 -- both support FP8
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, TargetSet::family_gfx94()));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, TargetSet::family_gfx94()));
    // IterateK compositions valid across gfx94 family
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, TargetSet::family_gfx94()));
}

TEST(WaveTileValidation, Gfx90aAcceptsSameTilesAsCDNABaseline)
{
    // gfx90a has same MFMA tile set as the baseline (no FP8)
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 16, GpuTarget::gfx90a));
    // gfx90a has no FP8 MFMA support
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx90a));
}

TEST(WaveTileValidation, BF8HasSameTilesAsFP8)
{
    EXPECT_TRUE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::BF8_FNUZ, 32, 32, 32, GpuTarget::gfx942));
}

// ============================================================================
// makeSpec: GpuTarget parameter
// ============================================================================

TEST(MakeSpec, AcceptsGpuTargetParameter)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        GpuTarget::gfx942);

    EXPECT_EQ(k.workgroup_size, 256);
}

TEST(MakeSpec, AcceptsTargetSetCdna)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.workgroup_size, 256);
}

// ============================================================================
// makeSpec: Pipeline::Memory + Scheduling
// ============================================================================

TEST(MakeSpec, AcceptsMemoryPipelineWithIntrawaveScheduling)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile         = {128, 128, 32},
                      .block_waves        = {2, 2, 1},
                      .wave_tile          = {16, 16, 16},
                      .pipeline           = Pipeline::Memory,
                      .pipeline_scheduler = PipelineScheduler::Intrawave},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline, Pipeline::Memory);
    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Intrawave);
}

TEST(MakeSpec, AcceptsMemoryPipelineWithInterwaveScheduling)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile         = {128, 128, 32},
                      .block_waves        = {2, 2, 1},
                      .wave_tile          = {16, 16, 16},
                      .pipeline           = Pipeline::Memory,
                      .pipeline_scheduler = PipelineScheduler::Interwave},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline, Pipeline::Memory);
    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Interwave);
}

TEST(MakeSpec, DefaultsSchedulingToIntrawave)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline_scheduler, PipelineScheduler::Intrawave);
}

// ============================================================================
// makeSpec: quantized GEMM (INT4 weight with scale tensor)
// ============================================================================

TEST(MakeSpec, PlainGemmHasGroupSizeZero)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.group_size, 0);
}

TEST(MakeSpec, QuantizedBAddsScaleTensorToPhysicalTable)
{
    constexpr auto k = makeSpec(
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32,
                                                              .group_size  = 128}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, C, scale
}

TEST(MakeSpec, ScaleTensorGetsCorrectSlotAndDtype)
{
    constexpr auto k = makeSpec(
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32,
                                                              .group_size  = 128}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(slot(k, "scale"), 3);
    EXPECT_EQ(dtype(k, "scale"), DataType::FP32);
    EXPECT_EQ(layout(k, "scale"), Layout::Row);
}

TEST(MakeSpec, GroupSizeMatchesQuantizationConfig)
{
    constexpr auto k = makeSpec(
        Signature{
            .dtype   = DataType::FP16,
            .tensors = {Tensor{.name     = "B",
                               .dtype    = DataType::I4,
                               .quantize = Quantization{.scale_name = "scale", .group_size = 64}}},
            .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.group_size, 64);
}

TEST(MakeSpec, ScaleAccessorReturnsScaleTensor)
{
    constexpr auto k = makeSpec(
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.scale().dtype, DataType::FP32);
    EXPECT_EQ(k.scale().args_slot, 3);
}

TEST(MakeSpec, QuantizedGemmWithEpiloguePutsScaleAfterD0)
{
    constexpr auto k =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name     = "B",
                                              .dtype    = DataType::I4,
                                              .quantize = Quantization{.scale_name = "scale"}}},
                           .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                       AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());

    // A, B, D(output), bias(D0), scale = 5
    EXPECT_EQ(k.num_physical_tensors, 5);
    EXPECT_EQ(slot(k, "bias"), 3);  // D0
    EXPECT_EQ(slot(k, "scale"), 4); // scale after D0
}

TEST(MakeSpec, RhsDtypeIsI4InQuantizedGemm)
{
    constexpr auto k =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name     = "B",
                                              .dtype    = DataType::I4,
                                              .quantize = Quantization{.scale_name = "scale"}}},
                           .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());

    EXPECT_EQ(k.rhs().dtype, DataType::I4);
    EXPECT_EQ(k.lhs().dtype, DataType::FP16);
}

// ============================================================================
// makeSpec: numDTensors() derivation
// ============================================================================

TEST(MakeSpec, PlainGemmHasZeroDTensors)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.numDTensors(), 0);
}

TEST(MakeSpec, GemmAddHasOneDTensor)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.numDTensors(), 1);
}

TEST(MakeSpec, QuantizedGemmHasZeroDTensors)
{
    constexpr auto k =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name     = "B",
                                              .dtype    = DataType::I4,
                                              .quantize = Quantization{.scale_name = "scale"}}},
                           .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());

    // Scale is NOT a D tensor -- num_d_tensors excludes it
    EXPECT_EQ(k.numDTensors(), 0);
    EXPECT_EQ(k.num_physical_tensors, 4); // A, B, C, scale
}

TEST(MakeSpec, QuantizedGemmAddHasOneDTensor)
{
    constexpr auto k =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name     = "B",
                                              .dtype    = DataType::I4,
                                              .quantize = Quantization{.scale_name = "scale"}}},
                           .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                       AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());

    // bias is D0, scale is separate -- num_d_tensors counts only bias
    EXPECT_EQ(k.numDTensors(), 1);
    EXPECT_EQ(k.num_physical_tensors, 5); // A, B, D, bias, scale
}

// ============================================================================
// GemmAlgorithm padding flags
// ============================================================================

TEST(GemmAlgorithm, PaddingFlagsDefaultToFalse)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_FALSE(k.pad_m);
    EXPECT_FALSE(k.pad_n);
}

TEST(GemmAlgorithm, PadMCanBeSetToTrue)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pad_m       = true},
        TargetSet::cdna());

    EXPECT_TRUE(k.pad_m);
    EXPECT_FALSE(k.pad_n);
}

TEST(GemmAlgorithm, PadNCanBeSetToTrue)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pad_n       = true},
        TargetSet::cdna());

    EXPECT_FALSE(k.pad_m);
    EXPECT_TRUE(k.pad_n);
}

TEST(GemmAlgorithm, BothPaddingFlagsCanBeEnabled)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pad_m       = true,
                      .pad_n       = true},
        TargetSet::cdna());

    EXPECT_TRUE(k.pad_m);
    EXPECT_TRUE(k.pad_n);
}

// ============================================================================
// Pipeline enum variants
// ============================================================================

TEST(MakeSpec, AcceptsPipelineV3)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pipeline    = Pipeline::V3},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline, Pipeline::V3);
}

TEST(MakeSpec, AcceptsPipelineV4)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pipeline    = Pipeline::V4},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline, Pipeline::V4);
}

TEST(MakeSpec, AcceptsPipelinePreshuffle)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile  = {128, 128, 32},
                      .block_waves = {2, 2, 1},
                      .wave_tile   = {16, 16, 16},
                      .pipeline    = Pipeline::Preshuffle},
        TargetSet::cdna());

    EXPECT_EQ(k.pipeline, Pipeline::Preshuffle);
}

// ============================================================================
// TilePartitioner enum variants
// ============================================================================

TEST(MakeSpec, AcceptsTilePartitionerDirect)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile       = {128, 128, 32},
                      .block_waves      = {2, 2, 1},
                      .wave_tile        = {16, 16, 16},
                      .tile_partitioner = TilePartitioner::Direct},
        TargetSet::cdna());

    EXPECT_EQ(k.tile_partitioner, TilePartitioner::Direct);
}

TEST(MakeSpec, AcceptsTilePartitionerStreamK)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile       = {128, 128, 32},
                      .block_waves      = {2, 2, 1},
                      .wave_tile        = {16, 16, 16},
                      .tile_partitioner = TilePartitioner::StreamK},
        TargetSet::cdna());

    EXPECT_EQ(k.tile_partitioner, TilePartitioner::StreamK);
}

// ============================================================================
// StoreStrategy enum variants
// ============================================================================

TEST(MakeSpec, AcceptsStoreStrategyDirect2D)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{.block_tile     = {128, 128, 32},
                      .block_waves    = {2, 2, 1},
                      .wave_tile      = {16, 16, 16},
                      .store_strategy = StoreStrategy::Direct2D},
        TargetSet::cdna());

    EXPECT_EQ(k.store_strategy, StoreStrategy::Direct2D);
}

// ============================================================================
// Explicit acc_dtype override
// ============================================================================

TEST(MakeSpec, ExplicitAccDtypeIsPreserved)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16,
                  .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C", .acc_dtype = DataType::FP16}}},
        GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.acc_dtype, DataType::FP16);
}

// ============================================================================
// isValidWaveTile with unsupported dtypes
// ============================================================================

TEST(WaveTileValidation, RejectsI64)
{
    EXPECT_FALSE(isValidWaveTile(DataType::I64, 16, 16, 16, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::I64, 32, 32, 16, TargetSet::cdna()));
}

TEST(WaveTileValidation, RejectsFP64)
{
    EXPECT_FALSE(isValidWaveTile(DataType::FP64, 16, 16, 4, TargetSet::cdna()));
    EXPECT_FALSE(isValidWaveTile(DataType::FP64, 32, 32, 8, TargetSet::cdna()));
}

// ============================================================================
// Quantized GEMM + multiple epilogue ops
// ============================================================================

TEST(MakeSpec, QuantizedGemmWithMultipleEpilogueOps)
{
    constexpr auto k =
        makeSpec(Signature{.dtype   = DataType::FP16,
                           .tensors = {Tensor{.name     = "B",
                                              .dtype    = DataType::I4,
                                              .quantize = Quantization{.scale_name = "scale"}}},
                           .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                       AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                       ReluOp{.in = "D", .out = "E"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());

    // Physical tensors: A, B, E(output), bias(D0), scale
    EXPECT_EQ(k.num_physical_tensors, 5);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "E"), 2);     // final output
    EXPECT_EQ(slot(k, "bias"), 3);  // D0
    EXPECT_EQ(slot(k, "scale"), 4); // scale tensor

    // Verify epilogue ops
    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Relu));

    // Verify dtypes
    EXPECT_EQ(dtype(k, "B"), DataType::I4);
    EXPECT_EQ(dtype(k, "scale"), DataType::FP32);
    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
}

// ============================================================================
// makeSpec: two consecutive AddOps (Add+Add -> 2 D tensors)
// ============================================================================

TEST(MakeSpec, TwoConsecutiveAddOpsProduceTwoDTensors)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias0", .out = "D"},
                                                    AddOp{.lhs = "D", .rhs = "bias1", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.numDTensors(), 2);
    EXPECT_EQ(k.num_physical_tensors, 5); // A, B, E(output), bias0(D0), bias1(D1)
    EXPECT_EQ(slot(k, "bias0"), 3);       // D0
    EXPECT_EQ(slot(k, "bias1"), 4);       // D1
    EXPECT_EQ(slot(k, "E"), 2);           // final output
    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Add);
    EXPECT_EQ(k.epilogue_ops[1], EpilogueOp::Add);
}

// ============================================================================
// makeSpec: maximum epilogue ops (boundary test for kMaxEpilogueOps=4)
// ============================================================================

TEST(MakeSpec, AcceptsMaxEpilogueOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    // 2 epilogue ops (Add + Relu) -- well under the limit of 4
    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Relu));
}

// ============================================================================
// Epilogue generalization: ordering, chaining, interleaving
// ============================================================================

TEST(MakeSpec, UnaryOnlyWithoutBinaryOp)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    ReluOp{.in = "C", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Relu);
    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(k.numDTensors(), 0);
}

TEST(MakeSpec, ChainedUnaryOps)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    ReluOp{.in = "C", .out = "D"},
                                                    SigmoidOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Relu);
    EXPECT_EQ(k.epilogue_ops[1], EpilogueOp::Sigmoid);
    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(slot(k, "E"), 2);
}

TEST(MakeSpec, UnaryBeforeBinaryOp)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    ReluOp{.in = "C", .out = "D"},
                                                    AddOp{.lhs = "D", .rhs = "bias", .out = "E"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Relu);
    EXPECT_EQ(k.epilogue_ops[1], EpilogueOp::Add);
    EXPECT_EQ(k.num_physical_tensors, 4);
    EXPECT_EQ(slot(k, "E"), 2);
    EXPECT_EQ(slot(k, "bias"), 3);
}

TEST(MakeSpec, InterleavedBinaryUnaryBinary)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"},
                                                    MulOp{.lhs = "E", .rhs = "scale", .out = "F"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 3);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Add);
    EXPECT_EQ(k.epilogue_ops[1], EpilogueOp::Relu);
    EXPECT_EQ(k.epilogue_ops[2], EpilogueOp::Mul);
    EXPECT_EQ(k.numDTensors(), 2);
    EXPECT_EQ(slot(k, "bias"), 3);
    EXPECT_EQ(slot(k, "scale"), 4);
    EXPECT_EQ(slot(k, "F"), 2);
}

TEST(MakeSpec, MulOpOnly)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    MulOp{.lhs = "C", .rhs = "scale", .out = "D"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Mul);
    EXPECT_EQ(k.numDTensors(), 1);
    EXPECT_EQ(slot(k, "scale"), 3);
}

TEST(MakeSpec, AllActivationVariants)
{
    constexpr auto gelu = makeSpec(Signature{.dtype = DataType::FP16,
                                             .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                       GeluOp{.in = "C", .out = "D"}}},
                                   GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                   TargetSet::cdna());
    EXPECT_EQ(gelu.epilogue_ops[0], EpilogueOp::Gelu);

    constexpr auto fast_gelu =
        makeSpec(Signature{.dtype = DataType::FP16,
                           .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                     FastGeluOp{.in = "C", .out = "D"}}},
                 GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                 TargetSet::cdna());
    EXPECT_EQ(fast_gelu.epilogue_ops[0], EpilogueOp::FastGelu);

    constexpr auto silu = makeSpec(Signature{.dtype = DataType::FP16,
                                             .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                       SiluOp{.in = "C", .out = "D"}}},
                                   GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                   TargetSet::cdna());
    EXPECT_EQ(silu.epilogue_ops[0], EpilogueOp::Silu);

    constexpr auto sigmoid = makeSpec(Signature{.dtype = DataType::FP16,
                                                .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                        SigmoidOp{.in = "C", .out = "D"}}},
                                      GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                      TargetSet::cdna());
    EXPECT_EQ(sigmoid.epilogue_ops[0], EpilogueOp::Sigmoid);
}

TEST(MakeSpec, EpilogueOpsPreserveInsertionOrder)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    SigmoidOp{.in = "C", .out = "D"},
                                                    AddOp{.lhs = "D", .rhs = "bias", .out = "E"},
                                                    FastGeluOp{.in = "E", .out = "F"}}},
                                GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_epilogue_ops, 3);
    EXPECT_EQ(k.epilogue_ops[0], EpilogueOp::Sigmoid);
    EXPECT_EQ(k.epilogue_ops[1], EpilogueOp::Add);
    EXPECT_EQ(k.epilogue_ops[2], EpilogueOp::FastGelu);
}
