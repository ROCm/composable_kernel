// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/resolve.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::AddOp;
using ::rocm_ck::BinaryOpLike;
using ::rocm_ck::DataType;
using ::rocm_ck::FastGeluOp;
using ::rocm_ck::GeluOp;
using ::rocm_ck::GemmOp;
using ::rocm_ck::kMaxTensors;
using ::rocm_ck::Layout;
using ::rocm_ck::MulOp;
using ::rocm_ck::Quantization;
using ::rocm_ck::ReluOp;
using ::rocm_ck::resolve;
using ::rocm_ck::Scalar;
using ::rocm_ck::ScaleOp;
using ::rocm_ck::SigmoidOp;
using ::rocm_ck::Signature;
using ::rocm_ck::SiluOp;
using ::rocm_ck::SoftmaxOp;
using ::rocm_ck::Tensor;
using ::rocm_ck::UnaryOpLike;

// ============================================================================
// Simple GemmOp resolution
// ============================================================================

TEST(Resolve, ResolvesSimpleGemmToThreeTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_tensors, 3);
}

TEST(Resolve, CascadesSignatureDtypeToAllGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").dtype, DataType::FP16);
    EXPECT_EQ(r.tensor("B").dtype, DataType::FP16);
    EXPECT_EQ(r.tensor("C").dtype, DataType::FP16);
}

TEST(Resolve, AssignsRank2ToGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").rank, 2);
    EXPECT_EQ(r.tensor("B").rank, 2);
    EXPECT_EQ(r.tensor("C").rank, 2);
}

TEST(Resolve, AssignsRowColRowLayoutToGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Row);
    EXPECT_EQ(r.tensor("B").layout, Layout::Col);
    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
}

// ============================================================================
// Custom tensor names
// ============================================================================

TEST(Resolve, AcceptsCustomTensorNames)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "X", .rhs = "Y", .out = "Z"}}});

    EXPECT_EQ(r.tensor("X").rank, 2);
    EXPECT_EQ(r.tensor("Y").rank, 2);
    EXPECT_EQ(r.tensor("Z").rank, 2);
}

// ============================================================================
// dtype cascade
// ============================================================================

TEST(Resolve, CascadesBF16DtypeToAllTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").dtype, DataType::BF16);
    EXPECT_EQ(r.tensor("C").dtype, DataType::BF16);
}

TEST(Resolve, AllowsPerTensorDtypeOverride)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("C").dtype, DataType::FP32);
    EXPECT_EQ(r.tensor("A").dtype, DataType::FP16); // cascade still applies to A
}

// ============================================================================
// Explicit tensor rank/layout overrides
// ============================================================================

TEST(Resolve, AllowsPerTensorRankOverride)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name = "A", .rank = 3}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").rank, 3);
}

TEST(Resolve, AllowsPerTensorLayoutOverride)
{
    // Override B from default Col to Row (RxR layout)
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Row); // default preserved
    EXPECT_EQ(r.tensor("B").layout, Layout::Row); // overridden from Col
    EXPECT_EQ(r.tensor("C").layout, Layout::Row); // default preserved
}

TEST(Resolve, AllowsMultipleLayoutOverrides)
{
    // Override both A and B (CxC layout)
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                              Tensor{.name = "B", .layout = Layout::Col}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Col);
    EXPECT_EQ(r.tensor("B").layout, Layout::Col);
    EXPECT_EQ(r.tensor("C").layout, Layout::Row); // default preserved
}

// ============================================================================
// GEMM + Add + Relu chain
// ============================================================================

TEST(Resolve, ResolvesGemmAddReluToSixTensors)
{
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                            ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.num_tensors, 6); // A, B, C, bias, D, E
}

TEST(Resolve, PropagatesRankAndLayoutThroughEpilogueChain)
{
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                            ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.tensor("C").rank, 2);
    EXPECT_EQ(r.tensor("bias").rank, 2);
    EXPECT_EQ(r.tensor("bias").layout, Layout::Row);
    EXPECT_EQ(r.tensor("D").rank, 2);
    EXPECT_EQ(r.tensor("D").layout, Layout::Row);
    EXPECT_EQ(r.tensor("E").rank, 2);
    EXPECT_EQ(r.tensor("E").layout, Layout::Row);
}

TEST(Resolve, PropagatesRankAndLayoutThroughDiamondDAG)
{
    // Diamond: GEMM->C splits into two Add paths, then joins.
    //   C -> Add(C,bias1)->D1 --> Add(D1,D2)->E
    //   C -> Add(C,bias2)->D2 -+
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            AddOp{.lhs = "C", .rhs = "bias1", .out = "D1"},
                            AddOp{.lhs = "C", .rhs = "bias2", .out = "D2"},
                            AddOp{.lhs = "D1", .rhs = "D2", .out = "E"}}});

    EXPECT_EQ(r.num_tensors, 8); // A, B, C, bias1, D1, bias2, D2, E
    EXPECT_EQ(r.tensor("D1").rank, 2);
    EXPECT_EQ(r.tensor("D2").rank, 2);
    EXPECT_EQ(r.tensor("E").rank, 2);
    EXPECT_EQ(r.tensor("bias1").layout, Layout::Row);
    EXPECT_EQ(r.tensor("E").layout, Layout::Row);
}

TEST(Resolve, AssignsSequentialIndicesToChainedOps)
{
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                            ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.tensorIndex("A"), 0);
    EXPECT_EQ(r.tensorIndex("B"), 1);
    EXPECT_EQ(r.tensorIndex("C"), 2);
    EXPECT_EQ(r.tensorIndex("bias"), 3);
    EXPECT_EQ(r.tensorIndex("D"), 4);
    EXPECT_EQ(r.tensorIndex("E"), 5);
}

// ============================================================================
// Standalone AddOp
// ============================================================================

TEST(Resolve, ResolvesStandaloneAddWithoutImpliedRank)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_tensors, 3);
    EXPECT_EQ(r.tensor("A").rank, 0);              // no op implies rank
    EXPECT_EQ(r.tensor("A").layout, Layout::Auto); // no op implies layout
}

// ============================================================================
// Conflict detection -- redundant identical sets are silent
// ============================================================================

TEST(Resolve, AllowsRedundantIdenticalLayoutFromTwoGemmOps)
{
    // GemmOp1 outputs "C" as Row. GemmOp2 uses "C" as lhs (also Row).
    // Two ops set the same layout -> no conflict.
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            GemmOp{.lhs = "C", .rhs = "D", .out = "E"}}});

    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
    EXPECT_EQ(r.tensor("C").rank, 2);
}

TEST(Resolve, AllowsPropagationThroughAddWithConsistentLayout)
{
    // GemmOp sets C=Row. AddOp connects C to bias and D.
    // Propagation sets bias and D to Row (matching C) -> no conflict.
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                            AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}});

    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
    EXPECT_EQ(r.tensor("bias").layout, Layout::Row);
    EXPECT_EQ(r.tensor("D").layout, Layout::Row);
}

// ============================================================================
// FMHA pattern: two GemmOps + SoftmaxOp
// ============================================================================

TEST(Resolve, ResolvesFMHATwoGemmSoftmaxPattern)
{
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "Q", .rhs = "K", .out = "S"},
                            SoftmaxOp{.in = "S", .out = "P"},
                            GemmOp{.lhs = "P", .rhs = "V", .out = "O"}}});

    EXPECT_EQ(r.num_tensors, 6); // Q, K, S, P, V, O
    EXPECT_EQ(r.tensor("Q").rank, 2);
    EXPECT_EQ(r.tensor("S").rank, 2);
    EXPECT_EQ(r.tensor("P").rank, 2); // propagated via SoftmaxOp
    EXPECT_EQ(r.tensor("O").rank, 2);
}

// ============================================================================
// Scalar tracking
// ============================================================================

TEST(Resolve, PreservesScalarNamesAndDtypes)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .scalars = {Scalar{.name = "alpha", .dtype = DataType::FP32},
                              Scalar{.name = "beta", .dtype = DataType::FP32}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_scalars, 2);
    EXPECT_EQ(r.scalar("alpha").dtype, DataType::FP32);
    EXPECT_EQ(r.scalar("beta").dtype, DataType::FP32);
    EXPECT_EQ(r.scalarIndex("alpha"), 0);
    EXPECT_EQ(r.scalarIndex("beta"), 1);
}

TEST(Resolve, ReportsZeroScalarsWhenNoneDeclared)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_scalars, 0);
}

// ============================================================================
// findTensor / findScalar (constexpr, not consteval -- returns -1 on miss)
// ============================================================================

TEST(Resolve, FindsTensorByName)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.findTensor("A"), 0);
    EXPECT_EQ(r.findTensor("C"), 2);
}

TEST(Resolve, ReturnsNegativeOneForUnknownTensor)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.findTensor("Z"), -1);
}

TEST(Resolve, ReturnsNegativeOneForUnknownScalar)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.findScalar("nonexistent"), -1);
}

// ============================================================================
// Quantized tensors
// ============================================================================

TEST(Resolve, QuantizedBAutoRegistersScaleTensor)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32,
                                                              .group_size  = 128}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    // A, B, C from GemmOp + scale auto-registered = 4 tensors
    EXPECT_EQ(r.num_tensors, 4);
}

TEST(Resolve, ScaleTensorGetsDtypeFromQuantization)
{
    constexpr auto r = resolve(
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    // Scale tensor dtype comes from Quantization, not the signature cascade
    EXPECT_EQ(r.tensor("scale").dtype, DataType::FP32);
}

TEST(Resolve, ScaleTensorGetsRank2RowLayout)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name = "scale"}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("scale").rank, 2);
    EXPECT_EQ(r.tensor("scale").layout, Layout::Row);
}

TEST(Resolve, QuantizedTensorKeepsOwnDtype)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name = "scale"}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("B").dtype, DataType::I4);
    EXPECT_EQ(r.tensor("A").dtype, DataType::FP16); // cascade still works
}

TEST(Resolve, QuantizedResolvedTensorCarriesQuantizeInfo)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name  = "scale",
                                                              .scale_dtype = DataType::FP32,
                                                              .group_size  = 64}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_TRUE(r.tensor("B").quantize.has_value());
    EXPECT_EQ(r.tensor("B").quantize->scale_name, "scale");
    EXPECT_EQ(r.tensor("B").quantize->scale_dtype, DataType::FP32);
    EXPECT_EQ(r.tensor("B").quantize->group_size, 64);
}

TEST(Resolve, NonQuantizedTensorHasNoQuantizeInfo)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name = "scale"}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_FALSE(r.tensor("A").quantize.has_value());
    EXPECT_FALSE(r.tensor("C").quantize.has_value());
    EXPECT_FALSE(r.tensor("scale").quantize.has_value());
}

TEST(Resolve, QuantizedGemmWithEpiloguePreservesScaleTensor)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .tensors = {Tensor{.name     = "B",
                                     .dtype    = DataType::I4,
                                     .quantize = Quantization{.scale_name = "scale"}}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                              AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                              ReluOp{.in = "D", .out = "E"}}});

    // A, B, C, bias, D, E from ops + scale auto-registered = 7
    EXPECT_EQ(r.num_tensors, 7);
    EXPECT_EQ(r.tensor("scale").dtype, DataType::FP32);
    EXPECT_TRUE(r.tensor("B").quantize.has_value());
}

// ============================================================================
// C++20 concepts
// ============================================================================

TEST(Concepts, ClassifiesAddAndMulAsBinaryOpLike)
{
    EXPECT_TRUE(BinaryOpLike<AddOp>);
    EXPECT_TRUE(BinaryOpLike<MulOp>);
    EXPECT_FALSE(BinaryOpLike<ReluOp>);
    EXPECT_FALSE(BinaryOpLike<SoftmaxOp>);
}

TEST(Concepts, ClassifiesActivationsAsUnaryOpLike)
{
    EXPECT_TRUE(UnaryOpLike<ReluOp>);
    EXPECT_TRUE(UnaryOpLike<FastGeluOp>);
    EXPECT_TRUE(UnaryOpLike<GeluOp>);
    EXPECT_TRUE(UnaryOpLike<SiluOp>);
    EXPECT_TRUE(UnaryOpLike<SigmoidOp>);
    EXPECT_TRUE(UnaryOpLike<SoftmaxOp>);
    EXPECT_FALSE(UnaryOpLike<AddOp>);
    EXPECT_FALSE(UnaryOpLike<GemmOp>);
}

TEST(Concepts, ClassifiesGemmOpAsBinaryButNotUnary)
{
    // GemmOp has lhs/rhs/out AND is special-cased, not generic BinaryOpLike
    // (it has .lhs, .rhs, .out but is handled separately in registerSlots)
    EXPECT_TRUE(BinaryOpLike<GemmOp>); // structurally matches, but dispatch special-cases it
    EXPECT_FALSE(UnaryOpLike<GemmOp>);
}

TEST(Concepts, ClassifiesScaleOpAsUnaryNotBinary)
{
    EXPECT_TRUE(UnaryOpLike<ScaleOp>);
    EXPECT_FALSE(BinaryOpLike<ScaleOp>);
}

// ============================================================================
// ScaleOp with explicit Scalar
// ============================================================================

TEST(Resolve, ScaleOpReferencesExplicitScalar)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .scalars = {Scalar{.name = "alpha", .dtype = DataType::FP32}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                              ScaleOp{.in = "C", .out = "D", .scale = "alpha"}}});

    EXPECT_EQ(r.num_tensors, 4); // A, B, C, D
    EXPECT_EQ(r.num_scalars, 1);
    EXPECT_EQ(r.scalar("alpha").dtype, DataType::FP32);
    EXPECT_EQ(r.scalarIndex("alpha"), 0);
}

TEST(Resolve, ScaleOpPreservesScalarDtype)
{
    constexpr auto r = resolve( //
        Signature{.dtype   = DataType::FP16,
                  .scalars = {Scalar{.name = "scale_factor", .dtype = DataType::FP16}},
                  .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                              ScaleOp{.in = "C", .out = "D", .scale = "scale_factor"}}});

    EXPECT_EQ(r.scalar("scale_factor").dtype, DataType::FP16);
}

// ============================================================================
// Boundary: signature at kMaxTensors
// ============================================================================

TEST(Resolve, HandlesSignatureWithManyTensors)
{
    // Create a chain of AddOps to generate many tensors (close to kMaxTensors).
    // Each AddOp creates 3 tensors (lhs, rhs, out). We'll create a chain that
    // approaches the limit.
    // kMaxTensors is 16, so a signature with 3 GEMMs (each with 3 tensors = 9)
    // plus some adds should get close.
    constexpr auto r = resolve( //
        Signature{.dtype = DataType::FP16,
                  .ops   = {GemmOp{.lhs = "A1", .rhs = "B1", .out = "C1"},
                            GemmOp{.lhs = "A2", .rhs = "B2", .out = "C2"},
                            GemmOp{.lhs = "A3", .rhs = "B3", .out = "C3"},
                            AddOp{.lhs = "C1", .rhs = "C2", .out = "D1"},
                            AddOp{.lhs = "D1", .rhs = "C3", .out = "D2"}}});

    // A1, B1, C1, A2, B2, C2, A3, B3, C3, D1, D2 = 11 tensors
    EXPECT_EQ(r.num_tensors, 11);
    EXPECT_EQ(r.tensor("A1").dtype, DataType::FP16);
    EXPECT_EQ(r.tensor("D2").dtype, DataType::FP16);
}
