// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/arch_properties.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::ArchFamily;
using ::rocm_ck::DataType;
using ::rocm_ck::GpuTarget;
using ::rocm_ck::isValidWaveTile;
using ::rocm_ck::TargetProperties;
using ::rocm_ck::TargetSet;

// ============================================================================
// TargetProperties
// ============================================================================

TEST(TargetProperties, ReturnsWave64ForCDNA)
{
    EXPECT_EQ(properties(GpuTarget::gfx90a).wavefront_size, 64);
    EXPECT_EQ(properties(GpuTarget::gfx942).wavefront_size, 64);
    EXPECT_EQ(properties(GpuTarget::gfx950).wavefront_size, 64);
}

TEST(TargetProperties, ReturnsWave32ForRDNA)
{
    EXPECT_EQ(properties(GpuTarget::gfx1100).wavefront_size, 32);
    EXPECT_EQ(properties(GpuTarget::gfx1101).wavefront_size, 32);
    EXPECT_EQ(properties(GpuTarget::gfx1102).wavefront_size, 32);
    EXPECT_EQ(properties(GpuTarget::gfx1150).wavefront_size, 32);
    EXPECT_EQ(properties(GpuTarget::gfx1151).wavefront_size, 32);
}

TEST(TargetProperties, CDNAArchFamily)
{
    EXPECT_EQ(properties(GpuTarget::gfx90a).arch_family, ArchFamily::CDNA);
    EXPECT_EQ(properties(GpuTarget::gfx942).arch_family, ArchFamily::CDNA);
    EXPECT_EQ(properties(GpuTarget::gfx950).arch_family, ArchFamily::CDNA);
}

TEST(TargetProperties, RDNAArchFamily)
{
    EXPECT_EQ(properties(GpuTarget::gfx1100).arch_family, ArchFamily::RDNA);
    EXPECT_EQ(properties(GpuTarget::gfx1101).arch_family, ArchFamily::RDNA);
    EXPECT_EQ(properties(GpuTarget::gfx1102).arch_family, ArchFamily::RDNA);
    EXPECT_EQ(properties(GpuTarget::gfx1150).arch_family, ArchFamily::RDNA);
    EXPECT_EQ(properties(GpuTarget::gfx1151).arch_family, ArchFamily::RDNA);
}

// ============================================================================
// isCDNA / isRDNA predicates
// ============================================================================

TEST(ArchFamily, IsCDNAForAllCDNATargets)
{
    EXPECT_TRUE(isCDNA(GpuTarget::gfx90a));
    EXPECT_TRUE(isCDNA(GpuTarget::gfx942));
    EXPECT_TRUE(isCDNA(GpuTarget::gfx950));
    EXPECT_FALSE(isCDNA(GpuTarget::gfx1100));
    EXPECT_FALSE(isCDNA(GpuTarget::gfx1101));
    EXPECT_FALSE(isCDNA(GpuTarget::gfx1102));
    EXPECT_FALSE(isCDNA(GpuTarget::gfx1150));
    EXPECT_FALSE(isCDNA(GpuTarget::gfx1151));
}

TEST(ArchFamily, IsRDNAForAllRDNATargets)
{
    EXPECT_TRUE(isRDNA(GpuTarget::gfx1100));
    EXPECT_TRUE(isRDNA(GpuTarget::gfx1101));
    EXPECT_TRUE(isRDNA(GpuTarget::gfx1102));
    EXPECT_TRUE(isRDNA(GpuTarget::gfx1150));
    EXPECT_TRUE(isRDNA(GpuTarget::gfx1151));
    EXPECT_FALSE(isRDNA(GpuTarget::gfx90a));
    EXPECT_FALSE(isRDNA(GpuTarget::gfx942));
    EXPECT_FALSE(isRDNA(GpuTarget::gfx950));
}

// ============================================================================
// wavefrontSize free function
// ============================================================================

TEST(WavefrontSize, MatchesTargetProperties)
{
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx90a), 64);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx942), 64);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx950), 64);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1100), 32);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1101), 32);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1102), 32);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1150), 32);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1151), 32);
}

// ============================================================================
// TargetSet -- construction
// ============================================================================

TEST(TargetSet, DefaultConstructsToEmpty)
{
    constexpr TargetSet ts{};
    EXPECT_TRUE(ts.is_empty());
    EXPECT_EQ(ts.count(), 0);
}

TEST(TargetSet, ImplicitConversionFromSingleTarget)
{
    constexpr TargetSet ts = GpuTarget::gfx942;
    EXPECT_TRUE(ts.is_single_target());
    EXPECT_EQ(ts.single_target(), GpuTarget::gfx942);
    EXPECT_EQ(ts.count(), 1);
}

// ============================================================================
// TargetSet -- named constructors
// ============================================================================

TEST(TargetSet, AllContainsEveryTarget)
{
    constexpr auto ts = TargetSet::all();
    EXPECT_EQ(ts.count(), 8);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx90a));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx942));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx950));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1100));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1101));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1102));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1150));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1151));
}

TEST(TargetSet, CdnaContainsOnlyCDNATargets)
{
    constexpr auto ts = TargetSet::cdna();
    EXPECT_EQ(ts.count(), 3);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx90a));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx942));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx950));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1100));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1101));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1102));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1150));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1151));
}

TEST(TargetSet, RdnaContainsOnlyRDNATargets)
{
    constexpr auto ts = TargetSet::rdna();
    EXPECT_EQ(ts.count(), 5);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1100));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1101));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1102));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1150));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1151));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx90a));
}

TEST(TargetSet, FamilyGfx9MatchesCdna) { EXPECT_EQ(TargetSet::family_gfx9(), TargetSet::cdna()); }

TEST(TargetSet, FamilyGfx94ExcludesGfx90a)
{
    constexpr auto ts = TargetSet::family_gfx94();
    EXPECT_EQ(ts.count(), 2);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx942));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx950));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx90a));
}

TEST(TargetSet, FamilyGfx11MatchesRdna) { EXPECT_EQ(TargetSet::family_gfx11(), TargetSet::rdna()); }

TEST(TargetSet, FamilyGfx115ContainsRDNA35Only)
{
    constexpr auto ts = TargetSet::family_gfx115();
    EXPECT_EQ(ts.count(), 2);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1150));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx1151));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1100));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1101));
    EXPECT_FALSE(ts.contains(GpuTarget::gfx1102));
}

TEST(TargetSet, OnlyWithOneTarget)
{
    constexpr auto ts = TargetSet::only(GpuTarget::gfx942);
    EXPECT_EQ(ts.count(), 1);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx942));
}

TEST(TargetSet, OnlyWithTwoTargets)
{
    constexpr auto ts = TargetSet::only(GpuTarget::gfx942, GpuTarget::gfx950);
    EXPECT_EQ(ts.count(), 2);
    EXPECT_TRUE(ts.contains(GpuTarget::gfx942));
    EXPECT_TRUE(ts.contains(GpuTarget::gfx950));
}

TEST(TargetSet, OnlyWithThreeTargets)
{
    constexpr auto ts = TargetSet::only(GpuTarget::gfx90a, GpuTarget::gfx942, GpuTarget::gfx950);
    EXPECT_EQ(ts.count(), 3);
}

// Note: OnlyWithOneTarget, OnlyWithTwoTargets, OnlyWithThreeTargets test the
// variadic arity of TargetSet::only() overloads (1, 2, 3 parameters).

// ============================================================================
// TargetSet -- set operations
// ============================================================================

TEST(TargetSet, ExcludingRemovesOneTarget)
{
    constexpr auto base    = TargetSet::cdna();
    constexpr auto without = base.excluding(GpuTarget::gfx90a);
    EXPECT_EQ(without.count(), 2);
    EXPECT_FALSE(without.contains(GpuTarget::gfx90a));
    EXPECT_TRUE(without.contains(GpuTarget::gfx942));
    EXPECT_TRUE(without.contains(GpuTarget::gfx950));
}

TEST(TargetSet, UnionCombinesSets)
{
    constexpr auto a        = TargetSet::only(GpuTarget::gfx90a);
    constexpr auto b        = TargetSet::only(GpuTarget::gfx1151);
    constexpr auto combined = a.union_with(b);
    EXPECT_EQ(combined.count(), 2);
    EXPECT_TRUE(combined.contains(GpuTarget::gfx90a));
    EXPECT_TRUE(combined.contains(GpuTarget::gfx1151));
}

TEST(TargetSet, IntersectReturnsCommonTargets)
{
    constexpr auto all  = TargetSet::all();
    constexpr auto cdna = TargetSet::cdna();
    EXPECT_EQ(all.intersect_with(cdna), cdna);
}

TEST(TargetSet, MinusRemovesTargets)
{
    constexpr auto all  = TargetSet::all();
    constexpr auto rdna = TargetSet::rdna();
    EXPECT_EQ(all.minus(rdna), TargetSet::cdna());
}

TEST(TargetSet, MinusWithEmptySetIsIdentity)
{
    constexpr TargetSet empty{};
    constexpr auto cdna = TargetSet::cdna();
    EXPECT_EQ(cdna.minus(empty), cdna);
}

TEST(TargetSet, EmptyMinusAnythingIsEmpty)
{
    constexpr TargetSet empty{};
    constexpr auto cdna = TargetSet::cdna();
    EXPECT_TRUE(empty.minus(cdna).is_empty());
}

TEST(TargetSet, EmptyUnionIsIdentity)
{
    constexpr TargetSet empty{};
    constexpr auto cdna = TargetSet::cdna();
    EXPECT_EQ(empty.union_with(cdna), cdna);
    EXPECT_EQ(cdna.union_with(empty), cdna);
}

TEST(TargetSet, EmptyIntersectIsEmpty)
{
    constexpr TargetSet empty{};
    constexpr auto cdna = TargetSet::cdna();
    EXPECT_TRUE(empty.intersect_with(cdna).is_empty());
}

// ============================================================================
// TargetSet -- operators
// ============================================================================

TEST(TargetSet, OperatorOrDelegatesToUnion)
{
    constexpr auto a = TargetSet::only(GpuTarget::gfx942);
    constexpr auto b = TargetSet::only(GpuTarget::gfx950);
    EXPECT_EQ((a | b).count(), 2);
}

TEST(TargetSet, OperatorAndDelegatesToIntersect)
{
    EXPECT_EQ(TargetSet::all() & TargetSet::cdna(), TargetSet::cdna());
}

TEST(TargetSet, OperatorMinusDelegatesToMinus)
{
    EXPECT_EQ(TargetSet::all() - TargetSet::rdna(), TargetSet::cdna());
}

TEST(TargetSet, EqualitySameSetReturnsTrue) { EXPECT_EQ(TargetSet::cdna(), TargetSet::cdna()); }

TEST(TargetSet, InequalityDifferentSetsReturnsTrue)
{
    EXPECT_NE(TargetSet::cdna(), TargetSet::rdna());
}

// ============================================================================
// TargetSet -- queries
// ============================================================================

TEST(TargetSet, ContainsReturnsTrueForMember)
{
    EXPECT_TRUE(TargetSet::cdna().contains(GpuTarget::gfx90a));
}

TEST(TargetSet, ContainsReturnsFalseForNonMember)
{
    EXPECT_FALSE(TargetSet::cdna().contains(GpuTarget::gfx1151));
}

TEST(TargetSet, IsSingleTargetTrueForOne)
{
    EXPECT_TRUE(TargetSet::only(GpuTarget::gfx942).is_single_target());
}

TEST(TargetSet, IsSingleTargetFalseForMultiple)
{
    EXPECT_FALSE(TargetSet::cdna().is_single_target());
}

TEST(TargetSet, IsSingleTargetFalseForEmpty) { EXPECT_FALSE(TargetSet{}.is_single_target()); }

TEST(TargetSet, WavefrontSizeUniformForCDNA) { EXPECT_EQ(TargetSet::cdna().wavefront_size(), 64); }

TEST(TargetSet, WavefrontSizeUniformForRDNA) { EXPECT_EQ(TargetSet::rdna().wavefront_size(), 32); }

TEST(TargetSet, WavefrontSizeThrowsOnMixedCDNAAndRDNA)
{
    constexpr auto mixed = TargetSet::all();
    // wavefront_size() should throw when set mixes wave64 (CDNA) and wave32 (RDNA)
    bool caught = false;
    try
    {
        int wf = mixed.wavefront_size();
        (void)wf; // suppress unused warning if exception is not thrown
    }
    catch(...)
    {
        caught = true;
    }
    EXPECT_TRUE(caught);
}

TEST(TargetSet, IsAllCdnaPredicateWorks)
{
    EXPECT_TRUE(TargetSet::cdna().is_all_cdna());
    EXPECT_FALSE(TargetSet::rdna().is_all_cdna());
    EXPECT_FALSE(TargetSet::all().is_all_cdna());
    EXPECT_FALSE(TargetSet{}.is_all_cdna());
}

TEST(TargetSet, IsAllRdnaPredicateWorks)
{
    EXPECT_TRUE(TargetSet::rdna().is_all_rdna());
    EXPECT_FALSE(TargetSet::cdna().is_all_rdna());
    EXPECT_FALSE(TargetSet::all().is_all_rdna());
    EXPECT_FALSE(TargetSet{}.is_all_rdna());
}

// ============================================================================
// TargetSet -- iteration
// ============================================================================

TEST(TargetSet, ForEachIteratesAllTargets)
{
    int count = 0;
    TargetSet::cdna().for_each([&count](GpuTarget) { count++; });
    EXPECT_EQ(count, 3);
}

TEST(TargetSet, ForEachOnEmptySetDoesNothing)
{
    int count = 0;
    TargetSet{}.for_each([&count](GpuTarget) { count++; });
    EXPECT_EQ(count, 0);
}

// ============================================================================
// isValidWaveTile -- single target
// ============================================================================

TEST(IsValidWaveTile, FP32MFMATilesOnCDNA)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 4, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 8, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 16, 16, 16, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 4, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP32, 32, 32, 8, GpuTarget::gfx942));
}

TEST(IsValidWaveTile, FP32RejectsInvalidTiles)
{
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 32, 32, 16, GpuTarget::gfx90a));
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 8, 8, 4, GpuTarget::gfx90a));
}

TEST(IsValidWaveTile, FP16MFMATilesOnCDNA)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 32, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 8, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 32, 32, 16, GpuTarget::gfx942));
}

TEST(IsValidWaveTile, BF16MatchesFP16Tiles)
{
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16, GpuTarget::gfx90a));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 32, 32, 16, GpuTarget::gfx942));
}

TEST(IsValidWaveTile, INT8MFMATilesOnCDNA)
{
    EXPECT_TRUE(isValidWaveTile(DataType::I8, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::I8, 16, 16, 32, GpuTarget::gfx942));
}

TEST(IsValidWaveTile, FP8FNUZNotSupportedOnGfx90a)
{
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx90a));
}

TEST(IsValidWaveTile, FP8FNUZBaseTilesOnGfx942)
{
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 32, GpuTarget::gfx942));
}

TEST(IsValidWaveTile, FP8FNUZIterateKTilesOnGfx942Plus)
{
    // IterateK compositions of base FP8 MFMA -- available on gfx942+
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx942));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 64, GpuTarget::gfx950));
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 16, 16, 64, GpuTarget::gfx950));
    // Still not on gfx90a
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 32, GpuTarget::gfx90a));
}

TEST(IsValidWaveTile, WMMATilesOnRDNA)
{
    // All RDNA targets share identical WMMA: 16x16x16 for FP16, BF16, INT8
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx1100));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16, GpuTarget::gfx1100));
    EXPECT_TRUE(isValidWaveTile(DataType::I8, 16, 16, 16, GpuTarget::gfx1100));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx1101));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx1102));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx1150));
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, GpuTarget::gfx1151));
    EXPECT_TRUE(isValidWaveTile(DataType::BF16, 16, 16, 16, GpuTarget::gfx1151));
    EXPECT_TRUE(isValidWaveTile(DataType::I8, 16, 16, 16, GpuTarget::gfx1151));
}

TEST(IsValidWaveTile, WMMARejectsNon16x16x16)
{
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 32, 32, 8, GpuTarget::gfx1100));
    EXPECT_FALSE(isValidWaveTile(DataType::FP16, 16, 16, 32, GpuTarget::gfx1151));
}

TEST(IsValidWaveTile, WMMARejectsFP32)
{
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 16, 16, 16, GpuTarget::gfx1100));
    EXPECT_FALSE(isValidWaveTile(DataType::FP32, 16, 16, 16, GpuTarget::gfx1151));
}

// ============================================================================
// isValidWaveTile -- TargetSet (intersection semantics)
// ============================================================================

TEST(IsValidWaveTile, IntersectionAcrossCDNATargets)
{
    // FP16 16x16x16 valid on all CDNA
    EXPECT_TRUE(isValidWaveTile(DataType::FP16, 16, 16, 16, TargetSet::cdna()));
}

TEST(IsValidWaveTile, IntersectionRejectsWhenAnyTargetFails)
{
    // FP8 32x32x16 valid on gfx942 but NOT on gfx90a
    EXPECT_FALSE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, TargetSet::cdna()));
    // Valid for gfx94 family only
    EXPECT_TRUE(isValidWaveTile(DataType::FP8_FNUZ, 32, 32, 16, TargetSet::family_gfx94()));
}
