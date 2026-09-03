// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "ck_tile/host/device_memory.hpp"

#include "gtest/gtest.h"

#include <vector>

namespace {

using QKPad = ck_tile::detail::LdsPaddingConfig<true, 256, 16>;
using VPad  = ck_tile::detail::LdsPaddingConfig<true, 256, 32>;
using NoPad = ck_tile::detail::LdsPaddingConfig<false, 0, 0>;

#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx125__)
static_assert(ck_tile::detail::is_qr_tdm_padding_supported_arch_v);
#else
static_assert(!ck_tile::detail::is_qr_tdm_padding_supported_arch_v);
#endif
#elif defined(CK_USE_GFX1250)
static_assert(ck_tile::detail::is_qr_tdm_padding_supported_arch_v);
#else
static_assert(!ck_tile::detail::is_qr_tdm_padding_supported_arch_v);
#endif

template <typename DataType, typename Descriptor>
constexpr ck_tile::index_t
byte_offset(const Descriptor& descriptor, ck_tile::index_t row, ck_tile::index_t col)
{
    return descriptor.calculate_offset(ck_tile::make_tuple(row, col)) * sizeof(DataType);
}

template <typename DataType>
using PaddedQDescriptor =
    decltype(ck_tile::detail::
                 make_qr_tdm_row_major_lds_descriptor<DataType, 128, 128, QKPad, 16>());

struct QTag
{
    using PaddingConfig                                   = QKPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 0;
    [[maybe_unused]] static constexpr bool kTranspose     = false;
};

struct KTag
{
    using PaddingConfig                                   = QKPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 1;
    [[maybe_unused]] static constexpr bool kTranspose     = false;
};

struct VTag
{
    using PaddingConfig                                   = VPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 2;
    [[maybe_unused]] static constexpr bool kTranspose     = true;
};

template <ck_tile::index_t M>
using TestFmhaShape = ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<16, 16, 32>,
                                             true>;

using TestFmhaTraits = ck_tile::TileFmhaTraits<false,
                                               false,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                               false,
                                               false,
                                               false,
                                               ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>;

template <typename DataType, ck_tile::index_t M>
using TestFmhaProblem =
    ck_tile::BlockFmhaPipelineProblem<DataType,
                                      DataType,
                                      DataType,
                                      float,
                                      float,
                                      DataType,
                                      uint8_t,
                                      float,
                                      DataType,
                                      float,
                                      DataType,
                                      TestFmhaShape<M>,
                                      false,
                                      ck_tile::ComposedAttention<0>,
                                      ck_tile::SimplifiedGenericAttentionMask<false>,
                                      false,
                                      TestFmhaTraits>;

template <typename BaseProblem, typename QDataType_, typename KDataType_, typename VDataType_>
struct TestProblemWithDataTypes : BaseProblem
{
    using QDataType = QDataType_;
    using KDataType = KDataType_;
    using VDataType = VDataType_;
};

template <typename BaseShape, ck_tile::index_t QKHeadDim, ck_tile::index_t VHeadDim>
struct TestShapeWithHeadDims : BaseShape
{
    static constexpr ck_tile::index_t kQKHeaddim = QKHeadDim;
    static constexpr ck_tile::index_t kN1        = VHeadDim;
};

template <typename BaseProblem, typename Shape>
struct TestProblemWithShape : BaseProblem
{
    using BlockFmhaShape = Shape;
};

template <typename BaseShape,
          ck_tile::index_t M0,
          ck_tile::index_t N0,
          ck_tile::index_t K0,
          ck_tile::index_t K1,
          ck_tile::index_t SubQKHeadDim>
struct TestShapeWithGeometry : BaseShape
{
    static constexpr ck_tile::index_t kM0           = M0;
    static constexpr ck_tile::index_t kN0           = N0;
    static constexpr ck_tile::index_t kK0           = K0;
    static constexpr ck_tile::index_t kK1           = K1;
    static constexpr ck_tile::index_t kSubQKHeaddim = SubQKHeadDim;
};

template <typename BaseShape, bool IsVRowMajor>
struct TestShapeWithVLayout : BaseShape
{
    static constexpr bool IsVLayoutRowMajor = IsVRowMajor;
};

template <typename BaseShape, ck_tile::index_t NumWarps_>
struct TestShapeWithNumWarps : BaseShape
{
    static constexpr ck_tile::index_t NumWarps = NumWarps_;
};

template <typename Selection>
constexpr bool is_disabled_selection()
{
    return std::is_same_v<typename Selection::Q, NoPad> &&
           std::is_same_v<typename Selection::K, NoPad> &&
           std::is_same_v<typename Selection::V, NoPad>;
}

using SelectionBaseProblem = TestFmhaProblem<ck_tile::half_t, 128>;
using MixedTypeProblem     = TestProblemWithDataTypes<SelectionBaseProblem,
                                                      ck_tile::half_t,
                                                      ck_tile::bf16_t,
                                                      ck_tile::half_t>;
using FloatProblem         = TestProblemWithDataTypes<SelectionBaseProblem, float, float, float>;
using PackedProblem        = TestProblemWithDataTypes<SelectionBaseProblem,
                                                      ck_tile::pk_fp4_t,
                                                      ck_tile::pk_fp4_t,
                                                      ck_tile::pk_fp4_t>;
using QK64Shape            = TestShapeWithHeadDims<TestFmhaShape<128>, 64, 128>;
using V64Shape             = TestShapeWithHeadDims<TestFmhaShape<128>, 128, 64>;
using QK64Problem          = TestProblemWithShape<SelectionBaseProblem, QK64Shape>;
using V64Problem           = TestProblemWithShape<SelectionBaseProblem, V64Shape>;
using TwoWarpProblem =
    TestProblemWithShape<SelectionBaseProblem, TestShapeWithNumWarps<TestFmhaShape<128>, 2>>;
using M96Problem =
    TestProblemWithShape<SelectionBaseProblem,
                         TestShapeWithGeometry<TestFmhaShape<128>, 96, 64, 32, 32, 128>>;
using N32Problem =
    TestProblemWithShape<SelectionBaseProblem,
                         TestShapeWithGeometry<TestFmhaShape<128>, 128, 32, 32, 32, 128>>;
using K064Problem =
    TestProblemWithShape<SelectionBaseProblem,
                         TestShapeWithGeometry<TestFmhaShape<128>, 128, 64, 64, 32, 128>>;
using K164Problem =
    TestProblemWithShape<SelectionBaseProblem,
                         TestShapeWithGeometry<TestFmhaShape<128>, 128, 64, 32, 64, 128>>;
using SubQK64Problem =
    TestProblemWithShape<SelectionBaseProblem,
                         TestShapeWithGeometry<TestFmhaShape<128>, 128, 64, 32, 32, 64>>;
using VColumnMajorProblem =
    TestProblemWithShape<SelectionBaseProblem, TestShapeWithVLayout<TestFmhaShape<128>, false>>;

static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<MixedTypeProblem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<FloatProblem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<PackedProblem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<QK64Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<V64Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<TwoWarpProblem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<M96Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<N32Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<K064Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<K164Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<SubQK64Problem>>());
static_assert(is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<VColumnMajorProblem>>());
static_assert(
    is_disabled_selection<ck_tile::detail::QrTdmPaddingSelection<SelectionBaseProblem, false>>());

static_assert(ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 16>);
static_assert(ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 32>);
static_assert(ck_tile::detail::is_valid_lds_padding_config_v<false, 0, 0>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<false, 256, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 0, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 192, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 2048, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 516>);

static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kEnabled);
static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kPadInterval == 5);
static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kPadAmount == 3);
static_assert(ck_tile::detail::EncodedTdmPadding<VPad>::kPadInterval == 5);
static_assert(ck_tile::detail::EncodedTdmPadding<VPad>::kPadAmount == 7);
static_assert(!ck_tile::detail::EncodedTdmPadding<NoPad>::kEnabled);
static_assert(ck_tile::detail::EncodedTdmPadding<NoPad>::kPadInterval == 0);
static_assert(ck_tile::detail::EncodedTdmPadding<NoPad>::kPadAmount == 0);

constexpr auto q_nopad_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t, 128, 128, NoPad, 16>();
static_assert(byte_offset<ck_tile::bf16_t>(q_nopad_bf16_desc, 0, 127) == 254);
static_assert(byte_offset<ck_tile::bf16_t>(q_nopad_bf16_desc, 1, 0) == 256);
static_assert(q_nopad_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 32768);

constexpr auto q_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t, 128, 128, QKPad, 16>();
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 0, 0) == 0);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 0, 127) == 254);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 1, 0) == 272);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 127, 127) == 34798);
static_assert(q_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 34800);

constexpr auto k_prefill_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t, 64, 128, QKPad, 16>();
static_assert(k_prefill_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 17392);

constexpr auto k_decode_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t, 64, 32, QKPad, 16>();
static_assert(k_decode_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 4336);

constexpr auto v_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t, 64, 128, VPad, 16>();
static_assert(byte_offset<ck_tile::bf16_t>(v_bf16_desc, 1, 0) == 288);
static_assert(v_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 18400);

constexpr auto q_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t, 128, 128, QKPad, 16>();
static_assert(byte_offset<ck_tile::half_t>(q_half_desc, 1, 0) == 272);
static_assert(q_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 34800);

constexpr auto k_prefill_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t, 64, 128, QKPad, 16>();
static_assert(k_prefill_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 17392);

constexpr auto k_decode_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t, 64, 32, QKPad, 16>();
static_assert(k_decode_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 4336);

constexpr auto v_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t, 64, 128, VPad, 16>();
static_assert(byte_offset<ck_tile::half_t>(v_half_desc, 1, 0) == 288);
static_assert(v_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 18400);

static_assert(!ck_tile::is_detected<PaddedQDescriptor, ck_tile::pk_fp4_t>::value);

template <typename DataType>
constexpr bool validate_production_geometries()
{
    using PrefillProblem = TestFmhaProblem<DataType, 128>;
    using DecodeProblem  = TestFmhaProblem<DataType, 64>;

    return ck_tile::detail::validate_qr_tdm_issue_geometry<QTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<KTag, PrefillProblem, true>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<VTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<QTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<KTag, DecodeProblem, false>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<VTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<QTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<KTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<VTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<QTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<KTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<VTag, DecodeProblem>();
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
static_assert(validate_production_geometries<ck_tile::bf16_t>());
static_assert(validate_production_geometries<ck_tile::half_t>());
#endif

template <typename Layout>
constexpr bool has_aligned_production_regions()
{
    if constexpr(Layout::kDoubleBuffer)
    {
        return Layout::kQOffset % 256 == 0 && Layout::kK0Offset % 256 == 0 &&
               Layout::kK1Offset % 256 == 0 && Layout::kV0Offset % 256 == 0 &&
               Layout::kV1Offset % 256 == 0;
    }
    else
    {
        return Layout::kQOffset % 256 == 0 && Layout::kK0Offset % 256 == 0 &&
               Layout::kV0Offset % 256 == 0;
    }
}

template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct TestLegacyPhaseLayout
{
    using Production = ck_tile::detail::QrTdmLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    static_assert(Production::kDoubleBuffer,
                  "legacy-phase diagnostics are defined only for the prefill path");

    static constexpr ck_tile::index_t kQOffset    = 0;
    static constexpr ck_tile::index_t kK0Offset   = 0;
    static constexpr ck_tile::index_t kK1Offset   = Production::kKBytes;
    static constexpr ck_tile::index_t kV0Offset   = 2 * Production::kKBytes + 256;
    static constexpr ck_tile::index_t kV1Offset   = kV0Offset + Production::kVBytes;
    static constexpr ck_tile::index_t kArenaBytes = kV1Offset + Production::kVBytes;
    static constexpr bool kHasAlignedRegions      = kQOffset % 256 == 0 && kK0Offset % 256 == 0 &&
                                               kK1Offset % 256 == 0 && kV0Offset % 256 == 0 &&
                                               kV1Offset % 256 == 0;

    static_assert(kQOffset + Production::kQBytes <= kV0Offset);
    static_assert(kK0Offset + Production::kKBytes <= kK1Offset);
    static_assert(kK1Offset + Production::kKBytes <= kV0Offset);
    static_assert(kV0Offset + Production::kVBytes <= kV1Offset);
    static_assert(kV1Offset + Production::kVBytes <= kArenaBytes);
};

template <typename DataType>
constexpr bool validate_arena_layouts()
{
    using PrefillProblem = TestFmhaProblem<DataType, 128>;
    using DecodeProblem  = TestFmhaProblem<DataType, 64>;
    using Policy         = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy;

    using PrefillAll = typename Policy::template LdsArenaLayout<PrefillProblem, QKPad, QKPad, VPad>;
    using DecodeAll  = typename Policy::template LdsArenaLayout<DecodeProblem, QKPad, QKPad, VPad>;

    static_assert(PrefillAll::kQOffset == 0);
    static_assert(PrefillAll::kK0Offset == 0);
    static_assert(PrefillAll::kK1Offset == 17408);
    static_assert(PrefillAll::kV0Offset == 34816);
    static_assert(PrefillAll::kV1Offset == 53248);
    static_assert(PrefillAll::kArenaBytes == 71680);
    static_assert(DecodeAll::kQOffset == 0);
    static_assert(DecodeAll::kK0Offset == 0);
    static_assert(DecodeAll::kV0Offset == 4352);
    static_assert(DecodeAll::kArenaBytes == 22784);

    using PrefillNone =
        typename Policy::template LdsArenaLayout<PrefillProblem, NoPad, NoPad, NoPad>;
    using PrefillQKV = typename Policy::template LdsArenaLayout<PrefillProblem, QKPad, QKPad, VPad>;
    using PrefillKV  = typename Policy::template LdsArenaLayout<PrefillProblem, NoPad, QKPad, VPad>;
    using PrefillK = typename Policy::template LdsArenaLayout<PrefillProblem, NoPad, QKPad, NoPad>;
    using PrefillV = typename Policy::template LdsArenaLayout<PrefillProblem, NoPad, NoPad, VPad>;
    static_assert(PrefillNone::kArenaBytes == 65536);
    static_assert(PrefillQKV::kArenaBytes == 71680);
    static_assert(PrefillKV::kArenaBytes == 71680);
    static_assert(PrefillK::kArenaBytes == 67584);
    static_assert(PrefillV::kArenaBytes == 69632);

    using DecodeNone = typename Policy::template LdsArenaLayout<DecodeProblem, NoPad, NoPad, NoPad>;
    using DecodeQKV  = typename Policy::template LdsArenaLayout<DecodeProblem, QKPad, QKPad, VPad>;
    using DecodeKV   = typename Policy::template LdsArenaLayout<DecodeProblem, NoPad, QKPad, VPad>;
    using DecodeK    = typename Policy::template LdsArenaLayout<DecodeProblem, NoPad, QKPad, NoPad>;
    using DecodeV    = typename Policy::template LdsArenaLayout<DecodeProblem, NoPad, NoPad, VPad>;
    static_assert(DecodeNone::kArenaBytes == 20480);
    static_assert(DecodeQKV::kArenaBytes == 22784);
    static_assert(DecodeKV::kArenaBytes == 22784);
    static_assert(DecodeK::kArenaBytes == 20736);
    static_assert(DecodeV::kArenaBytes == 22528);

    static_assert(has_aligned_production_regions<PrefillAll>());
    static_assert(has_aligned_production_regions<DecodeAll>());
    static_assert(PrefillAll::kK0Offset + PrefillAll::kKBytes <= PrefillAll::kK1Offset);
    static_assert(PrefillAll::kK1Offset + PrefillAll::kKBytes <= PrefillAll::kV0Offset);
    static_assert(PrefillAll::kQOffset + PrefillAll::kQBytes <= PrefillAll::kV0Offset);
    static_assert((PrefillAll::kK1Offset - PrefillAll::kK0Offset) % QKPad::kIntervalBytes == 0);
    static_assert((PrefillAll::kV1Offset - PrefillAll::kV0Offset) % VPad::kIntervalBytes == 0);
    static_assert(PrefillAll::kArenaBytes <= 128 * 1024);
    static_assert(ck_tile::integer_least_multiple(PrefillAll::kArenaBytes, 64 * 1024) * 2 <=
                  320 * 1024);

    using Legacy = TestLegacyPhaseLayout<PrefillProblem, QKPad, QKPad, VPad>;
    static_assert(Legacy::kK0Offset == 0);
    static_assert(Legacy::kK1Offset == 17392);
    static_assert(Legacy::kV0Offset == 35040);
    static_assert(Legacy::kV1Offset == 53440);
    static_assert(Legacy::kArenaBytes == 71840);
    static_assert(!Legacy::kHasAlignedRegions);

    return true;
}

static_assert(validate_arena_layouts<ck_tile::bf16_t>());
static_assert(validate_arena_layouts<ck_tile::half_t>());

template <typename DataType, ck_tile::index_t M>
constexpr bool validate_policy_coupling()
{
    using Problem  = TestFmhaProblem<DataType, M>;
    using Policy   = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using QConfig  = typename Policy::template LdsPaddingConfigQ<Problem>;
    using KConfig  = typename Policy::template LdsPaddingConfigK<Problem>;
    using VConfig  = typename Policy::template LdsPaddingConfigV<Problem>;
    using QRaw     = ck_tile::detail::EncodedTdmPadding<QConfig>;
    using KRaw     = ck_tile::detail::EncodedTdmPadding<KConfig>;
    using VRaw     = ck_tile::detail::EncodedTdmPadding<VConfig>;
    using Layout   = typename Policy::template LdsArenaLayout<Problem>;
    using Pipeline = ck_tile::BlockFmhaPipelineQRKSVSTdm<Problem>;

    constexpr auto q_desc = Policy::template MakeQLdsBlockDescriptor<Problem>();
    constexpr auto k_desc = Policy::template MakeKLdsBlockDescriptor<Problem, (M > 64)>();
    constexpr auto v_desc = Policy::template MakeVLdsBlockDescriptor<Problem>();

    static_assert(std::is_same_v<QConfig, NoPad>);
    static_assert(std::is_same_v<KConfig, QKPad>);
    static_assert(std::is_same_v<VConfig, VPad>);
    static_assert(!QRaw::kEnabled && QRaw::kPadInterval == 0 && QRaw::kPadAmount == 0);
    static_assert(KRaw::kEnabled && KRaw::kPadInterval == 5 && KRaw::kPadAmount == 3);
    static_assert(VRaw::kEnabled && VRaw::kPadInterval == 5 && VRaw::kPadAmount == 7);
    static_assert(q_desc.calculate_offset(ck_tile::make_tuple(1, 0)) == 128);
    static_assert(k_desc.get_element_space_size() * sizeof(DataType) == (M > 64 ? 17392 : 4336));
    static_assert(v_desc.get_element_space_size() * sizeof(DataType) == 18400);
    static_assert(Layout::kArenaBytes == (M > 64 ? 71680 : 22784));
    static_assert(Pipeline::GetSmemSize() == Layout::kArenaBytes);

    using EnabledQ   = ck_tile::detail::LdsPaddingConfig<true, 256, 16>;
    using EnabledRaw = ck_tile::detail::EncodedTdmPadding<EnabledQ>;
    constexpr auto enabled_desc =
        ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<DataType, M, 128, EnabledQ, 16>();
    static_assert(EnabledRaw::kPadInterval == 5 && EnabledRaw::kPadAmount == 3);
    static_assert(enabled_desc.get_element_space_size() > q_desc.get_element_space_size());
    static_assert(!std::is_same_v<ck_tile::remove_cvref_t<decltype(enabled_desc)>,
                                  ck_tile::remove_cvref_t<decltype(q_desc)>>);

    return true;
}

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
static_assert(validate_policy_coupling<ck_tile::bf16_t, 128>());
static_assert(validate_policy_coupling<ck_tile::bf16_t, 64>());
static_assert(validate_policy_coupling<ck_tile::half_t, 128>());
static_assert(validate_policy_coupling<ck_tile::half_t, 64>());
#else
static_assert(is_disabled_selection<
              ck_tile::detail::QrTdmPaddingSelection<TestFmhaProblem<ck_tile::bf16_t, 128>>>());
static_assert(is_disabled_selection<
              ck_tile::detail::QrTdmPaddingSelection<TestFmhaProblem<ck_tile::bf16_t, 64>>>());
static_assert(is_disabled_selection<
              ck_tile::detail::QrTdmPaddingSelection<TestFmhaProblem<ck_tile::half_t, 128>>>());
static_assert(is_disabled_selection<
              ck_tile::detail::QrTdmPaddingSelection<TestFmhaProblem<ck_tile::half_t, 64>>>());
#endif

using DispatchProblem = TestFmhaProblem<ck_tile::half_t, 128>;
static_assert(
    ck_tile::detail::uses_qr_tdm_lds_arena_v<ck_tile::BlockFmhaPipelineQRKSVSTdm<DispatchProblem>>);
static_assert(
    !ck_tile::detail::uses_qr_tdm_lds_arena_v<ck_tile::BlockFmhaPipelineQRKSVS<DispatchProblem>>);
static_assert(!ck_tile::detail::uses_qr_tdm_lds_arena_v<
              ck_tile::BlockFmhaPipelineQRKSVSAsync<DispatchProblem>>);
static_assert(!ck_tile::detail::uses_qr_tdm_lds_arena_v<
              ck_tile::BlockFmhaPipelineQRKSVSAsyncTrload<DispatchProblem>>);

struct RoundTripArgs
{
    const void* input;
    void* output;
    int* guard_ok;
};

template <typename TensorTag,
          typename Problem,
          typename QConfig,
          typename KConfig,
          typename VConfig,
          ck_tile::index_t RegionIndex>
struct QrTdmRoundTripKernel
{
    using Policy   = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Shape    = typename Problem::BlockFmhaShape;
    using DataType = std::conditional_t<TensorTag::Id == 0,
                                        typename Problem::QDataType,
                                        std::conditional_t<TensorTag::Id == 1,
                                                           typename Problem::KDataType,
                                                           typename Problem::VDataType>>;
    using Padding  = std::conditional_t<TensorTag::Id == 0,
                                        QConfig,
                                        std::conditional_t<TensorTag::Id == 1, KConfig, VConfig>>;
    using Layout   = typename Policy::template LdsArenaLayout<Problem, QConfig, KConfig, VConfig>;

    static constexpr ck_tile::index_t kBlockSize    = Problem::kBlockSize;
    static constexpr bool kPrefill                  = Shape::kM0 > 64;
    static constexpr ck_tile::index_t kRows         = TensorTag::Id == 0 ? Shape::kM0 : Shape::kN0;
    static constexpr ck_tile::index_t kCols         = TensorTag::Id == 0 ? Shape::kSubQKHeaddim
                                                      : TensorTag::Id == 1
                                                          ? (kPrefill ? Shape::kSubQKHeaddim : Shape::kK0)
                                                          : Shape::kN1;
    static constexpr ck_tile::index_t kRegionOffset = [] {
        if constexpr(TensorTag::Id == 0)
            return Layout::kQOffset;
        else if constexpr(TensorTag::Id == 1)
            return RegionIndex == 0 ? Layout::kK0Offset : Layout::kK1Offset;
        else
            return RegionIndex == 0 ? Layout::kV0Offset : Layout::kV1Offset;
    }();
    static constexpr ck_tile::index_t kRegionBytes = [] {
        if constexpr(TensorTag::Id == 0)
            return Layout::kQBytes;
        else if constexpr(TensorTag::Id == 1)
            return Layout::kKBytes;
        else
            return Layout::kVBytes;
    }();

    __device__ void operator()(RoundTripArgs args) const
    {
        using namespace ck_tile;
        alignas(256) __shared__ unsigned char arena[Layout::kArenaBytes + 16];
        const index_t tid = get_thread_local_1d_id();
        for(index_t i = tid; i < Layout::kArenaBytes + 16; i += Problem::kBlockSize)
            arena[i] = 0x5a;
        block_sync_lds();

        auto* region = reinterpret_cast<DataType*>(arena + kRegionOffset);
        constexpr auto lds_desc =
            detail::make_qr_tdm_row_major_lds_descriptor<DataType, kRows, kCols, Padding, 16>();
        auto lds_view         = make_tensor_view<address_space_enum::lds>(region, lds_desc);
        auto lds_write_window = make_tile_window(lds_view, lds_desc.get_lengths(), {0, 0});

        const auto input_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const DataType*>(args.input),
            make_tuple(kRows, kCols),
            make_tuple(kCols, 1),
            number<8>{},
            number<1>{});
        auto input_window = make_tile_window(input_view,
                                             make_tuple(number<kRows>{}, number<kCols>{}),
                                             {0, 0},
                                             detail::make_qr_tdm_writer_distribution < TensorTag,
                                             Problem,
                                             TensorTag::Id == 1 && kPrefill > ());

        TDMConfig config;
        using Raw                      = detail::EncodedTdmPadding<Padding>;
        config.pad_enable              = Raw::kEnabled;
        config.pad_config.pad_interval = Raw::kPadInterval;
        config.pad_config.pad_amount   = Raw::kPadAmount;
        load_tile_tdm(config, lds_write_window, input_window);
        s_wait_tensorcnt_barrier<0>();

        if constexpr(TensorTag::Id == 0)
        {
            auto output_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<DataType*>(args.output),
                make_tuple(kRows, kCols),
                make_tuple(kCols, 1),
                number<8>{},
                number<1>{});
            auto read_window =
                make_tile_window(lds_view,
                                 lds_desc.get_lengths(),
                                 {0, 0},
                                 Policy::template MakeQRegTileDistribution<Problem>());
            auto output_window =
                make_tile_window(output_view,
                                 lds_desc.get_lengths(),
                                 {0, 0},
                                 Policy::template MakeQRegTileDistribution<Problem>());
            store_tile(output_window, load_tile(read_window));
        }
        else if constexpr(TensorTag::Id == 1)
        {
            auto output_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<DataType*>(args.output),
                make_tuple(kRows, kCols),
                make_tuple(kCols, 1),
                number<8>{},
                number<1>{});
            constexpr index_t Windows = kPrefill ? Shape::kQKHeaddim / Shape::kK0 : 1;
            static_for<0, Windows, 1>{}([&](auto i) {
                constexpr auto lengths = make_tuple(number<Shape::kN0>{}, number<Shape::kK0>{});
                const array<index_t, 2> origin{0, i * Shape::kK0};
                auto read_window =
                    make_tile_window(lds_view,
                                     lengths,
                                     origin,
                                     Policy::template MakeKRegTileDistribution<Problem>());
                auto output_window =
                    make_tile_window(output_view,
                                     lengths,
                                     origin,
                                     Policy::template MakeKRegTileDistribution<Problem>());
                store_tile(output_window, load_tile(read_window));
            });
        }
        else
        {
            auto output_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<DataType*>(args.output),
                make_tuple(kCols, kRows),
                make_tuple(kRows, 1),
                number<8>{},
                number<1>{});
            constexpr index_t Windows = Shape::kN0 / Shape::kK1;
            static_for<0, Windows, 1>{}([&](auto i) {
                constexpr auto lengths = make_tuple(number<Shape::kK1>{}, number<Shape::kN1>{});
                const array<index_t, 2> origin{i * Shape::kK1, 0};
                auto read_window =
                    make_tile_window(lds_view,
                                     lengths,
                                     origin,
                                     Policy::template MakeVRegTileDistribution<Problem>());
                auto tile = load_tile_transpose(read_window);
                constexpr auto output_lengths =
                    make_tuple(number<Shape::kN1>{}, number<Shape::kK1>{});
                const array<index_t, 2> output_origin{0, i * Shape::kK1};
                auto output_window = make_tile_window(
                    output_view, output_lengths, output_origin, tile.get_tile_distribution());
                store_tile(output_window, tile);
            });
        }

        block_sync_lds();
        if(tid == 0)
        {
            int ok = 1;
            for(index_t i = 0; i < Layout::kArenaBytes; ++i)
                if((i < kRegionOffset || i >= kRegionOffset + kRegionBytes) && arena[i] != 0x5a)
                    ok = 0;
            for(index_t i = Layout::kArenaBytes; i < Layout::kArenaBytes + 16; ++i)
                if(arena[i] != 0x5a)
                    ok = 0;
            *args.guard_ok = ok;
        }
    }
};

template <typename TensorTag,
          typename Problem,
          typename QConfig,
          typename KConfig,
          typename VConfig>
bool run_qr_tdm_round_trip()
{
    using Kernel0  = QrTdmRoundTripKernel<TensorTag, Problem, QConfig, KConfig, VConfig, 0>;
    using DataType = typename Kernel0::DataType;
    constexpr ck_tile::index_t Rows = Kernel0::kRows;
    constexpr ck_tile::index_t Cols = Kernel0::kCols;

    std::vector<DataType> input(Rows * Cols);
    std::vector<DataType> output(Rows * Cols);
    for(ck_tile::index_t row = 0; row < Rows; ++row)
        for(ck_tile::index_t col = 0; col < Cols; ++col)
            input[row * Cols + col] = static_cast<DataType>((row * 17 + col * 3) % 127);

    ck_tile::DeviceMem input_device(input.size() * sizeof(DataType));
    ck_tile::DeviceMem output_device(output.size() * sizeof(DataType));
    ck_tile::DeviceMem guard_device(sizeof(int));
    input_device.ToDevice(input.data());
    output_device.SetZero();
    guard_device.SetZero();

    RoundTripArgs args{input_device.GetDeviceBuffer(),
                       output_device.GetDeviceBuffer(),
                       static_cast<int*>(guard_device.GetDeviceBuffer())};
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};
    const auto block_size = ck_tile::is_wave32() ? Problem::kBlockSize / 2 : Problem::kBlockSize;
    ck_tile::launch_kernel(stream,
                           ck_tile::make_kernel(Kernel0{}, dim3(1), dim3(block_size), 0, args));

    output_device.FromDevice(output.data());
    int guard_ok = 0;
    guard_device.FromDevice(&guard_ok);
    auto validate_result = [&]() {
        if(guard_ok != 1)
            return false;

        if constexpr(TensorTag::Id == 2)
        {
            for(ck_tile::index_t row = 0; row < Rows; ++row)
                for(ck_tile::index_t col = 0; col < Cols; ++col)
                    if(std::memcmp(&input[row * Cols + col],
                                   &output[col * Rows + row],
                                   sizeof(DataType)) != 0)
                        return false;
        }
        else if(std::memcmp(input.data(), output.data(), input.size() * sizeof(DataType)) != 0)
        {
            return false;
        }
        return true;
    };

    if(!validate_result())
    {
        std::cerr << "round-trip failure: tensor=" << TensorTag::Id
                  << ", M=" << Problem::BlockFmhaShape::kM0 << ", q=" << QConfig::kEnabled
                  << ", k=" << KConfig::kEnabled << ", v=" << VConfig::kEnabled << '\n';
        return false;
    }

    if constexpr(Kernel0::kPrefill && TensorTag::Id != 0)
    {
        using Kernel1 = QrTdmRoundTripKernel<TensorTag, Problem, QConfig, KConfig, VConfig, 1>;
        output_device.SetZero();
        guard_device.SetZero();
        ck_tile::launch_kernel(stream,
                               ck_tile::make_kernel(Kernel1{}, dim3(1), dim3(block_size), 0, args));
        output_device.FromDevice(output.data());
        guard_device.FromDevice(&guard_ok);
        if(!validate_result())
        {
            std::cerr << "round-trip secondary-region failure: tensor=" << TensorTag::Id
                      << ", M=" << Problem::BlockFmhaShape::kM0 << ", q=" << QConfig::kEnabled
                      << ", k=" << KConfig::kEnabled << ", v=" << VConfig::kEnabled << '\n';
            return false;
        }
    }
    return true;
}

template <typename DataType, ck_tile::index_t M>
bool run_round_trip_matrix()
{
    using Problem = TestFmhaProblem<DataType, M>;

    return run_qr_tdm_round_trip<QTag, Problem, NoPad, NoPad, NoPad>() &&
           run_qr_tdm_round_trip<KTag, Problem, NoPad, NoPad, NoPad>() &&
           run_qr_tdm_round_trip<VTag, Problem, NoPad, NoPad, NoPad>() &&
           run_qr_tdm_round_trip<QTag, Problem, QKPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<KTag, Problem, QKPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<VTag, Problem, QKPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<QTag, Problem, NoPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<KTag, Problem, NoPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<VTag, Problem, NoPad, QKPad, VPad>() &&
           run_qr_tdm_round_trip<QTag, Problem, NoPad, QKPad, NoPad>() &&
           run_qr_tdm_round_trip<KTag, Problem, NoPad, QKPad, NoPad>() &&
           run_qr_tdm_round_trip<VTag, Problem, NoPad, QKPad, NoPad>() &&
           run_qr_tdm_round_trip<QTag, Problem, NoPad, NoPad, VPad>() &&
           run_qr_tdm_round_trip<KTag, Problem, NoPad, NoPad, VPad>() &&
           run_qr_tdm_round_trip<VTag, Problem, NoPad, NoPad, VPad>();
}

TEST(QrTdmLdsPadding, CompileTimeConfiguration) { SUCCEED(); }

TEST(QrTdmLdsPadding, DeviceRoundTrip)
{
    EXPECT_TRUE((run_round_trip_matrix<ck_tile::bf16_t, 128>()));
    EXPECT_TRUE((run_round_trip_matrix<ck_tile::bf16_t, 64>()));
    EXPECT_TRUE((run_round_trip_matrix<ck_tile::half_t, 128>()));
    EXPECT_TRUE((run_round_trip_matrix<ck_tile::half_t, 64>()));
}

} // namespace
