// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

constexpr int DS_READ_TR_SIZE()
{
    return 8; // Literal constant, evaluated at compile time
}

namespace util {
template <typename Suffix, typename Sequence>
struct is_sequence_suffix
{
    static constexpr bool size_check = (Suffix::size() <= Sequence::size());

    static constexpr index_t start_pos = Sequence::size() - Suffix::size();
    using extract_indices = typename arithmetic_sequence_gen<start_pos, Sequence::size(), 1>::type;

    static constexpr bool value =
        size_check && (Suffix{} == decltype(Sequence::extract(extract_indices{})){});
};

template <index_t... Xs>
struct is_sequence_suffix<sequence<>, sequence<Xs...>>
{
    static constexpr bool value = true;
};

template <typename Suffix, typename Sequence>
constexpr bool is_sequence_suffix_v = is_sequence_suffix<Suffix, Sequence>::value;

} // namespace util

namespace detail {

template <typename LongSequence, typename ShortSequence>
struct is_terminal_split_sequence : std::false_type
{
};

/// Checks if first sequence has the same prefix as the second sequence and
/// the last two elements match the last element of the second sequence.
template <index_t... Longs, index_t... Shorts>
struct is_terminal_split_sequence<sequence<Longs...>, sequence<Shorts...>>
{
    using LongSequence  = sequence<Longs...>;
    using ShortSequence = sequence<Shorts...>;

    static constexpr index_t LongSize  = LongSequence::size();
    static constexpr index_t ShortSize = ShortSequence::size();
    static constexpr bool size_valid   = (ShortSize > 0) && (LongSize == ShortSize + 1);

    static constexpr bool prefix_matches = []() {
        if constexpr(!size_valid)
            return false;
        else if constexpr(ShortSize == 1)
            return true;
        else
        {
            using PrefixIndices = typename arithmetic_sequence_gen<0, ShortSize - 1, 1>::type;
            using LongPrefix    = decltype(LongSequence::extract(PrefixIndices{}));
            using ShortPrefix   = decltype(ShortSequence::extract(PrefixIndices{}));
            return std::is_same_v<LongPrefix, ShortPrefix>;
        }
    }();

    static constexpr bool terminal_matches = []() {
        if constexpr(!size_valid)
            return false;
        else
        {
            return LongSequence::at(LongSize - 2) * LongSequence::at(LongSize - 1) ==
                   ShortSequence::at(ShortSize - 1);
        }
    }();

    static constexpr bool value = prefix_matches && terminal_matches;
};

template <typename Lhs, typename Rhs>
struct is_same_or_terminal_split_sequence
{
    static constexpr bool value = std::is_same_v<Lhs, Rhs> ||
                                  is_terminal_split_sequence<Lhs, Rhs>::value ||
                                  is_terminal_split_sequence<Rhs, Lhs>::value;
};

template <typename LongMajorSequence,
          typename LongMinorSequence,
          typename ShortMajorSequence,
          typename ShortMinorSequence>
struct is_terminal_split_mapping : std::false_type
{
};

/// Checks if the longer Y->RHS mapping is identical to the shorter mapping except that the
/// shorter terminal dimension is represented by two adjacent terminal dimensions in the longer
/// mapping. This matches NormalizeEncodingForTranspose, which splits the terminal H dimension
/// from minor m into minors m and m+1.
template <index_t... LongMajors,
          index_t... LongMinors,
          index_t... ShortMajors,
          index_t... ShortMinors>
struct is_terminal_split_mapping<sequence<LongMajors...>,
                                 sequence<LongMinors...>,
                                 sequence<ShortMajors...>,
                                 sequence<ShortMinors...>>
{
    using LongMajorSequence  = sequence<LongMajors...>;
    using LongMinorSequence  = sequence<LongMinors...>;
    using ShortMajorSequence = sequence<ShortMajors...>;
    using ShortMinorSequence = sequence<ShortMinors...>;

    static constexpr index_t LongSize  = LongMajorSequence::size();
    static constexpr index_t ShortSize = ShortMajorSequence::size();
    static_assert(LongSize == LongMinorSequence::size(), "Y->RHS mapping ranks must match");
    static_assert(ShortSize == ShortMinorSequence::size(), "Y->RHS mapping ranks must match");

    static constexpr bool size_valid = (ShortSize > 0) && (LongSize == ShortSize + 1);

    static constexpr bool prefix_matches = []() {
        if constexpr(!size_valid)
            return false;
        else if constexpr(ShortSize == 1)
            return true;
        else
        {
            // Extract all majors and minors before the last major and minor of the shorter mapping
            using PrefixIndices    = typename arithmetic_sequence_gen<0, ShortSize - 1, 1>::type;
            using LongMajorPrefix  = decltype(LongMajorSequence::extract(PrefixIndices{}));
            using LongMinorPrefix  = decltype(LongMinorSequence::extract(PrefixIndices{}));
            using ShortMajorPrefix = decltype(ShortMajorSequence::extract(PrefixIndices{}));
            using ShortMinorPrefix = decltype(ShortMinorSequence::extract(PrefixIndices{}));
            // Every major and minor prefix must match exactly
            return std::is_same_v<LongMajorPrefix, ShortMajorPrefix> &&
                   std::is_same_v<LongMinorPrefix, ShortMinorPrefix>;
        }
    }();

    static constexpr bool terminal_matches = []() {
        if constexpr(!size_valid)
            return false;
        else
        {
            // Get the last major and minor of the shorter mapping
            constexpr index_t short_major = ShortMajorSequence::at(ShortSize - 1);
            constexpr index_t short_minor = ShortMinorSequence::at(ShortSize - 1);
            // Require same major, and sequential minors
            return LongMajorSequence::at(LongSize - 2) == short_major &&
                   LongMajorSequence::at(LongSize - 1) == short_major &&
                   LongMinorSequence::at(LongSize - 2) == short_minor &&
                   LongMinorSequence::at(LongSize - 1) == short_minor + 1;
        }
    }();

    static constexpr bool value = prefix_matches && terminal_matches;
};

template <typename LhsMajorSequence,
          typename LhsMinorSequence,
          typename RhsMajorSequence,
          typename RhsMinorSequence>
struct is_same_or_terminal_split_mapping
{
    static constexpr bool value = (std::is_same_v<LhsMajorSequence, RhsMajorSequence> &&
                                   std::is_same_v<LhsMinorSequence, RhsMinorSequence>) ||
                                  is_terminal_split_mapping<LhsMajorSequence,
                                                            LhsMinorSequence,
                                                            RhsMajorSequence,
                                                            RhsMinorSequence>::value ||
                                  is_terminal_split_mapping<RhsMajorSequence,
                                                            RhsMinorSequence,
                                                            LhsMajorSequence,
                                                            LhsMinorSequence>::value;
};

template <typename MajorSequence, typename MinorSequence, index_t TargetMajor, index_t TargetMinor>
struct rhs_sequence_contains : std::false_type
{
};

template <index_t... Majors, index_t... Minors, index_t TargetMajor, index_t TargetMinor>
struct rhs_sequence_contains<sequence<Majors...>, sequence<Minors...>, TargetMajor, TargetMinor>
{
    static_assert(sizeof...(Majors) == sizeof...(Minors), "RHS mapping ranks must match");

    static constexpr bool value =
        (false || ... || ((Majors == TargetMajor) && (Minors == TargetMinor)));
};

template <typename MajorTuple, typename MinorTuple, index_t TargetMajor, index_t TargetMinor>
struct rhs_mapping_contains
{
    static constexpr bool value = []() {
        if constexpr(MajorTuple::size() != MinorTuple::size())
            return false;
        else
        {
            // Evaluate the generator (rather than wrapping it in decltype) so the lambda
            // lives in an evaluated operand; lambdas in unevaluated operands are not valid C++17.
            constexpr auto match_flags = generate_sequence_v2(
                [](auto i) {
                    using MajorSequence = remove_cvref_t<decltype(MajorTuple{}[i])>;
                    using MinorSequence = remove_cvref_t<decltype(MinorTuple{}[i])>;
                    return number<(rhs_sequence_contains<MajorSequence,
                                                         MinorSequence,
                                                         TargetMajor,
                                                         TargetMinor>::value
                                       ? 1
                                       : 0)>{};
                },
                number<MajorTuple::size()>{});

            return match_flags.sum() != 0;
        }
    }();
};

template <typename ExpectedEncoding, typename ActualEncoding>
struct have_compatible_hs_lengthss
{
    static constexpr bool value = []() {
        if constexpr(ExpectedEncoding::NDimX != ActualEncoding::NDimX)
            return false;
        else
        {
            constexpr auto expected_hs = ExpectedEncoding::hs_lengthss_;
            constexpr auto actual_hs   = ActualEncoding::hs_lengthss_;

            // Evaluate the generator (rather than wrapping it in decltype) so the lambda
            // lives in an evaluated operand; lambdas in unevaluated operands are not valid C++17.
            constexpr auto compatible_flags = generate_sequence_v2(
                [](auto i) {
                    using ExpectedHs = remove_cvref_t<decltype(expected_hs[i])>;
                    using ActualHs   = remove_cvref_t<decltype(actual_hs[i])>;
                    return number<(
                        is_same_or_terminal_split_sequence<ExpectedHs, ActualHs>::value ? 1 : 0)>{};
                },
                number<ExpectedEncoding::NDimX>{});

            return compatible_flags.sum() == ExpectedEncoding::NDimX;
        }
    }();
};

template <typename ExpectedDistribution, typename ActualDistribution, typename DataType>
struct is_transpose_output_compatible
{
    using ExpectedDstr = remove_cvref_t<ExpectedDistribution>;
    using ActualDstr   = remove_cvref_t<ActualDistribution>;
    using ExpectedEnc  = typename ExpectedDstr::DstrEncode;
    using ActualEnc    = typename ActualDstr::DstrEncode;

    static constexpr auto expected_y_desc = ExpectedDstr{}.get_ys_to_d_descriptor();
    static constexpr auto actual_y_desc   = ActualDstr{}.get_ys_to_d_descriptor();

    using ExpectedYLengths = remove_cvref_t<decltype(to_sequence(expected_y_desc.get_lengths()))>;
    using ActualYLengths   = remove_cvref_t<decltype(to_sequence(actual_y_desc.get_lengths()))>;

    static constexpr index_t PackedSize = numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    static constexpr bool exact_match = std::is_same_v<ExpectedDstr, ActualDstr>;
    static constexpr bool same_logical_x_lengths =
        have_compatible_hs_lengthss<ExpectedEnc, ActualEnc>::value;
    static constexpr bool same_thread_element_space =
        expected_y_desc.get_element_space_size() == actual_y_desc.get_element_space_size();
    static constexpr bool same_flattened_y_order =
        is_same_or_terminal_split_sequence<ExpectedYLengths, ActualYLengths>::value;
    static constexpr bool same_y_to_rhs_mapping =
        is_same_or_terminal_split_mapping<typename ExpectedEnc::Ys2RHsMajor,
                                          typename ExpectedEnc::Ys2RHsMinor,
                                          typename ActualEnc::Ys2RHsMajor,
                                          typename ActualEnc::Ys2RHsMinor>::value;

    static constexpr bool compatible_vector_grouping = []() {
        if constexpr(ExpectedYLengths::size() == 0 || ActualYLengths::size() == 0)
            return false;
        else
        {
            constexpr index_t expected_vector = ExpectedYLengths::at(ExpectedYLengths::size() - 1);
            constexpr index_t actual_vector   = ActualYLengths::at(ActualYLengths::size() - 1);

            return expected_vector % PackedSize == 0 && actual_vector % PackedSize == 0 &&
                   (expected_vector % actual_vector == 0 || actual_vector % expected_vector == 0);
        }
    }();

    static constexpr bool value =
        exact_match ||
        (same_logical_x_lengths && same_thread_element_space && same_flattened_y_order &&
         same_y_to_rhs_mapping && compatible_vector_grouping);
};

template <typename ExpectedDistribution, typename ActualDistribution, typename DataType>
inline constexpr bool is_transpose_output_compatible_v =
    is_transpose_output_compatible<ExpectedDistribution, ActualDistribution, DataType>::value;

} // namespace detail

// Default policy: Retains original 2D transpose behavior
template <typename DataType>
struct DefaultTranspose
{
#if defined(__gfx950__)
    template <index_t LaneGroupSize, index_t NumBitType>
    struct Quad
    {
        static_assert(LaneGroupSize == 64 || LaneGroupSize == 32 || LaneGroupSize == 16,
                      "LaneGroupSize must be 64, 32, or 16");

        // The tile is defined by the LaneGroupSize, which defines the number of lanes in the M/N
        // dimensions for the MMA instruction defined by warp gemm.
        // The LaneGroupSize is subdivided into groups of 16 (finer granularity of MMA
        // instructions), we define these as major subtiles. Each of these major subtile is divided
        // into minor subtiles which group the lanes exchanging data during the transpose Example
        // LaneGroupSize = 16, 16 bit type:
        //  - There is 1 group of 16 lanes (1 major subtile)
        //  - Each major subtile is divided into 4 minor subtiles of (4x4) -> 4 lanes transpose
        //    the minor subtile and each lane holds 4 elements

        // all load transpose instructions use 64 bit right now
        static constexpr index_t InstructionBits = 64;
        // Subtile major dimension is fixed
        static constexpr index_t SubtileMajorDimension = 16;
        // Number of subtile major
        static constexpr index_t NumSubtilesMajor = LaneGroupSize / 16;
        // number of elements loaded by each lane with single instruction, but also number
        // of consecutive lanes in a subtile. Subtile is squared (NLanes x NElementsPerLane)
        static constexpr index_t SubtileMinorDimension = InstructionBits / NumBitType;
        // Number of subtiles minor inside each subtile major
        static constexpr index_t NumSubtilesMinor = 16 / SubtileMinorDimension;

        using InputEncoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<SubtileMinorDimension>,
                  sequence<NumSubtilesMajor, NumSubtilesMinor, SubtileMinorDimension>>,
            tuple<sequence<2, 1, 2>>,
            tuple<sequence<0, 0, 1>>,
            sequence<2>,
            sequence<2>>;

        using OutputEncoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<LaneGroupSize>, sequence<SubtileMinorDimension>>,
            tuple<sequence<1>>,
            tuple<sequence<0>>,
            sequence<2>,
            sequence<0>>;
    };

    static constexpr index_t PackedSize      = numeric_traits<remove_cvref_t<DataType>>::PackedSize;
    static constexpr index_t NumBitsDataType = (sizeof(DataType) * 8) / PackedSize;

    // Select based on data size
    template <index_t LaneGroupSize>
    using QuadInputEncoding = typename Quad<LaneGroupSize, NumBitsDataType>::InputEncoding;

    template <index_t LaneGroupSize>
    using QuadOutputEncoding = typename Quad<LaneGroupSize, NumBitsDataType>::OutputEncoding;
#else // now this branch just for gfx1250
    template <index_t LaneGroupSize, index_t NumBitType>
    struct Quad
    {
        static_assert(LaneGroupSize == 16 || LaneGroupSize == 32 || LaneGroupSize == 64,
                      "LaneGroupSize must be 16, 32, or 64");

        // gfx1250 load transpose instructions use 128 bits for 16-bit types, 64 bits for 8-bit
        static constexpr index_t InstructionBits = (NumBitType >= 16) ? 128 : 64;
        // Subtile major dimension is fixed
        static constexpr index_t SubtileMajorDimension = 16;
        // Number of subtile major
        static constexpr index_t NumSubtilesMajor = LaneGroupSize / 16;
        // number of elements loaded by each lane with single instruction, but also number
        // of consecutive lanes in a subtile. Subtile is squared (NLanes x NElementsPerLane)
        static constexpr index_t SubtileMinorDimension = InstructionBits / NumBitType;
        // Number of subtiles minor inside each subtile major
        static constexpr index_t NumSubtilesMinor = 16 / SubtileMinorDimension;

        static constexpr auto make_input_encoding()
        {
            if constexpr(NumBitType >= 16)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<SubtileMinorDimension>,
                          sequence<NumSubtilesMajor, NumSubtilesMinor, SubtileMinorDimension>>,
                    tuple<sequence<2, 2, 1>>,
                    tuple<sequence<0, 1, 0>>,
                    sequence<2>,
                    sequence<2>>{};
            }
            else if constexpr(NumBitType == 8)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<NumSubtilesMinor, SubtileMinorDimension / NumSubtilesMinor>,
                          sequence<NumSubtilesMajor, NumSubtilesMinor, SubtileMinorDimension>>,
                    tuple<sequence<2, 1, 2, 1>>,
                    tuple<sequence<0, 0, 1, 1>>,
                    sequence<2>,
                    sequence<2>>{};
            }
            else
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<2, NumSubtilesMajor, 2, 8>, sequence<16>>,
                    tuple<sequence<1, 1, 1>>,
                    tuple<sequence<2, 0, 3>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{};
            }
        }
        using InputEncoding = decltype(make_input_encoding());

        static constexpr auto make_output_encoding()
        {
            if constexpr(NumBitType >= 8)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<LaneGroupSize>, sequence<SubtileMinorDimension>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<2>,
                    sequence<0>>{};
            }
            else
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<16>, sequence<2, NumSubtilesMajor, 16>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2, 2>,
                    sequence<1, 2>>{};
            }
        }
        using OutputEncoding = decltype(make_output_encoding());
    };

    static constexpr index_t PackedSize      = numeric_traits<remove_cvref_t<DataType>>::PackedSize;
    static constexpr index_t NumBitsDataType = (sizeof(DataType) * 8) / PackedSize;

    // Select based on data size
    template <index_t LaneGroupSize>
    using QuadInputEncoding = typename Quad<LaneGroupSize, NumBitsDataType>::InputEncoding;

    template <index_t LaneGroupSize>
    using QuadOutputEncoding = typename Quad<LaneGroupSize, NumBitsDataType>::OutputEncoding;
#endif
    // Number of elements a single transpose instruction loads per lane along the minor
    // (terminal K) dimension. Architecture/data-type specific (e.g. gfx1250 uses 128-bit
    // ds_read_tr for 16-bit types -> 8, vs 64-bit on gfx950 -> 4). Independent of
    // LaneGroupSize, so any valid value (16) selects the right Quad specialization.
    static constexpr index_t SubtileMinorDimension =
        Quad<16, NumBitsDataType>::SubtileMinorDimension;

    // Always swap last two dimensions
    static constexpr auto transpose_dims = sequence<1, 0>{};

    // Programmable: Element grouping function
    static constexpr auto group_func = [](auto idx) {
        return idx; // Identity mapping
    };

    template <typename InDstrEncode, bool ReverseDirection, index_t LaneGroupSize>
    struct ValidationTraitsImpl
    {
        using QuadEncoding             = std::conditional_t<ReverseDirection,
                                                            QuadOutputEncoding<LaneGroupSize>,
                                                            QuadInputEncoding<LaneGroupSize>>;
        static constexpr auto I0       = number<0>{};
        static constexpr auto I1       = number<1>{};
        static constexpr auto input_hs = InDstrEncode::hs_lengthss_;
        static constexpr auto quad_hs  = QuadEncoding::hs_lengthss_;
        // 1. Must be 2D tensor
        static constexpr bool dims_valid = (InDstrEncode::NDimX == 2);
        // 2. Quad pattern must be suffix of input pattern
        static constexpr bool suffix_valid_dim0 =
            util::is_sequence_suffix_v<decltype(quad_hs[I0]), decltype(input_hs[I0])>;
        static constexpr bool suffix_valid_dim1 =
            util::is_sequence_suffix_v<decltype(quad_hs[I1]), decltype(input_hs[I1])>;

        // 3. PS->RHS mapping constraints
        static constexpr auto input_ps_major = InDstrEncode::ps_to_rhss_major_;
        static constexpr auto input_ps_minor = InDstrEncode::ps_to_rhss_minor_;

        static constexpr auto quad_ps_major0 = QuadEncoding::ps_to_rhss_major_[I0];
        static constexpr auto quad_ps_minor0 = QuadEncoding::ps_to_rhss_minor_[I0];

        static constexpr auto input_ps_major_last =
            input_ps_major[number<input_ps_major.size() - 1>{}];
        static constexpr auto input_ps_minor_last =
            input_ps_minor[number<input_ps_minor.size() - 1>{}];

        using psys_offset = ck_tile::sequence<input_hs[I0].size() - quad_hs[I0].size(),
                                              input_hs[I1].size() - quad_hs[I1].size()>;
        static constexpr auto shifted_quad_ps_minor0 = generate_sequence_v2(
            [](auto i) {
                return number<quad_ps_minor0[i] + psys_offset{}[quad_ps_major0[i] - 1]>{};
            },
            number<quad_ps_minor0.size()>{});

        static constexpr bool ps_mapping_valid =
            util::is_sequence_suffix_v<decltype(quad_ps_major0), decltype(input_ps_major_last)> &&
            util::is_sequence_suffix_v<decltype(shifted_quad_ps_minor0),
                                       decltype(input_ps_minor_last)>;

        // 4. YS->RHS mapping constraints
        static constexpr auto input_ys_major = InDstrEncode::ys_to_rhs_major_;
        static constexpr auto input_ys_minor = InDstrEncode::ys_to_rhs_minor_;
        static constexpr auto quad_ys_major  = QuadEncoding::ys_to_rhs_major_;
        static constexpr auto quad_ys_minor  = QuadEncoding::ys_to_rhs_minor_;

        static_assert(quad_ys_major.back() == 2 && quad_ys_minor.back() == quad_hs[I1].size() - 1,
                      "YS->RHS mapping must be the last dimension");
        static constexpr bool ys_mapping_valid =
            (input_ys_major.back() == 2) && (input_ys_minor.back() == input_hs[I1].size() - 1);

        static constexpr bool value = dims_valid && suffix_valid_dim0 && suffix_valid_dim1 &&
                                      ps_mapping_valid && ys_mapping_valid;
    };

    template <typename InDstrEncode, bool ReverseDirection = false>
    struct ValidationTraits
    {
        static constexpr bool value =
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 64>::value ||
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 32>::value ||
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 16>::value;
        static constexpr index_t LaneGroupSize =
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 64>::value   ? 64
            : ValidationTraitsImpl<InDstrEncode, ReverseDirection, 32>::value ? 32
            : ValidationTraitsImpl<InDstrEncode, ReverseDirection, 16>::value ? 16
                                                                              : 0;
    };
};

namespace detail {

// Normalize a distribution encoding so that the last K sub-dimension equals SubtileMinorDim.
// ds_read_tr loads SubtileMinorDim elements per instruction; this width is architecture- and
// data-type-specific (e.g. gfx1250 uses a 128-bit instruction for 16-bit types, gfx950 uses
// 64-bit), so it is sourced from the transpose Policy rather than hardcoded.
// When the last K sub-dim is a multiple of SubtileMinorDim, this struct splits it into
// sequence<..., factor, SubtileMinorDim> and adds a corresponding Y dimension.
// This is a no-op when the encoding is already compatible.
template <typename DstrEncode, typename DataType, typename Policy = DefaultTranspose<DataType>>
struct NormalizeEncodingForTranspose
{
    static_assert(DstrEncode::NDimX == 2,
                  "NormalizeEncodingForTranspose only supports 2D distributions");
    static_assert(DstrEncode::NDimY > 0,
                  "NormalizeEncodingForTranspose requires at least one Y dimension");

    static constexpr index_t NumBits =
        sizeof(DataType) * 8 / numeric_traits<remove_cvref_t<DataType>>::PackedSize;
    static constexpr index_t SubtileMinorDim = Policy::SubtileMinorDimension;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    static constexpr auto k_dim = DstrEncode::hs_lengthss_[I1];
    static_assert(k_dim.size() > 0, "NormalizeEncodingForTranspose requires a non-empty K dim");
    static constexpr index_t last_k = k_dim[number<k_dim.size() - 1>{}];

    static_assert(last_k <= SubtileMinorDim || last_k % SubtileMinorDim == 0,
                  "terminal K dim must be divisible by the transpose subtile size");

    static constexpr bool needs_split =
        (last_k > SubtileMinorDim) && (last_k % SubtileMinorDim == 0);
    static constexpr index_t split_factor = needs_split ? (last_k / SubtileMinorDim) : 1;

    static constexpr bool ps_maps_terminal_k =
        rhs_mapping_contains<typename DstrEncode::Ps2RHssMajor,
                             typename DstrEncode::Ps2RHssMinor,
                             2,
                             k_dim.size() - 1>::value;
    static_assert(!needs_split || !ps_maps_terminal_k,
                  "cannot split terminal K while a P mapping points at it");

    static constexpr bool ys_last_maps_terminal_k =
        DstrEncode::ys_to_rhs_major_.back() == 2 &&
        DstrEncode::ys_to_rhs_minor_.back() == k_dim.size() - 1;
    static_assert(!needs_split || ys_last_maps_terminal_k,
                  "terminal K split requires the last Y mapping to point at terminal K");

    static constexpr auto new_k_dim = []() {
        if constexpr(needs_split)
            return k_dim.pop_back()
                .push_back(number<split_factor>{})
                .push_back(number<SubtileMinorDim>{});
        else
            return k_dim;
    }();

    static constexpr auto new_hs = generate_tuple(
        [](auto i) {
            if constexpr(i == 0)
                return DstrEncode::hs_lengthss_[I0];
            else
                return new_k_dim;
        },
        number<2>{});

    static constexpr auto new_ys_major = []() {
        if constexpr(needs_split)
            return DstrEncode::ys_to_rhs_major_.push_back(number<2>{});
        else
            return DstrEncode::ys_to_rhs_major_;
    }();

    static constexpr auto new_ys_minor = []() {
        if constexpr(needs_split)
            return DstrEncode::ys_to_rhs_minor_.push_back(
                number<DstrEncode::ys_to_rhs_minor_[number<DstrEncode::NDimY - 1>{}] + 1>{});
        else
            return DstrEncode::ys_to_rhs_minor_;
    }();

    using type = tile_distribution_encoding<typename DstrEncode::RsLengths,
                                            remove_cvref_t<decltype(new_hs)>,
                                            typename DstrEncode::Ps2RHssMajor,
                                            typename DstrEncode::Ps2RHssMinor,
                                            remove_cvref_t<decltype(new_ys_major)>,
                                            remove_cvref_t<decltype(new_ys_minor)>>;
};

} // namespace detail

template <typename TileDistribution_, typename DataType_, typename Policy>
struct TransposeTileDistrChecker
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

    using Validator = typename Policy::template ValidationTraits<InDstrEncode>;

    static constexpr bool distr_encoding_valid = Validator::value;
};

// this is used to generate the transposed output tile distribution encoding
// based on the input tile distribution encoding
template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy       = DefaultTranspose<DataType_>,
          bool ReverseDirection = false>
struct TransposeTileDistributionTraits
{
    using RawDstrEncode = remove_cvref_t<TileDistributionEncoding_>;
    // Normalize only for ReverseDirection=true (InputTileDistributionTraits).
    // The original block encoding's last K sub-dim may exceed SubtileMinorDim and need splitting.
    // For ReverseDirection=false (OutputTileDistributionTraits), the encoding is already
    // transposed and has the correct structure.
    using InDstrEncode = std::conditional_t<
        ReverseDirection,
        typename detail::NormalizeEncodingForTranspose<RawDstrEncode, DataType_, Policy>::type,
        RawDstrEncode>;
    static constexpr auto input_hs_lengthss = InDstrEncode::hs_lengthss_;
    static constexpr index_t LaneGroupSize =
        Policy::template ValidationTraits<InDstrEncode, ReverseDirection>::LaneGroupSize;
    static_assert(Policy::template ValidationTraits<InDstrEncode, ReverseDirection>::value,
                  "The input tile distribution encoding is not valid for transpose!");

    using QuadInputEncoding  = std::conditional_t< //
        ReverseDirection,
        typename Policy::template QuadOutputEncoding<LaneGroupSize>,
        typename Policy::template QuadInputEncoding<LaneGroupSize>>;
    using QuadOutputEncoding = std::conditional_t< //
        ReverseDirection,
        typename Policy::template QuadInputEncoding<LaneGroupSize>,
        typename Policy::template QuadOutputEncoding<LaneGroupSize>>;

    static constexpr auto quad_input_hs_lengthss  = QuadInputEncoding::hs_lengthss_;
    static constexpr auto quad_output_hs_lengthss = QuadOutputEncoding::hs_lengthss_;

    static constexpr auto input_ps_to_rhss_major = InDstrEncode::ps_to_rhss_major_;
    static constexpr auto input_ps_to_rhss_minor = InDstrEncode::ps_to_rhss_minor_;
    static constexpr auto input_ys_to_rhs_major  = InDstrEncode::ys_to_rhs_major_;
    static constexpr auto input_ys_to_rhs_minor  = InDstrEncode::ys_to_rhs_minor_;

    static constexpr auto I0                            = number<0>{};
    static constexpr auto quad_input_ps_to_rhss_major0  = QuadInputEncoding::ps_to_rhss_major_[I0];
    static constexpr auto quad_input_ps_to_rhss_minor0  = QuadInputEncoding::ps_to_rhss_minor_[I0];
    static constexpr auto quad_output_ps_to_rhss_major0 = QuadOutputEncoding::ps_to_rhss_major_[I0];
    static constexpr auto quad_output_ps_to_rhss_minor0 = QuadOutputEncoding::ps_to_rhss_minor_[I0];
    static constexpr auto quad_output_ys_to_rhs_major   = QuadOutputEncoding::ys_to_rhs_major_;
    static constexpr auto quad_output_ys_to_rhs_minor   = QuadOutputEncoding::ys_to_rhs_minor_;

    static constexpr index_t dim0 = Policy::transpose_dims[0];
    static constexpr index_t dim1 = Policy::transpose_dims[1];

    static constexpr auto swap_one_and_two = [](const index_t idx) {
        return (idx == 1) ? 2 : (idx == 2) ? 1 : idx;
    };

    // for transpose load
    // remove the quad_input_hs_lengthss from the input_hs_lengthss for each dimension and reverse
    // dims and append the quad_output_hs_lengthss to the end of each dimension
    static constexpr auto outer_hs_lengthss = generate_tuple(
        [](auto i) {
            constexpr auto input_i   = input_hs_lengthss[i];
            constexpr auto outer_len = input_i.size() - quad_input_hs_lengthss[i].size();
            return typename sequence_split<decltype(input_i), outer_len>::left_type{};
        },
        number<InDstrEncode::NDimX>{});
    static constexpr auto reversed_outer_hs_lengthss = tuple_reverse(outer_hs_lengthss);
    static constexpr auto dst_out_hs_lengthss        = generate_tuple(
        [](auto i) {
            auto outer_i = reversed_outer_hs_lengthss[i];
            // append the reversed quad output hs lengths to the outer hs lengths
            return outer_i.push_back(quad_output_hs_lengthss[i]);
        },
        number<InDstrEncode::NDimX>{});

    // for PS->RHS mapping(both major and minor), we need to modify the last element (which is for
    // thread distr) of the major sequence
    static constexpr auto dst_ps_to_rhss_major = generate_tuple(
        // for major because of dst_out_hs_lengthss is reversed, this index also need to be reversed
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_major.size() - 1)
            {
                constexpr auto current_size             = input_ps_to_rhss_major[i].size();
                constexpr auto reduce_size              = quad_input_ps_to_rhss_major0.size();
                constexpr auto quad_out                 = quad_output_ps_to_rhss_major0;
                constexpr auto reduced_ps_to_rhss_major = input_ps_to_rhss_major[i].extract(
                    typename arithmetic_sequence_gen<0, current_size - reduce_size, 1>::type{});
                return reduced_ps_to_rhss_major.transform(swap_one_and_two).push_back(quad_out);
            }
            else
            {
                // For all other sequences (i.e. warp), keep them unchanged
                return input_ps_to_rhss_major[i].transform(swap_one_and_two);
            }
        },
        number<input_ps_to_rhss_major.size()>{});

    static constexpr auto quad_idx_offset =
        transform_tuples([](auto x) { return number<x.size()>{}; }, reversed_outer_hs_lengthss);

    // minus 1 because RsLength is not counted
    static constexpr auto quad_output_ps_minor_offset = to_sequence(generate_tuple_for(
        [](auto x) { return quad_idx_offset[number<x - 1>{}]; }, quad_output_ps_to_rhss_major0));
    static constexpr auto quad_output_ys_minor_offset = to_sequence(generate_tuple_for(
        [](auto x) { return quad_idx_offset[number<x - 1>{}]; }, quad_output_ys_to_rhs_major));

    static constexpr auto dst_ps_to_rhss_minor = generate_tuple(
        [](auto i) {
            constexpr auto input_i = input_ps_to_rhss_minor[i];
            if constexpr(i == input_ps_to_rhss_minor.size() - 1)
            {
                constexpr auto outer_len = input_i.size() - quad_input_ps_to_rhss_minor0.size();
                constexpr auto outer_ps =
                    typename sequence_split<decltype(input_i), outer_len>::left_type{};

                return outer_ps.push_back(quad_output_ps_minor_offset +
                                          quad_output_ps_to_rhss_minor0);
            }
            else
            {
                // For all other sequences, keep them unchanged
                return input_i;
            }
        },
        number<input_ps_to_rhss_minor.size()>{});

    static constexpr auto outer_input_ys_to_rhs_major = input_ys_to_rhs_major.extract(
        typename arithmetic_sequence_gen<0,
                                         input_ys_to_rhs_major.size() -
                                             QuadInputEncoding::ys_to_rhs_major_.size(),
                                         1>::type{});

    // for major because of dst_out_hs_lengthss is reversed, this index also need to be reversed
    static constexpr auto dst_ys_to_rhs_major =
        outer_input_ys_to_rhs_major.transform(swap_one_and_two)
            .push_back(QuadOutputEncoding::ys_to_rhs_major_);

    static constexpr auto outer_input_ys_to_rhs_minor = input_ys_to_rhs_minor.extract(
        typename arithmetic_sequence_gen<0,
                                         input_ys_to_rhs_minor.size() -
                                             QuadInputEncoding::ys_to_rhs_minor_.size(),
                                         1>::type{});

    static constexpr auto dst_ys_to_rhs_minor = outer_input_ys_to_rhs_minor.push_back(
        quad_output_ys_minor_offset + QuadOutputEncoding::ys_to_rhs_minor_);

    using TransposedDstrEncode =
        tile_distribution_encoding<typename InDstrEncode::RsLengths,
                                   remove_cvref_t<decltype(dst_out_hs_lengthss)>,
                                   remove_cvref_t<decltype(dst_ps_to_rhss_major)>,
                                   remove_cvref_t<decltype(dst_ps_to_rhss_minor)>,
                                   remove_cvref_t<decltype(dst_ys_to_rhs_major)>,
                                   remove_cvref_t<decltype(dst_ys_to_rhs_minor)>>;
};

template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy = DefaultTranspose<DataType_>>
using OutputTileDistributionTraits =
    TransposeTileDistributionTraits<TileDistributionEncoding_, DataType_, Policy, false>;
template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy = DefaultTranspose<DataType_>>
using InputTileDistributionTraits =
    TransposeTileDistributionTraits<TileDistributionEncoding_, DataType_, Policy, true>;

template <typename InnerEncode,
          index_t kLeadIterPerWarp,
          index_t kSecondIterPerWarp,
          index_t kLeadNumWarps,
          index_t kSecondNumWarps>
CK_TILE_HOST_DEVICE constexpr auto InputTileDistributionEncoding()
{
    constexpr auto block_outer_dst_encoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<kSecondIterPerWarp, kSecondNumWarps>,
                                         sequence<kLeadIterPerWarp, kLeadNumWarps>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<2, 1>,
                                   sequence<0, 0>>{};
    constexpr auto blk_distr_encode =
        detail::make_embed_tile_distribution_encoding(block_outer_dst_encoding, InnerEncode{});

    return blk_distr_encode;
}

/**
 * @brief transpose loads tile from a tensor and returns the resulting tensor with a new
 * (transposed) tile distribution. use SFINAE to ensure the tile distribution encoding is valid.
 *
 * This function is intended for use with statically distributed tensor tiles, where the input
 * and output tile distributions differ due to the transpose operation. It ensures that the
 * element space size and vector length remain consistent between the input and output
 * distributions.
 *
 * @tparam DistributedTensor_     The type of the tensor containing the transposed tile data.
 * @tparam BottomTensorView_      The type of the bottom tensor view.
 * @tparam WindowLengths_         The type representing the window lengths.
 * @tparam TileDistribution_      The type representing the tile distribution.
 * @tparam NumCoord               The number of coordinates (dimensions).
 * @tparam Policy                 The transpose policy to use (defaults to DefaultTranspose).
 * the last is SFINAE to ensure the tile distribution encoding is valid.
 *
 * @param out_tensor              A statically distributed tensor containing the transposed tile
 * data.
 * @param tile_window             The tile window with static distribution to load and transpose.
 * @param offset                  The offset (in elements) added to the base address before
 * indexing.
 *
 * @note
 * - The function uses compile-time checks to ensure the input and output tile distributions
 *   are compatible in terms of element space size and vector length.
 * - The transpose operation is performed according to the specified Policy.
 */
template <
    typename DistributedTensor_,
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord,
    typename Policy             = DefaultTranspose<typename BottomTensorView_::DataType>,
    index_t i_access_unsupport_ = -1,
    bool oob_conditional_check  = true,
    bool static_move_ys         = false,
    typename                    = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                             typename BottomTensorView_::DataType,
                                                                             Policy>::distr_encoding_valid,
                                                   Policy>>
CK_TILE_DEVICE void load_tile_transpose_with_offset(
    DistributedTensor_& out_tensor,
    const tile_window_with_static_distribution<BottomTensorView_,
                                               WindowLengths_,
                                               TileDistribution_,
                                               NumCoord>& __restrict__ tile_window,
    index_t offset,
    number<i_access_unsupport_>          = {},
    bool_constant<oob_conditional_check> = {},
    bool_constant<static_move_ys>        = {})
{
    auto trans_tensor =
        tile_window.template load_transpose_with_offset<Policy,
                                                        number<i_access_unsupport_>{},
                                                        bool_constant<oob_conditional_check>{},
                                                        bool_constant<static_move_ys>{}>(offset);
    constexpr auto input_distr  = TileDistribution_{};
    constexpr auto output_distr = typename DistributedTensor_::StaticTileDistribution{};

    // Check that out_tensor's distribution matches the expected transposed-load distribution.
    // Exact equality is accepted, as is the narrow case where the terminal Y dimension is
    // represented as either one dimension or two adjacent split dimensions with identical flat
    // per-thread access order.
    using OutTileDstrEncode = typename OutputTileDistributionTraits<
        typename TileDistribution_::DstrEncode,
        typename BottomTensorView_::DataType>::TransposedDstrEncode;
    using ExpectedDstr = decltype(make_static_tile_distribution(OutTileDstrEncode{}));
    using ActualDstr   = remove_cvref_t<decltype(output_distr)>;
    static_assert(detail::is_transpose_output_compatible_v<ExpectedDstr,
                                                           ActualDstr,
                                                           typename BottomTensorView_::DataType>,
                  "out_tensor distribution is not compatible with transpose load output");

    // Check that the datatype of out_tensor matches that of the bottom tensor view.
    static_assert(std::is_same_v<typename DistributedTensor_::DataType,
                                 typename BottomTensorView_::DataType>);

    constexpr auto y_in_desc  = input_distr.get_ys_to_d_descriptor();
    constexpr auto y_out_desc = output_distr.get_ys_to_d_descriptor();

    constexpr index_t NDimYIn = input_distr.get_num_of_dimension_y();

    constexpr auto y_in_lengths = to_sequence(y_in_desc.get_lengths());

    constexpr auto y_in_element_space_size  = y_in_desc.get_element_space_size();
    constexpr auto y_out_element_space_size = y_out_desc.get_element_space_size();
    static_assert(y_in_element_space_size == y_out_element_space_size,
                  "the element space size is not the same!");

    constexpr index_t vecLoadSize = y_in_lengths[NDimYIn - 1];
    constexpr index_t num_of_access =
        reduce_on_sequence(y_in_lengths, multiplies<>{}, number<1>{}) / vecLoadSize;

    constexpr index_t packed_size =
        numeric_traits<remove_cvref_t<typename BottomTensorView_::DataType>>::PackedSize;
    static_assert(vecLoadSize % packed_size == 0, "vecLoadSize must be divisible by packed_size");
    using DataVec = array<typename BottomTensorView_::DataType, vecLoadSize / packed_size>;
    static_for<0, num_of_access, 1>{}([&](auto iAccess) {
        out_tensor.get_thread_buffer().template set_as<DataVec>(
            number<iAccess>{},
            trans_tensor.get_thread_buffer().template get_as<DataVec>(number<iAccess>{}));
    });
}

/**
 * @brief transpose loads tile from a tensor and returns the resulting tensor with a new
 * (transposed) tile distribution. use SFINAE to ensure the tile distribution encoding is valid.
 *
 * This function is intended for use with statically distributed tensor tiles, where the input
 * and output tile distributions differ due to the transpose operation. It ensures that the
 * element space size and vector length remain consistent between the input and output
 * distributions.
 *
 * @tparam DistributedTensor_     The type of the tensor containing the transposed tile data.
 * @tparam BottomTensorView_      The type of the bottom tensor view.
 * @tparam WindowLengths_         The type representing the window lengths.
 * @tparam TileDistribution_      The type representing the tile distribution.
 * @tparam NumCoord               The number of coordinates (dimensions).
 * @tparam Policy                 The transpose policy to use (defaults to DefaultTranspose).
 * the last is SFINAE to ensure the tile distribution encoding is valid.
 *
 * @param out_tensor              A statically distributed tensor containing the transposed tile
 * data.
 * @param tile_window             The tile window with static distribution to load and transpose.
 * indexing.
 *
 * @note
 * - The function uses compile-time checks to ensure the input and output tile distributions
 *   are compatible in terms of element space size and vector length.
 * - The transpose operation is performed according to the specified Policy.
 */
template <
    typename DistributedTensor_,
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord,
    typename Policy = DefaultTranspose<typename BottomTensorView_::DataType>,
    typename        = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                 typename BottomTensorView_::DataType,
                                                                 Policy>::distr_encoding_valid,
                                       Policy>>
CK_TILE_DEVICE void
load_tile_transpose(DistributedTensor_& out_tensor,
                    const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& __restrict__ tile_window)
{
    load_tile_transpose_with_offset(out_tensor, tile_window, 0);
}

template <
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord,
    typename Policy = DefaultTranspose<typename BottomTensorView_::DataType>,
    typename        = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                 typename BottomTensorView_::DataType,
                                                                 Policy>::distr_encoding_valid,
                                       Policy>>
CK_TILE_DEVICE auto
load_tile_transpose(const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& __restrict__ tile_window)
{
    using OutTileDstrEncode = typename OutputTileDistributionTraits<
        typename TileDistribution_::DstrEncode,
        typename BottomTensorView_::DataType>::TransposedDstrEncode;
    auto out_tensor = make_static_distributed_tensor<typename BottomTensorView_::DataType>(
        make_static_tile_distribution(OutTileDstrEncode{}));

    load_tile_transpose_with_offset(out_tensor, tile_window, 0);

    return out_tensor;
}

/**
 * @brief Mixed-precision transpose load: converts input data type to output data type while
 * transposing.
 *
 * This function enables transposing from one data type (e.g., fp8) to another (e.g., fp16) in a
 * single operation. The input tile distribution encoding must be valid for the input data type,
 * and the output distribution will be generated based on the output data type.
 *
 * @tparam DistributedTensor_     The output tensor type with desired output data type.
 * @tparam BottomTensorView_      The input tensor view (may have different data type than output).
 * @tparam WindowLengths_         The type representing the window lengths.
 * @tparam TileDistribution_      The type representing the tile distribution for input.
 * @tparam NumCoord_              The number of coordinates (dimensions).
 * @tparam Policy                 The transpose policy (should validate against input type).
 *
 * @note
 * - Input and output must have compatible element space sizes (total byte count per Y-space).
 * - Type conversion is performed element-by-element during the copy.
 * - The validation uses the input data type for quad pattern checking.
 * - The output distribution is generated based on the output data type.
 */
template <
    typename DistributedTensor_,
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord_,
    index_t UnaryOpSize_,
    typename PassThroughPack_,
    typename Policy = DefaultTranspose<typename BottomTensorView_::DataType>,
    typename        = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                 typename BottomTensorView_::DataType,
                                                                 Policy>::distr_encoding_valid,
                                       Policy>>
CK_TILE_DEVICE void load_tile_transpose_convert_with_offset(
    DistributedTensor_& out_tensor,
    const tile_window_with_static_distribution<BottomTensorView_,
                                               WindowLengths_,
                                               TileDistribution_,
                                               NumCoord_>& __restrict__ tile_window,
    const index_t offset,
    number<UnaryOpSize_>            = {},
    PassThroughPack_ elementwise_op = {})
{
    using SrcDataType = typename BottomTensorView_::DataType;
    using DstDataType = typename DistributedTensor_::DataType;

    auto trans_tensor           = tile_window.template load_transpose_with_offset<Policy>(offset);
    constexpr auto input_distr  = TileDistribution_{};
    constexpr auto output_distr = typename DistributedTensor_::StaticTileDistribution{};

    constexpr auto y_in_desc  = input_distr.get_ys_to_d_descriptor();
    constexpr auto y_out_desc = output_distr.get_ys_to_d_descriptor();

    constexpr auto y_in_lengths  = to_sequence(y_in_desc.get_lengths());
    constexpr auto y_out_lengths = to_sequence(y_out_desc.get_lengths());

    constexpr auto y_in_element_space_size  = y_in_desc.get_element_space_size();
    constexpr auto y_out_element_space_size = y_out_desc.get_element_space_size();

    // For mixed precision: input and output element space sizes must be the same.
    static_assert(
        y_in_element_space_size == y_out_element_space_size,
        "For mixed precision transpose, input and output element space sizes must match!");

    // Ensure total element counts are consistent and divisible by the input vector length.
    constexpr index_t total_elems_in =
        reduce_on_sequence(y_in_lengths, multiplies<>{}, number<1>{});
    constexpr index_t total_elems_out =
        reduce_on_sequence(y_out_lengths, multiplies<>{}, number<1>{});
    static_assert(total_elems_in == total_elems_out,
                  "For mixed precision transpose, input/output element counts must match!");
    static_assert(total_elems_in % number<UnaryOpSize_>{} == 0,
                  "Input vector length must evenly divide total elements.");

    constexpr index_t num_of_access = total_elems_in / number<UnaryOpSize_>{};

    // Read as input type, convert to output type
    using SrcDataVec = ext_vector_t<SrcDataType, number<UnaryOpSize_>{}>;
    using DstDataVec = ext_vector_t<DstDataType, number<UnaryOpSize_>{}>;

    static_for<0, num_of_access, 1>{}([&](auto i) {
        elementwise_op(out_tensor.get_thread_buffer().template get_as<DstDataVec>()(i),
                       trans_tensor.get_thread_buffer().template get_as<SrcDataVec>()[i]);
    });
}

/**
 * @brief Mixed-precision transpose load with zero offset.
 *
 * Convenience wrapper for load_tile_transpose_convert_with_offset with offset=0.
 */
template <
    typename DistributedTensor_,
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord_,
    index_t UnaryOpSize_,
    typename PassThroughPack_,
    typename Policy = DefaultTranspose<typename BottomTensorView_::DataType>,
    typename        = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                 typename BottomTensorView_::DataType,
                                                                 Policy>::distr_encoding_valid,
                                       Policy>>
CK_TILE_DEVICE void load_tile_transpose_convert(
    DistributedTensor_& out_tensor,
    const tile_window_with_static_distribution<BottomTensorView_,
                                               WindowLengths_,
                                               TileDistribution_,
                                               NumCoord_>& __restrict__ tile_window,
    number<UnaryOpSize_>            = {},
    PassThroughPack_ elementwise_op = {})
{
    load_tile_transpose_convert_with_offset(
        out_tensor, tile_window, 0, number<UnaryOpSize_>{}, elementwise_op);
}

} // namespace ck_tile
