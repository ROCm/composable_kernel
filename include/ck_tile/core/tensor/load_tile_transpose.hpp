// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

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

// Default policy: Retains original 2D transpose behavior
template <typename DataType>
struct DefaultTranspose
{
    struct Quad16
    {
        using InputEncoding = tile_distribution_encoding<sequence<>,
                                                         tuple<sequence<4>, sequence<4, 4>>,
                                                         tuple<sequence<1, 2>>,
                                                         tuple<sequence<0, 0>>,
                                                         sequence<2>,
                                                         sequence<1>>;

        using OutputEncoding = tile_distribution_encoding<sequence<>,
                                                          tuple<sequence<16>, sequence<4>>,
                                                          tuple<sequence<1>>,
                                                          tuple<sequence<0>>,
                                                          sequence<2>,
                                                          sequence<0>>;
    };

    struct Quad8
    {
        using InputEncoding = tile_distribution_encoding<sequence<>,
                                                         tuple<sequence<8>, sequence<2, 8>>,
                                                         tuple<sequence<1, 2>>,
                                                         tuple<sequence<0, 0>>,
                                                         sequence<2>,
                                                         sequence<1>>;

        using OutputEncoding = tile_distribution_encoding<sequence<>,
                                                          tuple<sequence<16>, sequence<8>>,
                                                          tuple<sequence<1>>,
                                                          tuple<sequence<0>>,
                                                          sequence<2>,
                                                          sequence<0>>;
    };

    // Select based on data size
    using QuadInputEncoding = std::conditional_t<sizeof(DataType) == 2,
                                                 typename Quad16::InputEncoding,
                                                 typename Quad8::InputEncoding>;

    using QuadOutputEncoding = std::conditional_t<sizeof(DataType) == 2,
                                                  typename Quad16::OutputEncoding,
                                                  typename Quad8::OutputEncoding>;

    // Always swap last two dimensions
    static constexpr auto transpose_dims = sequence<1, 0>{};

    // Programmable: Element grouping function
    static constexpr auto group_func = [](auto idx) {
        return idx; // Identity mapping
    };

    template <typename InDstrEncode>
    struct ValidationTraits
    {
        static constexpr auto input_hs_lengthss = InDstrEncode::hs_lengthss_;
        static constexpr auto quad_hs_lengthss  = QuadInputEncoding::hs_lengthss_;
        // 1. Must be 2D tensor
        static constexpr bool dims_valid = (InDstrEncode::NDimX == 2);
        // 2. Quad pattern must be suffix of input pattern
        static constexpr bool suffix_valid_dim0 =
            util::is_sequence_suffix_v<decltype(quad_hs_lengthss.template get<0>()),
                                       decltype(input_hs_lengthss.template get<0>())>;
        static constexpr bool suffix_valid_dim1 =
            util::is_sequence_suffix_v<decltype(quad_hs_lengthss.template get<1>()),
                                       decltype(input_hs_lengthss.template get<1>())>;

        // 3. PS→RHS mapping constraints
        static constexpr auto input_ps_to_rhss_major = InDstrEncode::ps_to_rhss_major_;
        static constexpr auto input_ps_to_rhss_minor = InDstrEncode::ps_to_rhss_minor_;

        static constexpr index_t ndimp_outer = input_ps_to_rhss_major.size() - 1;
        static constexpr index_t ndimp_inner =
            input_ps_to_rhss_major[number<ndimp_outer>{}].size() - 1;

        static constexpr bool ps_mapping_valid =
            (input_ps_to_rhss_major[number<ndimp_outer>{}][number<ndimp_inner>{}] == 2) &&
            (input_ps_to_rhss_minor[number<ndimp_outer>{}][number<ndimp_inner>{}] ==
             input_hs_lengthss[number<1>{}].size() - 2) &&
            (input_ps_to_rhss_major[number<ndimp_outer>{}][number<ndimp_inner - 1>{}] == 1) &&
            (input_ps_to_rhss_minor[number<ndimp_outer>{}][number<ndimp_inner - 1>{}] ==
             input_hs_lengthss[number<0>{}].size() - 1);

        // 4. YS→RHS mapping constraints
        static constexpr auto input_ys_to_rhs_major = InDstrEncode::ys_to_rhs_major_;
        static constexpr auto input_ys_to_rhs_minor = InDstrEncode::ys_to_rhs_minor_;

        static constexpr bool ys_mapping_valid =
            (input_ys_to_rhs_major.back() == 2) &&
            (input_ys_to_rhs_minor.back() == input_hs_lengthss[number<1>{}].size() - 1) &&
            (input_ys_to_rhs_major[input_ys_to_rhs_major.size() - 2] == 1) &&
            (input_ys_to_rhs_minor[input_ys_to_rhs_minor.size() - 2] ==
             input_hs_lengthss[number<0>{}].size() - 2);

        static constexpr bool value = dims_valid && suffix_valid_dim0 && suffix_valid_dim1 &&
                                      ps_mapping_valid && ys_mapping_valid;
    };
};
template <typename TileDistribution_, typename DataType_, typename Policy>
struct TransposeTileDistrChecker
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

    using Validator = typename Policy::template ValidationTraits<InDstrEncode>;

    static constexpr bool distr_encoding_valid = Validator::value;
};

// this is used to generate the transposed output tile distribution encoding
// based on the input tile distribution encoding
template <typename TileDistribution_,
          typename DataType_,
          typename Policy = DefaultTranspose<DataType_>>
struct OutputTileDistributionTraits
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;
    static constexpr auto input_hs_lengthss       = InDstrEncode::hs_lengthss_;
    static constexpr auto quad_input_hs_lengthss  = Policy::QuadInputEncoding::hs_lengthss_;
    static constexpr auto quad_output_hs_lengthss = Policy::QuadOutputEncoding::hs_lengthss_;

    static constexpr auto input_ps_to_rhss_major = InDstrEncode::ps_to_rhss_major_;
    static constexpr auto input_ps_to_rhss_minor = InDstrEncode::ps_to_rhss_minor_;
    static constexpr auto input_ys_to_rhs_major  = InDstrEncode::ys_to_rhs_major_;
    static constexpr auto input_ys_to_rhs_minor  = InDstrEncode::ys_to_rhs_minor_;

    static constexpr auto quad_ps_to_rhss_major = Policy::QuadInputEncoding::ps_to_rhss_major_;
    static constexpr auto quad_ps_to_rhss_minor = Policy::QuadInputEncoding::ps_to_rhss_minor_;

    // for transpose load
    // append the reversed quad output hs lengths to the input hs lengthss after removing
    // the quad_input_hs_lengthss
    // then reverse the whole sequence to get the dst_out_hs_lengthss
    static constexpr auto reversed_quad_output_hs_lengthss = tuple_reverse(quad_output_hs_lengthss);

    static constexpr auto full_out_hs_lengthss = generate_tuple(
        [](auto i) {
            return input_hs_lengthss[i]
                .extract(typename arithmetic_sequence_gen<0,
                                                          input_hs_lengthss[i].size() -
                                                              quad_input_hs_lengthss[i].size(),
                                                          1>::type{})
                .push_back(reversed_quad_output_hs_lengthss[i]);
        },
        number<InDstrEncode::NDimX>{});

    static constexpr auto dst_out_hs_lengthss = tuple_reverse(full_out_hs_lengthss);

    // for PS→RHS mapping(both major and minor), we need to modify the last element of the major
    // sequence
    static constexpr auto modified_ps_to_rhss_major = generate_tuple(
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_major.size() - 1)
            {
                constexpr auto current_size             = input_ps_to_rhss_major[i].size();
                constexpr auto reduce_size              = quad_ps_to_rhss_major[number<0>{}].size();
                constexpr auto reduced_ps_to_rhss_major = input_ps_to_rhss_major[i].extract(
                    typename arithmetic_sequence_gen<0, current_size - reduce_size, 1>::type{});
                return reduced_ps_to_rhss_major.push_back(number<2>{});
            }
            else
            {
                // For all other sequences, keep them unchanged
                return input_ps_to_rhss_major[i];
            }
        },
        number<input_ps_to_rhss_major.size()>{});

    static constexpr auto minor_last_index =
        full_out_hs_lengthss[number<InDstrEncode::NDimX - 1>{}].size() - 1;
    static constexpr auto major_last_index = full_out_hs_lengthss[number<0>{}].size() - 1;

    static constexpr auto dst_ps_to_rhss_minor = generate_tuple(
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_minor.size() - 1)
            {
                constexpr auto current_size             = input_ps_to_rhss_minor[i].size();
                constexpr auto reduce_size              = quad_ps_to_rhss_minor[number<0>{}].size();
                constexpr auto reduced_ps_to_rhss_minor = input_ps_to_rhss_minor[i].extract(
                    typename arithmetic_sequence_gen<0, current_size - reduce_size, 1>::type{});
                return reduced_ps_to_rhss_minor.push_back(number<minor_last_index>{});
            }
            else
            {
                // For all other sequences, keep them unchanged
                return input_ps_to_rhss_minor[i];
            }
        },
        number<input_ps_to_rhss_minor.size()>{});

    // for major because of dst_out_hs_lengthss is reversed, this index also need to be reversed
    static constexpr auto swap_one_and_two = [](const index_t idx) {
        return (idx == 1) ? 2 : (idx == 2) ? 1 : idx;
    };
    static constexpr auto dst_ps_to_rhss_major = generate_tuple(
        [](auto i) { return modified_ps_to_rhss_major[i].transform(swap_one_and_two); },
        number<modified_ps_to_rhss_major.size()>{});

    static constexpr auto modified_input_ys_to_rhs_major =
        input_ys_to_rhs_major.pop_back().push_back(number<1>{});

    static constexpr auto dst_ys_to_rhs_major = generate_sequence_v2(
        [](auto i) { return number<swap_one_and_two(modified_input_ys_to_rhs_major[i])>{}; },
        number<modified_input_ys_to_rhs_major.size()>{});

    static constexpr auto dst_ys_to_rhs_minor =
        input_ys_to_rhs_minor.pop_back().push_back(number<major_last_index>{});

    using OutDstrEncode = tile_distribution_encoding<typename InDstrEncode::RsLengths,
                                                     remove_cvref_t<decltype(dst_out_hs_lengthss)>,
                                                     remove_cvref_t<decltype(dst_ps_to_rhss_major)>,
                                                     remove_cvref_t<decltype(dst_ps_to_rhss_minor)>,
                                                     remove_cvref_t<decltype(dst_ys_to_rhs_major)>,
                                                     remove_cvref_t<decltype(dst_ys_to_rhs_minor)>>;
};

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
 * @tparam BottomTensorView_      The type of the bottom tensor view.
 * @tparam WindowLengths_         The type representing the window lengths.
 * @tparam TileDistribution_      The type representing the tile distribution.
 * @tparam NumCoord               The number of coordinates (dimensions).
 * @tparam Policy                 The transpose policy to use (defaults to DefaultTranspose).
 * the last is SFINAE to ensure the tile distribution encoding is valid.
 *
 * @param tile_window             The tile window with static distribution to load and transpose.
 *
 * @return A statically distributed tensor containing the transposed tile data.
 *
 * @note
 * - The function uses compile-time checks to ensure the input and output tile distributions
 *   are compatible in terms of element space size and vector length.
 * - The transpose operation is performed according to the specified Policy.
 */
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
                                                               NumCoord>& tile_window)
{
    using OutTileDstrEncode =
        typename OutputTileDistributionTraits<TileDistribution_,
                                              typename BottomTensorView_::DataType>::OutDstrEncode;
    auto out_tensor = make_static_distributed_tensor<typename BottomTensorView_::DataType>(
        make_static_tile_distribution(OutTileDstrEncode{}));
    auto trans_tensor           = tile_window.template load_transpose<Policy>();
    constexpr auto input_distr  = TileDistribution_{};
    constexpr auto output_distr = make_static_tile_distribution(OutTileDstrEncode{});

    constexpr auto y_in_desc  = input_distr.get_ys_to_d_descriptor();
    constexpr auto y_out_desc = output_distr.get_ys_to_d_descriptor();

    constexpr index_t NDimYIn  = input_distr.get_num_of_dimension_y();
    constexpr index_t NDimYOut = output_distr.get_num_of_dimension_y();

    constexpr auto y_in_lengths  = to_sequence(y_in_desc.get_lengths());
    constexpr auto y_out_lengths = to_sequence(y_out_desc.get_lengths());

    constexpr auto y_in_element_space_size  = y_in_desc.get_element_space_size();
    constexpr auto y_out_element_space_size = y_out_desc.get_element_space_size();
    static_assert(y_in_element_space_size == y_out_element_space_size,
                  "the element space size is not the same!");
    static_assert(y_in_lengths[NDimYIn - 1] == y_out_lengths[NDimYOut - 1],
                  "the vector length is not the same!");
    constexpr index_t vecLoadSize = y_in_lengths[NDimYIn - 1];
    constexpr index_t num_of_access =
        reduce_on_sequence(y_in_lengths, multiplies{}, number<1>{}) / vecLoadSize;

    using DataVec = array<typename BottomTensorView_::DataType, vecLoadSize>;
    static_for<0, num_of_access, 1>{}([&](auto iAccess) {
        out_tensor.get_thread_buffer().template set_as<DataVec>(
            number<iAccess>{},
            trans_tensor.get_thread_buffer().template get_as<DataVec>(number<iAccess>{}));
    });

    return out_tensor;
}

} // namespace ck_tile
