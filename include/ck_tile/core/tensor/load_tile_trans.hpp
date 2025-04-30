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

template <typename T, typename = void>
struct valid_quad_tile_dstr_encode_for_transpose;

template <typename T>
struct valid_quad_tile_dstr_encode_for_transpose<T, std::enable_if_t<sizeof(T) == 2>>
{
    using TileDistrEncode = tile_distribution_encoding<sequence<>,
                                                       tuple<sequence<1, 4>, sequence<4, 4>>,
                                                       tuple<sequence<1, 2>>,
                                                       tuple<sequence<1, 0>>,
                                                       sequence<1, 2>,
                                                       sequence<0, 1>>;
};

template <typename TileDistribution_, typename DataType_>
struct tile_distribution_for_transpose_helper
{
    using DstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

    using ValidQuadDstrEncode =
        typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TileDistrEncode;

    // every dimx in ValidQuadDstrEncode should be the suffix of DstrEncode
    static constexpr auto valid_hs_lengthss  = ValidQuadDstrEncode::hs_lengthss_;
    static constexpr auto actual_hs_lengthss = DstrEncode::hs_lengthss_;
    static constexpr bool hs_lengthss_size_valid =
        (DstrEncode::NDimX == ValidQuadDstrEncode::NDimX);

    // make sure NDimX == 2; only support 2D transpose
    static constexpr bool hs_lengthss_size_dim_valid = (DstrEncode::NDimX == 2);
    // Check each element using helper function
    template <index_t I = 0>
    struct check_hs_lengthss_suffixes
    {
        static constexpr bool value =
            util::is_sequence_suffix_v<decltype(valid_hs_lengthss.template get<I>()),
                                       decltype(actual_hs_lengthss.template get<I>())> &&
            check_hs_lengthss_suffixes<I + 1>::value;
    };

    template <>
    struct check_hs_lengthss_suffixes<valid_hs_lengthss.size()>
    {
        static constexpr bool value = true;
    };

    static constexpr bool all_hs_lengthss_are_suffixes = check_hs_lengthss_suffixes<>::value;

    // static constexpr auto hs_lengthss      = DstrEncode::hs_lengthss_;
    static constexpr auto ps_to_rhss_major = DstrEncode::ps_to_rhss_major_;
    static constexpr auto ps_to_rhss_minor = DstrEncode::ps_to_rhss_minor_;
    static constexpr auto ys_to_rhs_major  = DstrEncode::ys_to_rhs_major_;
    static constexpr auto ys_to_rhs_minor  = DstrEncode::ys_to_rhs_minor_;

    // get hs_lengthss[0].size
    static constexpr index_t ndimp_outer_size = ps_to_rhss_major.size();
    // make sure ndimp >= 2
    static constexpr index_t ndimp_inner_size =
        ps_to_rhss_major[number<ndimp_outer_size - 1>{}].size();
    // make sure ps_to_rhss_major[ndimp-1] == 2
    // make sure ps_to_rhss_minor[ndimp-1] == hs_lengthss[1].size - 2;

    // the below two conditions are used to check whether encoding is based on quadrant.
    static constexpr bool ps_to_rhss_index0_valid =
        (ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] == 2) &&
        (ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] ==
         actual_hs_lengthss[number<1>{}].size() - 2);

    // make sure ps_to_rhss_major[ndimp-2] == 1
    // make sure ps_to_rhss_minor[ndimp-2] == hs_lengthss[0].size - 1;
    static constexpr bool ps_to_rhss_index1_valid =
        (ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] == 1) &&
        (ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] ==
         actual_hs_lengthss[number<0>{}].size() - 1);
    // get ys_to_rhs_major.size
    // make sure ys_to_rhs_major[ndim_y-1] == 2
    // make sure ys_to_rhs_minor[ndim_y-1] == hs_lengthss[0].size - 1;
    static constexpr index_t ndimy_size = ys_to_rhs_major.size();
    static constexpr bool ys_to_rhs_major_valid =
        ((ys_to_rhs_major[ndimy_size - 1] == 2) &&
         (ys_to_rhs_minor[ndimy_size - 1] == actual_hs_lengthss[number<1>{}].size() - 1)) &&
        ((ys_to_rhs_major[ndimy_size - 2] == 1) &&
         (ys_to_rhs_minor[ndimy_size - 2] == actual_hs_lengthss[number<0>{}].size() - 2));

    static constexpr bool distr_encoding_valid =
        hs_lengthss_size_valid && hs_lengthss_size_dim_valid && all_hs_lengthss_are_suffixes &&
        ps_to_rhss_index0_valid && ps_to_rhss_index1_valid && ys_to_rhs_major_valid;

    // if ndimy >= 2, others is iteration number per thread
    // get ys_to_rhs_major[ndim_y-2] and ys_to_rhs_minor[ndim_y-2]
    // static constexpr index_t iteration_number = ys_to_rhs_major[ndimy_size]
    // get other dims in ps_to_rhss_major from [0:ndimp-2)
    // static constexpr index_t iteration_number =
    //     ndimy_size == 1 ? 1
    //                     : reduce_on_sequence(
    //                           ys_to_rhs_major.extract(
    //                               typename arithmetic_sequence_gen<0, ndimy_size - 1,
    //                               1>::type{}),
    //                           multiplies{},
    //                           number<1>{});
};

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          typename                   = std::enable_if_t<tile_distribution_for_transpose_helper<
              TileDistribution_,
              typename BottomTensorView_::DataType>::distr_encoding_valid>>
CK_TILE_DEVICE auto
load_tile_transpose(const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window,
                    number<i_access>                     = {},
                    bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_transpose(number<i_access>{}, bool_constant<oob_conditional_check>{});
    // using transpose_tile_distribution_helper =
    //     tile_distribution_for_transpose_helper<typename TileDistribution_::DstrEncode,
    //                                            typename BottomTensorView_::DataType>;

    // constexpr index_t iteration_num = transpose_tile_distribution_helper::iteration_number;

    // static_for<0, iteration_num, 1>{}([&](auto i) {

    // });
}

} // namespace ck_tile
