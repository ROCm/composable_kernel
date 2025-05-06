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
                                                       tuple<sequence<4>, sequence<4, 4>>,
                                                       tuple<sequence<1, 2>>,
                                                       tuple<sequence<0, 0>>,
                                                       sequence<2>,
                                                       sequence<1>>;

    using TransposedTileDistrEncode = tile_distribution_encoding<sequence<>,
                                                                 tuple<sequence<16>, sequence<4>>,
                                                                 tuple<sequence<1>>,
                                                                 tuple<sequence<0>>,
                                                                 sequence<2>,
                                                                 sequence<0>>;
};

// template <typename TileDistribution_, typename DataType_>
// struct tile_distribution_for_transpose_helper
// {
//     using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

//     static constexpr auto actual_hs_lengthss = InDstrEncode::hs_lengthss_;
//     using ValidQuadDstrEncode =
//         typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TileDistrEncode;
//     using ValidOutQuadDstrEncode =
//         typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TransposedTileDistrEncode;
//     // every dimx in ValidQuadDstrEncode should be the suffix of DstrEncode
//     static constexpr auto valid_hs_lengthss     = ValidQuadDstrEncode::hs_lengthss_;
//     static constexpr auto valid_out_hs_lengthss = ValidOutQuadDstrEncode::hs_lengthss_;
//     static constexpr auto ps_to_rhss_major      = InDstrEncode::ps_to_rhss_major_;
//     static constexpr auto ps_to_rhss_minor      = InDstrEncode::ps_to_rhss_minor_;
//     static constexpr auto ys_to_rhs_major       = InDstrEncode::ys_to_rhs_major_;
//     static constexpr auto ys_to_rhs_minor       = InDstrEncode::ys_to_rhs_minor_;

//     static constexpr bool hs_lengthss_size_valid =
//         (InDstrEncode::NDimX == ValidQuadDstrEncode::NDimX);

//     static constexpr auto full_out_hs_lengthss = generate_tuple(
//         [&](auto i) {
//             return actual_hs_lengthss[i]
//                 .extract(typename arithmetic_sequence_gen<0,
//                                                           actual_hs_lengthss[i].size() -
//                                                               valid_hs_lengthss[i].size(),
//                                                           1>::type{})
//                 .push_back(valid_out_hs_lengthss[i]);
//         },
//         number<InDstrEncode::NDimX>{});

//     static constexpr auto modified_ps_to_rhss_major = generate_tuple(
//         [&](auto i) {
//             if constexpr(i == ps_to_rhss_major.size() - 1)
//             {
//                 // For the last sequence, remove the last element
//                 return ps_to_rhss_major[i].pop_back();
//             }
//             else
//             {
//                 // For all other sequences, keep them unchanged
//                 return ps_to_rhss_major[i];
//             }
//         },
//         number<ps_to_rhss_major.size()>{});

//     static constexpr auto modified_ps_to_rhss_minor = generate_tuple(
//         [&](auto i) {
//             if constexpr(i == ps_to_rhss_minor.size() - 1)
//             {
//                 // For the last sequence, remove the last element
//                 return ps_to_rhss_minor[i].pop_back();
//             }
//             else
//             {
//                 // For all other sequences, keep them unchanged
//                 return ps_to_rhss_minor[i];
//             }
//         },
//         number<ps_to_rhss_minor.size()>{});

//     using OutDstrEncode = tile_distribution_encoding<InDstrEncode::RsLengths,
//                                                      decltype(full_out_hs_lengthss),
//                                                      decltype(modified_ps_to_rhss_major),
//                                                      decltype(modified_ps_to_rhss_minor),
//                                                      InDstrEncode::Ys2RHsMajor,
//                                                      InDstrEncode::Ys2RHsMinor>;
//     // make sure NDimX == 2; only support 2D transpose
//     static constexpr bool hs_lengthss_size_dim_valid = (InDstrEncode::NDimX == 2);
//     // Check each element using helper function
//     template <index_t I = 0>
//     struct check_hs_lengthss_suffixes
//     {
//         static constexpr bool value =
//             util::is_sequence_suffix_v<decltype(valid_hs_lengthss.template get<I>()),
//                                        decltype(actual_hs_lengthss.template get<I>())> &&
//             check_hs_lengthss_suffixes<I + 1>::value;
//     };

//     template <>
//     struct check_hs_lengthss_suffixes<valid_hs_lengthss.size()>
//     {
//         static constexpr bool value = true;
//     };

//     static constexpr bool all_hs_lengthss_are_suffixes = check_hs_lengthss_suffixes<>::value;

//     // static constexpr auto hs_lengthss      = DstrEncode::hs_lengthss_;

//     // get hs_lengthss[0].size
//     static constexpr index_t ndimp_outer_size = ps_to_rhss_major.size();
//     // make sure ndimp >= 2
//     static constexpr index_t ndimp_inner_size =
//         ps_to_rhss_major[number<ndimp_outer_size - 1>{}].size();
//     // make sure ps_to_rhss_major[ndimp-1] == 2
//     // make sure ps_to_rhss_minor[ndimp-1] == hs_lengthss[1].size - 2;

//     // the below two conditions are used to check whether encoding is based on quadrant.
//     static constexpr bool ps_to_rhss_index0_valid =
//         (ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] == 2) &&
//         (ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] ==
//          actual_hs_lengthss[number<1>{}].size() - 2);

//     // make sure ps_to_rhss_major[ndimp-2] == 1
//     // make sure ps_to_rhss_minor[ndimp-2] == hs_lengthss[0].size - 1;
//     static constexpr bool ps_to_rhss_index1_valid =
//         (ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] == 1) &&
//         (ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] ==
//          actual_hs_lengthss[number<0>{}].size() - 1);
//     // get ys_to_rhs_major.size
//     // make sure ys_to_rhs_major[ndim_y-1] == 2
//     // make sure ys_to_rhs_minor[ndim_y-1] == hs_lengthss[0].size - 1;
//     static constexpr index_t ndimy_size = ys_to_rhs_major.size();
//     static constexpr bool ys_to_rhs_major_valid =
//         ((ys_to_rhs_major[ndimy_size - 1] == 2) &&
//          (ys_to_rhs_minor[ndimy_size - 1] == actual_hs_lengthss[number<1>{}].size() - 1)) &&
//         ((ys_to_rhs_major[ndimy_size - 2] == 1) &&
//          (ys_to_rhs_minor[ndimy_size - 2] == actual_hs_lengthss[number<0>{}].size() - 2));

//     static constexpr bool distr_encoding_valid =
//         hs_lengthss_size_valid && hs_lengthss_size_dim_valid && all_hs_lengthss_are_suffixes &&
//         ps_to_rhss_index0_valid && ps_to_rhss_index1_valid && ys_to_rhs_major_valid;

//     // if ndimy >= 2, others is iteration number per thread
//     // get ys_to_rhs_major[ndim_y-2] and ys_to_rhs_minor[ndim_y-2]
//     // static constexpr index_t iteration_number = ys_to_rhs_major[ndimy_size]
//     // get other dims in ps_to_rhss_major from [0:ndimp-2)
//     // static constexpr index_t iteration_number =
//     //     ndimy_size == 1 ? 1
//     //                     : reduce_on_sequence(
//     //                           ys_to_rhs_major.extract(
//     //                               typename arithmetic_sequence_gen<0, ndimy_size - 1,
//     //                               1>::type{}),
//     //                           multiplies{},
//     //                           number<1>{});
// };

template <typename TileDistribution_, typename DataType_>
struct TransposeTileDistrChecker
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

    static constexpr auto input_hs_lengthss = InDstrEncode::hs_lengthss_;

    using ValidQuadDstrEncode =
        typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TileDistrEncode;

    static constexpr auto quad_hs_lengthss = ValidQuadDstrEncode::hs_lengthss_;

    static constexpr auto input_ps_to_rhss_major = InDstrEncode::ps_to_rhss_major_;
    static constexpr auto input_ps_to_rhss_minor = InDstrEncode::ps_to_rhss_minor_;
    static constexpr auto input_ys_to_rhs_major  = InDstrEncode::ys_to_rhs_major_;
    static constexpr auto input_ys_to_rhs_minor  = InDstrEncode::ys_to_rhs_minor_;

    // only support 2D transpose
    static constexpr bool hs_lengthss_size_valid =
        (InDstrEncode::NDimX == ValidQuadDstrEncode::NDimX);

    static constexpr bool hs_lengthss_size_dim_valid = (InDstrEncode::NDimX == 2);

    // each element of quad_hs_lengthss is suffix of input_hs_lengthss
    template <index_t I = 0>
    struct check_hs_lengthss_suffixes
    {
        static constexpr bool value =
            util::is_sequence_suffix_v<decltype(quad_hs_lengthss.template get<I>()),
                                       decltype(input_hs_lengthss.template get<I>())> &&
            check_hs_lengthss_suffixes<I + 1>::value;
    };

    template <>
    struct check_hs_lengthss_suffixes<input_hs_lengthss.size()>
    {
        static constexpr bool value = true;
    };

    static constexpr bool all_hs_lengthss_are_suffixes = check_hs_lengthss_suffixes<>::value;

    static constexpr index_t ndimp_outer_size = input_ps_to_rhss_major.size();

    static constexpr index_t ndimp_inner_size =
        input_ps_to_rhss_major[number<ndimp_outer_size - 1>{}].size();

    // make sure ps_to_rhss_major[ndimp-1] == 2
    // make sure ps_to_rhss_minor[ndimp-1] == input_hs_lengthss[1].size - 2;
    static constexpr bool ps_to_rhss_index0_valid =
        (input_ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] == 2) &&
        (input_ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 1] ==
         input_hs_lengthss[number<1>{}].size() - 2);

    // make sure ps_to_rhss_major[ndimp-2] == 1
    // make sure ps_to_rhss_minor[ndimp-2] == input_hs_lengthss[0].size - 1;
    static constexpr bool ps_to_rhss_index1_valid =
        (input_ps_to_rhss_major[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] == 1) &&
        (input_ps_to_rhss_minor[number<ndimp_outer_size - 1>{}][ndimp_inner_size - 2] ==
         input_hs_lengthss[number<0>{}].size() - 1);

    static constexpr index_t ndimy_size = input_ys_to_rhs_major.size();
    static constexpr bool ys_to_rhs_major_valid =
        ((input_ys_to_rhs_major[ndimy_size - 1] == 2) &&
         (input_ys_to_rhs_minor[ndimy_size - 1] == input_hs_lengthss[number<1>{}].size() - 1)) &&
        ((input_ys_to_rhs_major[ndimy_size - 2] == 1) &&
         (input_ys_to_rhs_minor[ndimy_size - 2] == input_hs_lengthss[number<0>{}].size() - 2));

    static constexpr bool distr_encoding_valid =
        hs_lengthss_size_valid && hs_lengthss_size_dim_valid && all_hs_lengthss_are_suffixes &&
        ps_to_rhss_index0_valid && ps_to_rhss_index1_valid && ys_to_rhs_major_valid;
};

template <typename TileDistribution_, typename DataType_>
struct OutputTileDistributionTraits
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;
    static constexpr auto input_hs_lengthss = InDstrEncode::hs_lengthss_;
    using ValidQuadDstrEncode =
        typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TileDistrEncode;
    using ValidOutQuadDstrEncode =
        typename valid_quad_tile_dstr_encode_for_transpose<DataType_>::TransposedTileDistrEncode;

    static constexpr auto quad_input_hs_lengthss      = ValidQuadDstrEncode::hs_lengthss_;
    static constexpr auto quad_output_hs_lengthss     = ValidOutQuadDstrEncode::hs_lengthss_;
    static constexpr auto input_ps_to_rhss_major      = InDstrEncode::ps_to_rhss_major_;
    static constexpr auto input_ps_to_rhss_minor      = InDstrEncode::ps_to_rhss_minor_;
    static constexpr auto input_ys_to_rhs_major       = InDstrEncode::ys_to_rhs_major_;
    static constexpr auto input_ys_to_rhs_minor       = InDstrEncode::ys_to_rhs_minor_;

    static constexpr auto quad_ps_to_rhss_major = ValidQuadDstrEncode::ps_to_rhss_major_;
    static constexpr auto quad_ps_to_rhss_minor = ValidQuadDstrEncode::ps_to_rhss_minor_;
    
    //for transpose load
    static constexpr auto reversed_quad_output_hs_lengthss = tuple_reverse(quad_output_hs_lengthss);

    static constexpr auto full_out_hs_lengthss = generate_tuple(
        [](auto i) {
            return input_hs_lengthss[i]
                .extract(typename arithmetic_sequence_gen<0,
                    input_hs_lengthss[i].size() - quad_input_hs_lengthss[i].size(),
                    1>::type{})
                .push_back(reversed_quad_output_hs_lengthss[i]);
        },
        number<InDstrEncode::NDimX>{});
    
    static constexpr auto dst_out_hs_lengthss = tuple_reverse(full_out_hs_lengthss);
    
    static constexpr auto modified_ps_to_rhss_major = generate_tuple(
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_major.size() - 1)
            {
                constexpr auto current_size = input_ps_to_rhss_major[i].size();
                constexpr auto reduce_size = quad_ps_to_rhss_major[number<0>{}].size();
                constexpr auto reduced_ps_to_rhss_major = input_ps_to_rhss_major[i].extract(
                    typename arithmetic_sequence_gen<0, current_size - reduce_size, 1>::type{});
                return reduced_ps_to_rhss_major.push_back(number<2>{});
                //return reduced_ps_to_rhss_major;
            }
            else
            {
                // For all other sequences, keep them unchanged
                return input_ps_to_rhss_major[i];
            }
        },
        number<input_ps_to_rhss_major.size()>{});
    
    static constexpr auto minor_last_index = full_out_hs_lengthss[number<InDstrEncode::NDimX-1>{}].size()-1;
    static constexpr auto major_last_index = full_out_hs_lengthss[number<0>{}].size()-1;

    static constexpr auto dst_ps_to_rhss_minor = generate_tuple(
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_minor.size() - 1)
            {
                constexpr auto current_size = input_ps_to_rhss_minor[i].size();
                constexpr auto reduce_size = quad_ps_to_rhss_minor[number<0>{}].size();
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
    static constexpr auto dst_ps_to_rhss_major = generate_tuple(
            [](auto i) {
                constexpr auto seq = modified_ps_to_rhss_major[i];
                return generate_sequence_v2(
                    [&](auto j) { if constexpr(seq[j] == 1) 
                                      {return number<2>{};}
                                  else if constexpr(seq[j] == 2)
                                      {return number<1>{};}
                                  else {return seq[j];}},
                    number<seq.size()>{});
            },
    number<modified_ps_to_rhss_major.size()>{});

    static constexpr auto modified_input_ys_to_rhs_major = input_ys_to_rhs_major.pop_back().push_back(number<1>{});

    static constexpr auto dst_ys_to_rhs_major = generate_sequence_v2(
         [](auto i) {
            if constexpr(modified_input_ys_to_rhs_major[i]==1)
            {return number<2>{};}
            else if constexpr(modified_input_ys_to_rhs_major[i]==2)
            {return number<1>{};}
            else
            {return modified_input_ys_to_rhs_major[i];}
         },
         number<modified_input_ys_to_rhs_major.size()>{}
    );

    static constexpr auto dst_ys_to_rhs_minor = input_ys_to_rhs_minor.pop_back().push_back(number<major_last_index>{});



    using OutDstrEncode = tile_distribution_encoding<typename InDstrEncode::RsLengths,
                                                     remove_cvref_t<decltype(dst_out_hs_lengthss)>,
                                                     remove_cvref_t<decltype(dst_ps_to_rhss_major)>,
                                                     remove_cvref_t<decltype(dst_ps_to_rhss_minor)>,
                                                     remove_cvref_t<decltype(dst_ys_to_rhs_major)>,
                                                     remove_cvref_t<decltype(dst_ys_to_rhs_minor)>>;
};

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true,
          typename                   = std::enable_if_t<TransposeTileDistrChecker<
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
    using OutTileDstrEncode =
        typename OutputTileDistributionTraits<TileDistribution_,
                                              typename BottomTensorView_::DataType>::OutDstrEncode;
    //Debug<OutTileDstrEncode> cccc;
    auto out_tensor = make_static_distributed_tensor<typename BottomTensorView_::DataType>(
        make_static_tile_distribution(OutTileDstrEncode{}));
    auto trans_tensor =
        tile_window.load_transpose(number<i_access>{}, bool_constant<oob_conditional_check>{});
    //return trans_tensor;
    constexpr auto lds_load_distr  = TileDistribution_{};
    constexpr auto glb_store_distr = make_static_tile_distribution(OutTileDstrEncode{});

    constexpr auto y_in_desc  = lds_load_distr.get_ys_to_d_descriptor();
    constexpr auto y_out_desc = glb_store_distr.get_ys_to_d_descriptor();

    constexpr index_t NDimYIn  = lds_load_distr.get_num_of_dimension_y();
    constexpr index_t NDimYOut = glb_store_distr.get_num_of_dimension_y();

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

    // constexpr auto lds_distr_y_indx_zeros = uniform_sequence_gen_t<decltype(Policy::template
    // MakeLdsLoadTileDistribution<Problem>())::NDimY, 0>{};

    using DataVec = array<typename BottomTensorView_::DataType, vecLoadSize>;
    static_for<0, num_of_access, 1>{}([&](auto iAccess) {
        out_tensor.get_thread_buffer().template set_as<DataVec>(
            number<iAccess>{}, trans_tensor.get_thread_buffer().template get_as<DataVec>(number<iAccess>{}));
    });

    return out_tensor;
}

} // namespace ck_tile
