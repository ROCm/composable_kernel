// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

namespace detail {

template <typename, typename, typename, index_t>
struct reverse_slice_sequence_impl;

template <index_t x,
          index_t... xs,
          index_t m,
          index_t... ms,
          index_t id,
          index_t... ids,
          index_t SliceSize>
struct reverse_slice_sequence_impl<Sequence<x, xs...>,
                                   Sequence<m, ms...>,
                                   Sequence<id, ids...>,
                                   SliceSize>
{
    using old_scan =
        reverse_slice_sequence_impl<Sequence<xs...>, Sequence<ms...>, Sequence<ids...>, SliceSize>;

    static constexpr auto slice_size = old_scan::remaining_slice_sizes::Front().value;
    static constexpr auto slice_length =
        std::conditional_t<m, Number<math::gcd(x, slice_size)>, Number<x>>::value;

    using dim_lengths =
        typename sequence_merge<Sequence<slice_length>, typename old_scan::dim_lengths>::type;
    using dim_slices =
        typename sequence_merge<Sequence<x / slice_length>, typename old_scan::dim_slices>::type;
    using remaining_slice_sizes = typename sequence_merge<
        std::conditional_t<m, Sequence<slice_size / slice_length>, Sequence<slice_size>>,
        typename old_scan::remaining_slice_sizes>::type;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.Front().value == 1;
    static constexpr index_t _split_flag = std::conditional_t<m, Number<_flag>, Number<0>>::value;
    static constexpr index_t _split_idx =
        std::conditional_t<_split_flag, Number<id>, Number<0>>::value;

    static constexpr index_t split_flag = _split_flag || old_scan::split_flag;
    static constexpr index_t split_idx  = std::
        conditional_t<old_scan::split_flag, Number<old_scan::split_idx>, Number<_split_idx>>::value;
};

template <index_t x, index_t m, index_t id, index_t SliceSize>
struct reverse_slice_sequence_impl<Sequence<x>, Sequence<m>, Sequence<id>, SliceSize>
{
    static constexpr auto slice_size = SliceSize;
    static constexpr auto slice_length =
        std::conditional_t<m, Number<math::gcd(x, slice_size)>, Number<x>>::value;

    using dim_lengths = Sequence<slice_length>;
    using dim_slices  = Sequence<x / slice_length>;
    using remaining_slice_sizes =
        std::conditional_t<m, Sequence<slice_size / slice_length>, Sequence<slice_size>>;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.Front().value == 1;
    static constexpr index_t split_flag = std::conditional_t<m, Number<_flag>, Number<0>>::value;
    static constexpr index_t split_idx =
        std::conditional_t<split_flag, Number<id>, Number<0>>::value;
};

// clang-format off
// input a sequence(with optional mask), and the SliceSize : size per slice
// output the sequence each slice, and Number of slices
//
// e.g. <2, 1, 4, 2>, 8     -> lengths:<1, 1, 4, 2>    , nums: <2, 1, 1, 1>    : 2 slices  , slice_idx: 0
//      <4, 2, 4, 1, 2>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 1> : 16 slices , slice_idx: 2
//      <4, 2, 4, 1, 6>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 3> : 48 slices , slice_idx: 2
//      <4, 2, 5, 1, 2>, 10 -> lengths:<1, 1, 5, 1, 2> , nums: <4, 2, 1, 1, 1> : 8 slices  , slice_idx: 1
//
//      <4, 2, 8>, 64       -> lengths:<4, 2, 8>       , nums: <1, 1, 1>       : 1  slices , slice_idx: 0
//      <4, 2, 8>, 32       -> lengths:<2, 2, 8>       , nums: <2, 1, 1>       : 2  slices , slice_idx: 0
//      <4, 2, 8>, 16       -> lengths:<1, 2, 8>       , nums: <4, 1, 1>       : 4  slices , slice_idx: 0
//      <4, 2, 8>, 8        -> lengths:<1, 1, 8>       , nums: <4, 2, 1>       : 8  slices , slice_idx: 1
//      <4, 2, 8>, 4        -> lengths:<1, 1, 4>       , nums: <4, 2, 2>       : 16 slices , slice_idx: 2
//      <4, 2, 8>, 2        -> lengths:<1, 1, 2>       , nums: <4, 2, 4>       : 32 slices , slice_idx: 2
//      <4, 2, 8>, 1        -> lengths:<1, 1, 1>       , nums: <4, 2, 8>       : 64 slices , slice_idx: 2
//
//      <4, 2, 1, 4, 2> / 4 ->
// mask:<1, 1, 1, 0, 1>,    -> lengths:<1, 2, 1, 4, 2> , nums: <4, 1, 1, 1, 1> : 8 slices  , slice_idx: 0
//
// return Tuple<slice_lengths, slice_nums, slice_index>, slice_index is at which index will start
// have split slices (right -> left)
//  or the first index that sliced length is different from the original length
// clang-format on
template <typename Seq,
          index_t SliceSize,
          typename Mask = typename uniform_sequence_gen<Seq::Size(), 1>::type>
constexpr auto reverse_slice_sequence(Seq,
                                      Number<SliceSize>,
                                      Mask = typename uniform_sequence_gen<Seq::Size(), 1>::type{})
{
    static_assert(Seq::Size() == Mask::Size());
    using sliced_type =
        reverse_slice_sequence_impl<Seq,
                                    Mask,
                                    typename arithmetic_sequence_gen<0, Seq::Size(), 1>::type,
                                    SliceSize>;
    static_assert(sliced_type::remaining_slice_sizes::Front().value == 1,
                  "can not evenly divide this sequence, please check");
    return make_tuple(typename sliced_type::dim_lengths{},
                      typename sliced_type::dim_slices{},
                      Number<sliced_type::split_idx>{});
}

//
// slice tensor from x_dim, result in split in y_dim, not p_dim.
// We don't support slice cross p_dim (aka, slice different threads)
// also, sliced along y_dim need be the first dim of current dim.
// Multiply Y dim before sliced dim does not make sense
//
// e.g
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 32>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 4, 2, 4> -> OK
//                     |--> slice along this Y dim, is the first dim of X1, totally 4 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 8>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 2, 4> -> OK
//                           |--> slice along this Y dim, the P dim is 1 in the left, so is OK
//                                 totally 16 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 4>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 1, 4> -> Fail
//                              |--> slice along this P dim, will split threads, not supported
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 16>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 2, 2, 4> -> OK
//                           |--> slice along this Y dim, but this Y sim need to split into 2
//                           subdime
//                                the P dim in the left is 1, means actually not crossing P
//
template <typename Distribution, index_t... XSliceBegins, index_t... XSliceEnds>
__host__ __device__ constexpr auto slice_distribution_from_x(
    Distribution, Sequence<XSliceBegins...> x_slice_begins, Sequence<XSliceEnds...> x_slice_ends)
{
    // NOTE: this function need to be called under constexpr context,
    // due to https://wg21.link/p2280r0 we have to use non-reference type for distribution
    using Encoding = decltype(Distribution::GetStaticTileDistributionEncoding());

    static_assert(sizeof...(XSliceBegins) == sizeof...(XSliceEnds));

    constexpr auto x_slice_lengths = x_slice_ends - x_slice_begins;

    constexpr auto src_h_prefix_sum = Encoding::Detail::GetHDimLengthsPrefixSum();
    constexpr auto src_y_info       = Encoding::Detail::GetSortedYInfo();
    constexpr auto src_y_dims       = src_y_info[Number<0>{}];
    constexpr auto src_y_maps       = src_y_info[Number<1>{}];
    constexpr auto src_y_prefix_sum = src_y_info[Number<2>{}];

    constexpr auto sliced_hlen_yidx_ylen = [&]() constexpr
    {
        auto y_slice_sorted_origins = make_zero_multi_index<Distribution::NDimY>();
        auto y_slice_lengths =
            to_array<index_t, Distribution::NDimY>(Distribution{}.GetYs2DDescriptor().GetLengths());

        // This lambda will modify some value outside, so c++ will not treat return value as
        // constexpr
        // TODO: ugly
        auto new_h_lengths = transform_tuples(
            [&](auto h_len, auto id) {
                constexpr auto sliced_h =
                    reverse_slice_sequence(h_len, Number<x_slice_lengths[id]>{});

                constexpr auto sliced_h_lens  = sliced_h[Number<0>{}];
                constexpr auto sliced_h_index = sliced_h[Number<2>{}];

                // update y_slice_lengths
                constexpr auto uniformed_h_index = sliced_h_index + Number<src_h_prefix_sum[id]>{};
                constexpr auto found_y_index     = container_find(src_y_dims, uniformed_h_index);

                static_assert(found_y_index >= 0 && found_y_index < src_y_dims.Size(),
                              "not sliced at y dim, please check");

                static_for<0, sliced_h_index + 1, 1>{}([&](auto i) {
                    y_slice_lengths(src_y_maps[found_y_index - i]) =
                        sliced_h_lens[sliced_h_index - i];
                });
                // TODO: add validations not across p dim

                // NOTE: this y_origin is for all dims, not only current dim
                //       will later use pick to select target dim
                constexpr auto y_origin = [&]() {
                    constexpr auto h_trans = make_merge_transform_v3_division_mod(h_len);
                    auto h_origin_         = make_zero_multi_index<h_trans.NDimLow>();
                    h_trans.CalculateLowerIndex(h_origin_, Sequence<x_slice_begins[id].value>{});

                    auto y_origin_ = make_zero_multi_index<Distribution::NDimY>();
                    static_for<0, sliced_h_index + 1, 1>{}([&](auto i) {
                        y_origin_(found_y_index - i) = h_origin_[sliced_h_index - i];
                    });
                    return y_origin_;
                }();

                constexpr auto y_picks = typename arithmetic_sequence_gen<src_y_prefix_sum[id],
                                                                          src_y_prefix_sum[id + 1],
                                                                          1>::type{};

                set_container_subset(
                    y_slice_sorted_origins, y_picks, get_container_subset(y_origin, y_picks));
                return sliced_h_lens;
            },
            typename Encoding::HsLengthss{},
            typename arithmetic_sequence_gen<0, Encoding::HsLengthss::Size(), 1>::type{});

        auto y_slice_origins = container_reorder_given_old2new(y_slice_sorted_origins, src_y_maps);
        return make_tuple(new_h_lengths, y_slice_origins, y_slice_lengths);
    }
    ();

    return sliced_hlen_yidx_ylen;
}

} // namespace detail

template <typename StaticDistributedTensor_, index_t... SliceBegins, index_t... SliceEnds>
__host__ __device__ constexpr auto get_slice_tile(const StaticDistributedTensor_& tile,
                                                  Sequence<SliceBegins...> slice_begins,
                                                  Sequence<SliceEnds...> slice_ends)
{
    using Distribution = decltype(StaticDistributedTensor_::GetTileDistribution());
    using Encoding     = decltype(Distribution::GetStaticTileDistributionEncoding());
    using DataType     = typename StaticDistributedTensor_::DataType;

    constexpr auto sliced_hlen_yidx_ylen =
        detail::slice_distribution_from_x(Distribution{}, slice_begins, slice_ends);

    constexpr auto sliced_h_lengths       = sliced_hlen_yidx_ylen[Number<0>{}];
    constexpr auto sliced_y_origins_array = sliced_hlen_yidx_ylen[Number<1>{}];
    constexpr auto sliced_y_origins_size  = sliced_y_origins_array.Size();
    constexpr auto sliced_y_lengths_array = sliced_hlen_yidx_ylen[Number<2>{}];
    constexpr auto sliced_y_lengths_size  = sliced_y_lengths_array.Size();

    constexpr auto sliced_y_origins = TO_SEQUENCE(sliced_y_origins_array, sliced_y_origins_size);
    constexpr auto sliced_y_lengths = TO_SEQUENCE(sliced_y_lengths_array, sliced_y_lengths_size);

    using SlicedEnc =
        StaticTileDistributionEncoding<typename Encoding::RsLengths,
                                       decltype(sliced_h_lengths), // only need to change the
                                                                   // h_lengths type
                                       typename Encoding::Ps2RHssMajor,
                                       typename Encoding::Ps2RHssMinor,
                                       typename Encoding::Ys2RHsMajor,
                                       typename Encoding::Ys2RHsMinor>;

    auto sliced_tensor =
        make_static_distributed_tensor<DataType>(make_static_tile_distribution(SlicedEnc{}));

    sliced_tensor.GetThreadBuffer() = tile.GetSlicedThreadData(sliced_y_origins, sliced_y_lengths);

    return sliced_tensor;
}

template <typename DstStaticDistributedTensor_,
          typename SrcStaticDistributedTensor_,
          index_t... SliceBegins,
          index_t... SliceEnds>
__host__ __device__ constexpr auto set_slice_tile(DstStaticDistributedTensor_& dst_tile,
                                                  const SrcStaticDistributedTensor_& src_tile,
                                                  Sequence<SliceBegins...> slice_begins,
                                                  Sequence<SliceEnds...> slice_ends)
{
    using DstDistribution = decltype(DstStaticDistributedTensor_::GetTileDistribution());

    constexpr auto sliced_hlen_yidx_ylen =
        detail::slice_distribution_from_x(DstDistribution{}, slice_begins, slice_ends);

    constexpr auto sliced_h_lengths       = sliced_hlen_yidx_ylen[Number<0>{}];
    constexpr auto sliced_y_origins_array = sliced_hlen_yidx_ylen[Number<1>{}];
    constexpr auto sliced_y_origins_size  = sliced_y_origins_array.Size();
    constexpr auto sliced_y_lengths_array = sliced_hlen_yidx_ylen[Number<2>{}];
    constexpr auto sliced_y_lengths_size  = sliced_y_lengths_array.Size();

    constexpr auto sliced_y_origins = TO_SEQUENCE(sliced_y_origins_array, sliced_y_origins_size);
    constexpr auto sliced_y_lengths = TO_SEQUENCE(sliced_y_lengths_array, sliced_y_lengths_size);

    dst_tile.SetSlicedThreadData(sliced_y_origins, sliced_y_lengths, src_tile.GetThreadBuffer());
}

} // namespace tile_program
} // namespace ck
