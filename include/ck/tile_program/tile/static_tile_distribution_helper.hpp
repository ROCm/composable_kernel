// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {
namespace detail {

template <typename Distribution>
__host__ __device__ auto get_partition_index(Distribution)
{
    // only support warp-tile and block-tile
    static_assert(Distribution::NDimP == 1 or Distribution::NDimP == 2, "wrong!");

    if constexpr(Distribution::NDimP == 1)
    {
        return Array<index_t, 1>{get_lane_id()};
    }
    else if constexpr(Distribution::NDimP == 2)
    {
        return Array<index_t, 2>{get_warp_id(), get_lane_id()};
    }
}

template <typename OuterDstr, typename InnerDstr>
__host__ __device__ constexpr auto make_embed_tile_distribution_encoding(OuterDstr, InnerDstr)
{
    static_assert(OuterDstr::NDimX == InnerDstr::NDimX, "wrong!");

    constexpr index_t NDimHMajor = OuterDstr::NDimX;

    using RsLengths =
        sequence_merge_t<typename OuterDstr::RsLengths, typename InnerDstr::RsLengths>;

    constexpr auto hs_lengthss = generate_tuple(
        [&](auto i) {
            return merge_sequences(typename OuterDstr::HsLengthss{}[i],
                                   typename InnerDstr::HsLengthss{}[i]);
        },
        Number<NDimHMajor>{});

    //
    constexpr auto rhs_major_2_ndim_outer_rhs_minor = [&]() {
        Array<index_t, NDimHMajor + 1> rhs_major_2_ndim_outer_rhs_minor_;

        // R dimension
        rhs_major_2_ndim_outer_rhs_minor_(0) = OuterDstr::RsLengths::Size();

        // Hs dimensions
        static_for<0, NDimHMajor, 1>{}([&](auto i) {
            rhs_major_2_ndim_outer_rhs_minor_(i + 1) = typename OuterDstr::HsLengthss{}[i].Size();
        });

        return rhs_major_2_ndim_outer_rhs_minor_;
    }();

    // Ps2RHssMinor
    constexpr auto updated_inner_ps_2_rhss_minor = generate_tuple(
        [&](auto p) {
            constexpr auto inner_p_2_rhss_major = typename InnerDstr::Ps2RHssMajor{}[p];
            constexpr auto inner_p_2_rhss_minor = typename InnerDstr::Ps2RHssMinor{}[p];

            constexpr index_t ndim_tmp = inner_p_2_rhss_minor.Size();

            constexpr auto updated_inner_p_2_rhss_minor = [&]() {
                Array<index_t, ndim_tmp> updated_inner_p_2_rhss_minor_;

                for(index_t i = 0; i < ndim_tmp; i++)
                {
                    index_t rh_major = inner_p_2_rhss_major[i];

                    index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                    updated_inner_p_2_rhss_minor_(i) = inner_p_2_rhss_minor[i] + ndim_outer_h_minor;
                }

                return updated_inner_p_2_rhss_minor_;
            }();

            return TO_SEQUENCE(updated_inner_p_2_rhss_minor, ndim_tmp);
        },
        Number<InnerDstr::NDimP>{});

    // Ys2RHsMinor
    constexpr auto updated_inner_ys_2_rhs_minor = [&]() {
        constexpr auto inner_ys_2_rhs_major = typename InnerDstr::Ys2RHsMajor{};
        constexpr auto inner_ys_2_rhs_minor = typename InnerDstr::Ys2RHsMinor{};

        constexpr index_t ndim_tmp = inner_ys_2_rhs_minor.Size();

        constexpr auto updated_inner_ys_2_rhs_minor_ = [&]() {
            Array<index_t, ndim_tmp> updated_inner_ys_2_rhs_minor__;

            for(index_t i = 0; i < ndim_tmp; i++)
            {
                index_t rh_major = inner_ys_2_rhs_major[i];

                index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                updated_inner_ys_2_rhs_minor__(i) = inner_ys_2_rhs_minor[i] + ndim_outer_h_minor;
            }

            return updated_inner_ys_2_rhs_minor__;
        }();

        return TO_SEQUENCE(updated_inner_ys_2_rhs_minor_, ndim_tmp);
    }();

    //
    constexpr auto ps_2_rhss_major =
        container_concat(typename OuterDstr::Ps2RHssMajor{}, typename InnerDstr::Ps2RHssMajor{});

    constexpr auto ps_2_rhss_minor =
        container_concat(typename OuterDstr::Ps2RHssMinor{}, updated_inner_ps_2_rhss_minor);

    //
    constexpr auto ys_2_rhs_major =
        merge_sequences(typename OuterDstr::Ys2RHsMajor{}, typename InnerDstr::Ys2RHsMajor{});

    constexpr auto ys_2_rhs_minor =
        merge_sequences(typename OuterDstr::Ys2RHsMinor{}, updated_inner_ys_2_rhs_minor);

    return StaticTileDistributionEncoding<RsLengths,
                                          remove_cvref_t<decltype(hs_lengthss)>,
                                          remove_cvref_t<decltype(ps_2_rhss_major)>,
                                          remove_cvref_t<decltype(ps_2_rhss_minor)>,
                                          remove_cvref_t<decltype(ys_2_rhs_major)>,
                                          remove_cvref_t<decltype(ys_2_rhs_minor)>>{};
}

template <typename InDstr, index_t... InReduceDimXs>
__host__ __device__ constexpr auto
make_reduce_tile_distribution_encoding_impl(InDstr, Sequence<InReduceDimXs...> reduce_dim_xs_in)
{
    constexpr auto I1 = Number<1>{};

    // FIXME: increase if fail
    constexpr index_t max_ndim_r_out = 20;
    constexpr index_t max_ndim_y_out = 20;

    //
    constexpr index_t ndim_p               = InDstr::NDimP;
    constexpr index_t ndim_x_in            = InDstr::NDimX;
    constexpr index_t ndim_y_in            = InDstr::NDimY;
    constexpr index_t ndim_rh_major_in     = InDstr::NDimX + 1;
    constexpr index_t ndim_x_out           = ndim_x_in - sizeof...(InReduceDimXs);
    constexpr index_t max_ndim_rh_minor_in = InDstr::Detail::max_ndim_rh_minor_;

    // ndims_ps_low
    constexpr auto ndims_ps_low = generate_array(
        [&](auto i) { return InDstr::ps_to_rhss_major_[i].Size(); }, Number<ndim_p>{});

    // is_rh_major_in_for_reduce
    Array<bool, ndim_rh_major_in> is_rh_major_in_for_reduce{false};

    for(index_t i = 0; i < reduce_dim_xs_in.Size(); i++)
    {
        index_t rh_major = reduce_dim_xs_in[i] + 1;

        is_rh_major_in_for_reduce(rh_major) = true;
    }

    // is_y_in_for_reduce
    Array<bool, ndim_y_in> is_y_in_for_reduce{false};

    for(index_t i = 0; i < ndim_y_in; i++)
    {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];

        if(is_rh_major_in_for_reduce[rh_major])
        {
            is_y_in_for_reduce(i) = true;
        }
    }

    // is_rh_minor_in_for_y_reduce
    Array<Array<bool, max_ndim_rh_minor_in>, ndim_rh_major_in> is_rh_minor_in_for_y_reduce{{false}};

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];
        index_t rh_minor = InDstr::ys_to_rhs_minor_[i];

        if(is_y_in_for_reduce[i])
        {
            is_rh_minor_in_for_y_reduce(rh_major)(rh_minor) = true;
        }
    });

    // in2out_rh_major
    Array<index_t, ndim_rh_major_in> in2out_rh_major{-1};
    index_t cnt_ndim_rh_major_out = 0;

    for(index_t i = 0; i < ndim_rh_major_in; i++)
    {
        if(is_rh_major_in_for_reduce[i])
        {
            in2out_rh_major(i) = 0;
        }
        else
        {
            in2out_rh_major(i) = cnt_ndim_rh_major_out;

            cnt_ndim_rh_major_out++;
        }
    }

    // rs_lengths_out, in2out_rh_minor
    Array<index_t, max_ndim_r_out> rs_lengths_out{-1};
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_rh_major_in> in2out_rh_minor{{-1}};

    // loop over input R dim
    for(index_t i = 0; i < InDstr::rs_lengths_.Size(); i++)
    {
        // rs_lengths_out
        rs_lengths_out(i) = InDstr::rs_lengths_[i];

        // in2out_rh_minor
        in2out_rh_minor(0)(i) = i;
    }

    // loop over input H Dim
    index_t cnt_ndim_r_out = InDstr::rs_lengths_.Size();

    static_for<1, ndim_rh_major_in, 1>{}([&](auto rh_major_in) {
        constexpr auto h_major_in = rh_major_in - I1;

        constexpr index_t ndim_rh_minor_in = InDstr::hs_lengthss_[h_major_in].Size();

        if(is_rh_major_in_for_reduce[rh_major_in])
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                if(not is_rh_minor_in_for_y_reduce[rh_major_in][rh_minor_in])
                {
                    // rs_lengths_out
                    rs_lengths_out(cnt_ndim_r_out) = InDstr::hs_lengthss_[h_major_in][rh_minor_in];

                    // in2out_rh_minor
                    in2out_rh_minor(rh_major_in)(rh_minor_in) = cnt_ndim_r_out;

                    cnt_ndim_r_out++;
                }
            }
        }
        else
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                // in2out_rh_minor
                in2out_rh_minor(rh_major_in)(rh_minor_in) = rh_minor_in;
            }
        }
    });

    // ndim_r_out
    const index_t ndim_r_out = cnt_ndim_r_out;

    // ndims_hs_minor_out, hs_lengthss_out
    Array<index_t, ndim_x_out> ndims_hs_minor_out{-1};
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_x_out> hs_lengthss_out{{-1}};

    index_t cnt_ndim_x_out = 0;

    static_for<0, ndim_x_in, 1>{}([&](auto i) {
        if(not is_rh_major_in_for_reduce[i + I1])
        {
            // ndims_hs_minor_out
            ndims_hs_minor_out(cnt_ndim_x_out) = InDstr::hs_lengthss_[i].Size();

            // hs_lengthss_out
            static_for<0, InDstr::hs_lengthss_[i].Size(), 1>{}(
                [&](auto j) { hs_lengthss_out(cnt_ndim_x_out)(j) = InDstr::hs_lengthss_[i][j]; });

            cnt_ndim_x_out++;
        }
    });

    // ps_to_rhss_major_out, ps_to_rhss_minor_out
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_major_out{{-1}};
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_minor_out{{-1}};

    static_for<0, ndim_p, 1>{}([&](auto idim_p) {
        static_for<0, InDstr::ps_to_rhss_major_[idim_p].Size(), 1>{}([&](auto idim_low) {
            index_t rh_major_in = InDstr::ps_to_rhss_major_[idim_p][idim_low];
            index_t rh_minor_in = InDstr::ps_to_rhss_minor_[idim_p][idim_low];

            ps_to_rhss_major_out(idim_p)(idim_low) = in2out_rh_major[rh_major_in];
            ps_to_rhss_minor_out(idim_p)(idim_low) = in2out_rh_minor[rh_major_in][rh_minor_in];
        });
    });

    // ys_to_rhs_major_out, ys_to_rhs_minor_out
    Array<index_t, max_ndim_y_out> ys_to_rhs_major_out{-1};
    Array<index_t, max_ndim_y_out> ys_to_rhs_minor_out{-1};

    index_t cnt_ndim_y_out = 0;

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        if(not is_y_in_for_reduce[i])
        {
            index_t rh_major_in = InDstr::ys_to_rhs_major_[i];
            index_t rh_minor_in = InDstr::ys_to_rhs_minor_[i];

            ys_to_rhs_major_out(cnt_ndim_y_out) = in2out_rh_major[rh_major_in];
            ys_to_rhs_minor_out(cnt_ndim_y_out) = in2out_rh_minor[rh_major_in][rh_minor_in];

            cnt_ndim_y_out++;
        }
    });

    // ndim_y_out
    const index_t ndim_y_out = cnt_ndim_y_out;

    //
    return make_tuple(ndim_x_out,
                      ndim_p,
                      ndim_y_out,
                      ndim_r_out,
                      ndims_hs_minor_out,
                      ndims_ps_low,
                      rs_lengths_out,
                      hs_lengthss_out,
                      ps_to_rhss_major_out,
                      ps_to_rhss_minor_out,
                      ys_to_rhs_major_out,
                      ys_to_rhs_minor_out);
}

template <typename InDstr, index_t... InReduceDimXs>
__host__ __device__ constexpr auto
make_reduce_tile_distribution_encoding(InDstr, Sequence<InReduceDimXs...> reduce_dim_xs_in)
{
    constexpr auto impl = make_reduce_tile_distribution_encoding_impl(InDstr{}, reduce_dim_xs_in);

    constexpr index_t ndim_x             = impl.template At<0>();
    constexpr index_t ndim_p             = impl.template At<1>();
    constexpr index_t ndim_y             = impl.template At<2>();
    constexpr index_t ndim_r             = impl.template At<3>();
    constexpr auto ndims_hs_minor        = impl.template At<4>();
    constexpr auto ndims_ps_low          = impl.template At<5>();
    constexpr auto rs_lengths_impl       = impl.template At<6>();
    constexpr auto hs_lengthss_impl      = impl.template At<7>();
    constexpr auto ps_to_rhss_major_impl = impl.template At<8>();
    constexpr auto ps_to_rhss_minor_impl = impl.template At<9>();
    constexpr auto ys_to_rhs_major_impl  = impl.template At<10>();
    constexpr auto ys_to_rhs_minor_impl  = impl.template At<11>();

    constexpr auto rs_lengths  = TO_SEQUENCE(rs_lengths_impl, ndim_r);
    constexpr auto hs_lengthss = TO_TUPLE_OF_SEQUENCE(hs_lengthss_impl, ndim_x, ndims_hs_minor);
    constexpr auto ps_to_rhss_major =
        TO_TUPLE_OF_SEQUENCE(ps_to_rhss_major_impl, ndim_p, ndims_ps_low);
    constexpr auto ps_to_rhss_minor =
        TO_TUPLE_OF_SEQUENCE(ps_to_rhss_minor_impl, ndim_p, ndims_ps_low);
    constexpr auto ys_to_rhs_major = TO_SEQUENCE(ys_to_rhs_major_impl, ndim_y);
    constexpr auto ys_to_rhs_minor = TO_SEQUENCE(ys_to_rhs_minor_impl, ndim_y);

    return StaticTileDistributionEncoding<remove_cvref_t<decltype(rs_lengths)>,
                                          remove_cvref_t<decltype(hs_lengthss)>,
                                          remove_cvref_t<decltype(ps_to_rhss_major)>,
                                          remove_cvref_t<decltype(ps_to_rhss_minor)>,
                                          remove_cvref_t<decltype(ys_to_rhs_major)>,
                                          remove_cvref_t<decltype(ys_to_rhs_minor)>>{};
}

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
        auto y_slice_sorted_origins = make_zero_multi_index<Encoding::NDimY>();
        auto y_slice_lengths        = Encoding::Detail::ys_lengths_;

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

                    auto y_origin_ = make_zero_multi_index<Encoding::NDimY>();
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

    constexpr auto sliced_h_lengths       = sliced_hlen_yidx_ylen[Number<0>{}];
    constexpr auto sliced_y_origins_array = sliced_hlen_yidx_ylen[Number<1>{}];
    constexpr auto sliced_y_origins_size  = sliced_y_origins_array.Size();
    constexpr auto sliced_y_lengths_array = sliced_hlen_yidx_ylen[Number<2>{}];
    constexpr auto sliced_y_lengths_size  = sliced_y_lengths_array.Size();

    constexpr auto sliced_y_origins = TO_SEQUENCE(sliced_y_origins_array, sliced_y_origins_size);
    constexpr auto sliced_y_lengths = TO_SEQUENCE(sliced_y_lengths_array, sliced_y_lengths_size);

    return make_tuple(
        make_static_tile_distribution(
            StaticTileDistributionEncoding<typename Encoding::RsLengths,
                                           decltype(sliced_h_lengths), // only need to change the
                                                                       // h_lengths type
                                           typename Encoding::Ps2RHssMajor,
                                           typename Encoding::Ps2RHssMinor,
                                           typename Encoding::Ys2RHsMajor,
                                           typename Encoding::Ys2RHsMinor>{}),
        sliced_y_origins,
        sliced_y_lengths);
}

} // namespace detail
} // namespace tile_program
} // namespace ck
