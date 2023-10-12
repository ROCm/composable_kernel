// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {

template <typename RsLengths_,    // Sequence<...>
          typename HsLengthss_,   // Tuple<Sequence<...>, ...>
          typename Ps2RHssMajor_, // Tuple<Sequence<...>, ...>
          typename Ps2RHssMinor_, // Tuple<Sequence<...>, ...>
          typename Ys2RHsMajor_,  // Sequence<...>
          typename Ys2RHsMinor_>  // Sequence<...>
struct StaticTileDistributionEncoding
{
    using RsLengths    = remove_cvref_t<RsLengths_>;
    using HsLengthss   = remove_cvref_t<HsLengthss_>;
    using Ps2RHssMajor = remove_cvref_t<Ps2RHssMajor_>;
    using Ps2RHssMinor = remove_cvref_t<Ps2RHssMinor_>;
    using Ys2RHsMajor  = remove_cvref_t<Ys2RHsMajor_>;
    using Ys2RHsMinor  = remove_cvref_t<Ys2RHsMinor_>;

    static_assert(Ps2RHssMajor::Size() == Ps2RHssMinor::Size(), "wrong!");
    static_assert(Ys2RHsMajor::Size() == Ys2RHsMinor::Size(), "wrong!");

    static constexpr index_t NDimX = HsLengthss::Size();
    static constexpr index_t NDimP = Ps2RHssMajor::Size();
    static constexpr index_t NDimY = Ys2RHsMajor::Size();
    static constexpr index_t NDimR = RsLengths::Size();

    // FIXME: move into Detail
    static constexpr auto rs_lengths_       = RsLengths{};
    static constexpr auto hs_lengthss_      = HsLengthss{};
    static constexpr auto ps_to_rhss_major_ = Ps2RHssMajor{};
    static constexpr auto ps_to_rhss_minor_ = Ps2RHssMinor{};
    static constexpr auto ys_to_rhs_major_  = Ys2RHsMajor{};
    static constexpr auto ys_to_rhs_minor_  = Ys2RHsMinor{};

    // redundant but useful info
    // TODO: really bad code, should be over-hauled
    struct Detail
    {
        // ndim_rh_major_, ndim_span_mainor_
        static constexpr index_t ndim_rh_major_   = NDimX + 1;
        static constexpr index_t ndim_span_major_ = NDimX;

        // ndims_rhs_minor_[ndim_rh_major_]
        static constexpr auto ndims_rhs_minor_ = generate_array(
            [](auto i) {
                if constexpr(i.value == 0)
                {
                    return rs_lengths_.Size();
                }
                else
                {
                    return hs_lengthss_[i - Number<1>{}].Size();
                }
            },
            Number<ndim_rh_major_>{});

        // max_ndim_rh_minor_
        static constexpr index_t max_ndim_rh_minor_ =
            container_reduce(ndims_rhs_minor_, math::maximize<index_t>{}, 0);

        // rhs_lengthss_[ndim_rh_major_][max_ndim_rh_minor_]
        static constexpr auto rhs_lengthss_ =
            to_array_of_array(container_concat(make_tuple(rs_lengths_), hs_lengthss_));

        // ys_lengths_
        static constexpr auto ys_lengths_ = [] {
            Array<index_t, NDimY> ys_lengths_tmp{-1};

            for(index_t i = 0; i < NDimY; i++)
            {
                index_t rh_major = ys_to_rhs_major_[i];
                index_t rh_minor = ys_to_rhs_minor_[i];

                ys_lengths_tmp(i) = rhs_lengthss_[rh_major][rh_minor];
            }

            return ys_lengths_tmp;
        }();

        // rhs_major_minor_to_ys_[ndim_rh_majpr_][max_ndim_rh_minor_]
        static constexpr auto rhs_major_minor_to_ys_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, NDimX + 1> rhs_major_minor_to_ys_tmp{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t rh_major = ys_to_rhs_major_[i];
                constexpr index_t rh_minor = ys_to_rhs_minor_[i];

                rhs_major_minor_to_ys_tmp(rh_major)(rh_minor) = i;
            });

            return rhs_major_minor_to_ys_tmp;
        }();

        // ndims_span_minor_[NDimY]
        static constexpr auto ndims_span_minor_ = [] {
            Array<index_t, NDimX> ndims_span_minor{0};

            for(index_t i = 0; i < NDimY; i++)
            {
                const index_t span_major = ys_to_rhs_major_[i] - 1;

                ndims_span_minor(span_major)++;
            }

            return ndims_span_minor;
        }();

        // max_ndim_span_minor_
        static constexpr index_t max_ndim_span_minor_ =
            container_reduce(ndims_span_minor_, math::maximize<index_t>{}, 0);

        // rhs_major_minor_to_span_minor_ [ndim_rh_major_][max_ndim_rh_minor_]
        static constexpr auto rhs_major_minor_to_span_minor_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, ndim_rh_major_> rhs_major_minor_to_span_minor{
                {-1}};

            static_for<0, ndim_rh_major_, 1>{}([&](auto rh_major) {
                constexpr index_t ndim_rh_minor = ndims_rhs_minor_[rh_major];

                index_t cnt_ndim_span_minor = 0;

                static_for<0, ndim_rh_minor, 1>{}([&](auto rh_minor) {
                    constexpr index_t idim_y = rhs_major_minor_to_ys_[rh_major][rh_minor];

                    if(idim_y >= 0)
                    {
                        rhs_major_minor_to_span_minor(rh_major)(rh_minor) = cnt_ndim_span_minor;

                        cnt_ndim_span_minor++;
                    }
                });
            });

            return rhs_major_minor_to_span_minor;
        }();

        // ys_to_span_major_[NDimY]
        static constexpr auto ys_to_span_major_ =
            generate_array([](auto i) { return ys_to_rhs_major_[i] - 1; }, Number<NDimY>{});

        // ys_to_span_minor_[NDimY]
        static constexpr auto ys_to_span_minor_ = generate_array(
            [](auto i) {
                return rhs_major_minor_to_span_minor_[ys_to_rhs_major_[i]][ys_to_rhs_minor_[i]];
            },
            Number<NDimY>{});

        // distributed_spans_lengthss_[ndim_span_major_][max_ndim_span_minor_]
        static constexpr auto distributed_spans_lengthss_ = [] {
            Array<Array<index_t, max_ndim_span_minor_>, ndim_span_major_>
                distributed_spans_lengthss{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t rh_major = ys_to_rhs_major_[i];
                const index_t rh_minor = ys_to_rhs_minor_[i];

                const index_t h_length = hs_lengthss_[Number<rh_major - 1>{}][rh_minor];

                const index_t span_major = rh_major - 1;
                const index_t span_minor = rhs_major_minor_to_span_minor_[rh_major][rh_minor];

                distributed_spans_lengthss(span_major)(span_minor) = h_length;
            });

            return distributed_spans_lengthss;
        }();

        // ndims_distributed_spans_minor_[ndim_span_major_]
        static constexpr auto ndims_distributed_spans_minor_ = [] {
            Array<index_t, ndim_span_major_> ndims_distributed_spans_minor{0};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t span_major = ys_to_rhs_major_[i] - 1;

                ndims_distributed_spans_minor(span_major)++;
            });

            return ndims_distributed_spans_minor;
        }();

        // does_p_own_r_[NDimP][NDimR]
        static constexpr auto does_p_own_r_ = [] {
            if constexpr(NDimR > 0)
            {
                Array<Array<bool, NDimR>, NDimP> does_p_own_r{{false}};

                static_for<0, NDimP, 1>{}([&](auto idim_p) {
                    constexpr index_t ndim_low = ps_to_rhss_major_[idim_p].Size();

                    static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                        constexpr index_t rh_major = ps_to_rhss_major_[idim_p][idim_low];
                        constexpr index_t rh_minor = ps_to_rhss_minor_[idim_p][idim_low];

                        if constexpr(rh_major == 0)
                        {
                            does_p_own_r(idim_p)(rh_minor) = true;
                        }
                    });
                });

                return does_p_own_r;
            }
            else
            {
                return Array<Array<bool, NDimR>, NDimP>{};
            }
        }();

        // ps_over_rs_derivative_[NDimP][NDimR]
        static constexpr auto ps_over_rs_derivative_ = [] {
            if constexpr(NDimR > 0)
            {
                Array<Array<index_t, NDimR>, NDimP> ps_over_rs_derivative{{0}};

                static_for<0, NDimP, 1>{}([&](auto idim_p) {
                    constexpr index_t ndim_low = ps_to_rhss_major_[idim_p].Size();

                    index_t p_over_rh_derivative = 1;

                    static_for<ndim_low - 1, -1, -1>{}([&](auto idim_low) {
                        constexpr index_t rh_major = ps_to_rhss_major_[idim_p][idim_low];
                        constexpr index_t rh_minor = ps_to_rhss_minor_[idim_p][idim_low];

                        constexpr index_t rh_length = rhs_lengthss_[rh_major][rh_minor];

                        if constexpr(rh_major == 0)
                        {
                            ps_over_rs_derivative(idim_p)(rh_minor) = p_over_rh_derivative;
                        }

                        p_over_rh_derivative *= rh_length;
                    });
                });

                return ps_over_rs_derivative;
            }
            else
            {
                return Array<Array<index_t, NDimR>, NDimP>{};
            }
        }();

        // e.g. tuple<seq<1, 4, 32>, seq<4, 1, 4, 2, 4>> --> seq<3, 5> --> seq<0, 3, 8>
        __host__ __device__ static constexpr auto GetHDimLengthsPrefixSum()
        {
            // <len_d0, len_d1, ...>
            // e.g. tuple<seq<1, 4, 32>, seq<4, 1, 4, 2, 4>> --> seq<3, 5>
            constexpr auto uniformed_h_dim_lengths = generate_sequence_v2(
                [&](auto i) {
                    constexpr index_t size = HsLengthss{}[i].Size();
                    return Number<size>{};
                },
                Number<NDimX>{});

            // <0, len_d0, len_d0+len_d1, ...>
            // e.g. seq<3, 5> --> seq<0, 3, 8>
            constexpr auto h_dim_prefix_sum = prefix_sum_sequence(uniformed_h_dim_lengths);

            return h_dim_prefix_sum;
        }

        __host__ __device__ static constexpr auto GetUniformedIdxY2H()
        {
            constexpr auto all_ys_2_rhss = transform_sequences(
                [](auto major, auto minor) constexpr {
                    // <0, 0, len_d0, len_d0+len_d1, ...>
                    constexpr auto x_dim_prefix_sum =
                        merge_sequences(Sequence<0>{} /*for R dims*/, GetHDimLengthsPrefixSum());
                    return x_dim_prefix_sum.At(major) + minor;
                },
                Ys2RHsMajor{},
                Ys2RHsMinor{});

            return all_ys_2_rhss;
        }

        // return tuple<sorted_dims, sorted_maps, sorted_prefix_sum>
        template <typename IdxSeq, typename PrefixSumSeq>
        __host__ __device__ static constexpr auto GetSortedInfo(IdxSeq, PrefixSumSeq)
        {
            using sorted_idx =
                sequence_unique_sort<IdxSeq, math::less<index_t>, math::equal<index_t>>;

            constexpr auto sorted_dims = typename sorted_idx::type{};
            constexpr auto sorted_maps = typename sorted_idx::sorted2unsorted_map{};

            constexpr auto sorted_histogram =
                histogram_sorted_sequence(sorted_dims, PrefixSumSeq{});
            constexpr auto sorted_prefix_sum = prefix_sum_sequence(sorted_histogram);

            return make_tuple(sorted_dims, sorted_maps, sorted_prefix_sum);
        }

        __host__ __device__ static constexpr auto GetSortedYInfo()
        {
            return GetSortedInfo(GetUniformedIdxY2H(), GetHDimLengthsPrefixSum());
        }

        __host__ __device__ void Print() const
        {
            printf("StaticTileDistributionEncoding::Detail{");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("ndim_span_major_: ");
            print(ndim_span_major_);
            printf(", ");
            //
            printf("ndims_rhs_minor_: ");
            print(ndims_rhs_minor_);
            printf(", ");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("max_ndim_rh_minor_: ");
            print(max_ndim_rh_minor_);
            printf(", ");
            //
            printf("rhs_lengthss_: ");
            print(rhs_lengthss_);
            printf(", ");
            //
            printf("ys_lengths_: ");
            print(ys_lengths_);
            printf(", ");
            //
            printf("rhs_major_minor_to_ys_: ");
            print(rhs_major_minor_to_ys_);
            printf(", ");
            //
            printf("ndims_span_minor_: ");
            print(ndims_span_minor_);
            printf(", ");
            //
            printf("max_ndim_span_minor_: ");
            print(max_ndim_span_minor_);
            printf(", ");
            //
            printf("ys_to_span_major_: ");
            print(ys_to_span_major_);
            printf(", ");
            //
            printf("ys_to_span_minor_: ");
            print(ys_to_span_minor_);
            printf(", ");
            //
            printf("distributed_spans_lengthss_: ");
            print(distributed_spans_lengthss_);
            printf(", ");
            //
            printf("ndims_distributed_spans_minor_: ");
            print(ndims_distributed_spans_minor_);
            printf(", ");
            //
            printf("ps_over_rs_derivative_: ");
            print(ps_over_rs_derivative_);
            //
            printf("}");
        }
    };

    __host__ __device__ void Print() const
    {
        printf("StaticTileDistributionEncoding{");
        //
        printf("NDimX: %d, NDimP: %d, NDimY: %d, ", NDimX, NDimP, NDimY);
        //
        printf("rs_lengths_: ");
        print(rs_lengths_);
        printf(", ");
        //
        printf("hs_lengthss_: ");
        print(hs_lengthss_);
        printf(", ");
        //
        printf("ps_to_rhss_major_: ");
        print(ps_to_rhss_major_);
        printf(", ");
        //
        printf("ps_to_rhss_minor_: ");
        print(ps_to_rhss_minor_);
        printf(", ");
        //
        printf("ys_to_rhs_major_: ");
        print(ys_to_rhs_major_);
        printf(", ");
        //
        printf("ys_to_rhs_minor_: ");
        print(ys_to_rhs_minor_);
        printf(", ");
        //
        printf("Detail: ");
        print(Detail{});
        //
        printf("}");
    }
};

} // namespace tile_program
} // namespace ck
