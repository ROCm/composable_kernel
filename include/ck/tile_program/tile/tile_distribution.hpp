// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/macro_func_tensor_adaptor_from_encoding.hpp"

#include "ck/tile_program/tile/static_tile_distribution_encoding.hpp"

namespace ck {
namespace tile_program {

// distributed span
template <index_t... PartialHsLengths>
struct TileDistributedSpan
{
    using Impl = Sequence<PartialHsLengths...>;

    static constexpr auto impl_ = Impl{};

    __host__ __device__ static constexpr bool IsStatic() { return true; }
};

// distributed index
template <index_t... PartialHsIndices>
struct TileDistributedIndex
{
    using Impl = Sequence<PartialHsIndices...>;

    static constexpr auto impl_ = Impl{};

    __host__ __device__ static constexpr bool IsStatic() { return true; }
};

namespace detail {

template <index_t... Is>
__host__ __device__ constexpr auto make_tile_distributed_span(Sequence<Is...>)
{
    return TileDistributedSpan<Is...>{};
}

template <index_t... Is>
__host__ __device__ constexpr auto make_tile_distributed_index(Sequence<Is...>)
{
    return TileDistributedIndex<Is...>{};
}

} // namespace detail

template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_,
          typename TileDistributionDetail_> // FIXME: this is for hold ad-hoc but useful info,
                                            // should be more elegnat
struct TileDistribution
{
    using PsYs2XsAdaptor = remove_cvref_t<PsYs2XsAdaptor_>;
    using Ys2DDescriptor = remove_cvref_t<Ys2DDescriptor_>;
    using DstrEncode     = remove_cvref_t<StaticTileDistributionEncoding_>;
    using DstrDetail     = remove_cvref_t<TileDistributionDetail_>;

    static_assert(PsYs2XsAdaptor::IsStatic() && Ys2DDescriptor::IsStatic(),
                  "wrong! should be static");

    static constexpr index_t NDimX = PsYs2XsAdaptor::GetNumOfBottomDimension();
    static constexpr index_t NDimY = Ys2DDescriptor::GetNumOfTopDimension();
    static constexpr index_t NDimP = PsYs2XsAdaptor::GetNumOfTopDimension() - NDimY;
    static constexpr index_t NDimR = StaticTileDistributionEncoding_::NDimR;

    PsYs2XsAdaptor ps_ys_to_xs_;
    Ys2DDescriptor ys_to_d_;

    __host__ __device__ static constexpr index_t GetNumOfDimensionX() { return NDimX; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionY() { return NDimY; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionP() { return NDimP; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionR() { return NDimR; }

    __host__ __device__ static constexpr auto GetLengths()
    {
#if 0
        // FIXME: TensorAdaptor::GetBottomDimensionLengths is wrong. re-enable this after it's fixed
        ps_ys_to_xs_.GetBottomDimensionLengths();
#else
        return generate_tuple(
            [&](auto i) {
                constexpr index_t x_length =
                    container_reduce(typename DstrEncode::HsLengthss{}[i], math::multiplies{}, 1);

                return Number<x_length>{};
            },
            Number<NDimX>{});
#endif
    }

    __host__ __device__ constexpr const auto& GetPsYs2XsAdaptor() const { return ps_ys_to_xs_; }

    __host__ __device__ constexpr const auto& GetYs2DDescriptor() const { return ys_to_d_; }

    __host__ __device__ static constexpr auto GetStaticTileDistributionEncoding()
    {
        return DstrEncode{};
    }

#if 1
    // Calculate Replication index [R0, R1, ...] based on Partion index
    // FIXME: very nasty implementation
    template <typename PartitionIndex>
    __host__ __device__ auto CalculateRsIndexFromPsIndex(const PartitionIndex& ps_idx) const
    {
        static_assert(PartitionIndex::Size() == NDimP, "wrong!");

        const auto ps_ys_idx = container_concat(ps_idx, Array<index_t, NDimY>{0});

        const auto dummy_adaptor_coord = make_tensor_adaptor_coordinate(ps_ys_to_xs_, ps_ys_idx);

        Array<index_t, NDimR> rs_idx;

        static_for<0, NDimP, 1>{}([&](auto idim_p) {
            constexpr index_t ndim_low = DstrEncode::ps_to_rhss_major_[idim_p].Size();

            static_for<0, ndim_low, 1>{}([&](auto i) {
                constexpr index_t rh_major = DstrEncode::ps_to_rhss_major_[idim_p][i];
                constexpr index_t rh_minor = DstrEncode::ps_to_rhss_minor_[idim_p][i];

                // 0-th rh_major is the replicate dimension
                if constexpr(rh_major == 0)
                {
                    constexpr index_t adaptor_hidden_id =
                        DstrDetail::rh_major_minor_to_adaptor_hidden_idss_[rh_major][rh_minor];

                    // fill in
                    rs_idx(rh_minor) = dummy_adaptor_coord.GetHiddenIndex()[adaptor_hidden_id];
                }
            });
        });

        return rs_idx;
    }
#endif

    __host__ __device__ static constexpr auto GetDistributedSpans()
    {
        constexpr auto distributed_spans_impl = DstrEncode::Detail::distributed_spans_lengthss_;
        constexpr auto ndims_spans_minor      = DstrEncode::Detail::ndims_distributed_spans_minor_;

        return generate_tuple(
            [&](auto i) {
                constexpr auto span_impl          = distributed_spans_impl[i];
                constexpr index_t ndim_span_minor = ndims_spans_minor[i];

                constexpr auto span = TO_SEQUENCE(span_impl, ndim_span_minor);

                return detail::make_tile_distributed_span(span);
            },
            Number<NDimX>{});
    }

    // FIXME: it's hacky to get Y index from Distributed-Index
    template <typename DistributedIndices>
    __host__ __device__ static constexpr auto GetYIndicesFromDistributedIndices(DistributedIndices)
    {
        constexpr auto ys_idx_arr = [] {
            Array<index_t, NDimY> ys_idx;

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t span_major = DstrEncode::Detail::ys_to_span_major_[i];
                constexpr index_t span_minor = DstrEncode::Detail::ys_to_span_minor_[i];

                constexpr auto dstr_index = DistributedIndices{}[Number<span_major>{}];

                ys_idx(i) = dstr_index.impl_[span_minor];
            });

            return ys_idx;
        }();

        constexpr index_t ndim_y = NDimY;

        return TO_SEQUENCE(ys_idx_arr, ndim_y);
    }

    __host__ __device__ static constexpr bool IsStatic()
    {
        return PsYs2XsAdaptor::IsStatic() && Ys2DDescriptor::IsStatic();
    }

    __host__ __device__ void Print() const
    {
        printf("TileDistribution{");
        //
        printf("StaticTileDistributionEncoding: ");
        print(DstrEncode{});
        printf(", ");
        //
        printf("ps_ys_to_xs_: ");
        print(ps_ys_to_xs_);
        printf(", ");
        //
        printf("ys_to_d_: ");
        print(ys_to_d_);
        //
        printf("}");
    }
};

namespace detail {

template <index_t NDimMax>
__host__ __device__ constexpr auto make_sequential_index(index_t ibegin, index_t iend)
{
    Array<index_t, NDimMax> arr{0};

    for(index_t i = 0; i < iend - ibegin; ++i)
    {
        arr(i) = ibegin + i;
    }

    return arr;
}

// this returns a constexpr encoding of TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto
    make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_)
{
    using RsLengths    = typename StaticTileDistributionEncoding_::RsLengths;
    using HsLengthss   = typename StaticTileDistributionEncoding_::HsLengthss;
    using Ps2RHssMajor = typename StaticTileDistributionEncoding_::Ps2RHssMajor;
    using Ps2RHssMinor = typename StaticTileDistributionEncoding_::Ps2RHssMinor;
    using Ys2RHsMajor  = typename StaticTileDistributionEncoding_::Ys2RHsMajor;
    using Ys2RHsMinor  = typename StaticTileDistributionEncoding_::Ys2RHsMinor;

    // FIXME: increase max value if fail
    constexpr index_t kMaxNumTransforms = 20;
    constexpr index_t kMaxMetaDataSize  = 128;
    constexpr index_t kMaxNumDim        = 10;

    using Name     = IndexTransformEnum;
    using MetaData = MetaDataBuffer<kMaxMetaDataSize>;
    using NumDim   = index_t;
    using Dims     = Array<index_t, kMaxNumDim>;
    using Lengths  = Array<index_t, kMaxNumDim>;

    // Tile Adaptor
    //   bottom dims [x0, x1, x2, ...]
    //   top dims [p0, p1, ..., y0, y1, ...]
    constexpr index_t ndim_x = HsLengthss::Size();

    // Dim Ids: [idim_x_major, idim_x_minor] to [idim_hidden]
    Array<Array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_minor_to_hidden_ids;
    Array<Array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_minor_to_hidden_lengths;

    auto trans = Array<Tuple<Name, MetaData, NumDim, Dims, NumDim, Dims>, kMaxNumTransforms>{};

    index_t num_tran       = 0;
    index_t hidden_dim_cnt = ndim_x;

    // this is Replicate transform
    {
        constexpr index_t ndim_r_minor = RsLengths::Size();

        constexpr auto r_minor_lengths = RsLengths{};

        trans(num_tran++) = {
            IndexTransformEnum::Replicate,
            MetaData{to_array<index_t, ndim_r_minor>(r_minor_lengths)},
            NumDim{0},
            Dims{},
            NumDim{ndim_r_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_r_minor)};

        for(index_t i = 0; i < ndim_r_minor; ++i)
        {
            rh_major_minor_to_hidden_ids(0)(i)     = hidden_dim_cnt;
            rh_major_minor_to_hidden_lengths(0)(i) = r_minor_lengths[i];

            hidden_dim_cnt++;
        }
    };

    // these are Unmerge transforms for X dimesions
    static_for<0, ndim_x, 1>{}([&trans,
                                &num_tran,
                                &hidden_dim_cnt,
                                &rh_major_minor_to_hidden_ids,
                                &rh_major_minor_to_hidden_lengths](auto idim_x) {
        constexpr auto h_minor_lengths = tuple_element_t<idim_x, HsLengthss>{};

        constexpr index_t ndim_h_minor = h_minor_lengths.Size();

        trans(num_tran++) = {
            IndexTransformEnum::UnMerge,
            MetaData{to_array<index_t, ndim_h_minor>(h_minor_lengths)},
            NumDim{1},
            Dims{idim_x},
            NumDim{ndim_h_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_h_minor)};

        for(index_t i = 0; i < ndim_h_minor; ++i)
        {
            rh_major_minor_to_hidden_ids(idim_x + 1)(i)     = hidden_dim_cnt;
            rh_major_minor_to_hidden_lengths(idim_x + 1)(i) = h_minor_lengths[i];

            hidden_dim_cnt++;
        }
    });

    // transform: P dimensions
    constexpr index_t ndim_p = Ps2RHssMajor::Size();

    Dims hidden_dim_id_ps;

    static_for<0, ndim_p, 1>{}([&](auto iDimP) {
        //
        index_t hidden_dim_id_p = hidden_dim_cnt++;

        hidden_dim_id_ps(iDimP) = hidden_dim_id_p;

        constexpr auto p2RHsMajor = Ps2RHssMajor{}[iDimP];
        constexpr auto p2RHsMinor = Ps2RHssMinor{}[iDimP];

        static_assert(p2RHsMajor.Size() == p2RHsMinor.Size(), "wrong!");

        constexpr index_t ndim_low = p2RHsMajor.Size();

        Dims low_dims;
        Lengths low_lengths;

        for(index_t i = 0; i < ndim_low; ++i)
        {
            index_t rh_major = p2RHsMajor[i];
            index_t rh_minor = p2RHsMinor[i];
            low_dims(i)      = rh_major_minor_to_hidden_ids[rh_major][rh_minor];
            low_lengths(i)   = rh_major_minor_to_hidden_lengths[rh_major][rh_minor];
        }

        trans(num_tran++) = {IndexTransformEnum::Merge,
                             MetaData{to_array<index_t, ndim_low>(low_lengths)},
                             NumDim{ndim_low},
                             low_dims,
                             NumDim{1},
                             Dims{hidden_dim_id_p}};
    });

    constexpr index_t ndim_bottom = ndim_x;

    constexpr auto bottom_dim_ids = make_sequential_index<kMaxNumDim>(0, ndim_bottom);

    constexpr auto ys_to_rhs_major = Ys2RHsMajor{};
    constexpr auto ys_to_rhs_minor = Ys2RHsMinor{};

    constexpr index_t ndim_y   = Ys2RHsMajor::Size();
    constexpr index_t ndim_top = ndim_p + ndim_y;

    auto top_dim_ids = hidden_dim_id_ps;

    {
        for(index_t i = 0; i < ndim_y; ++i)
        {
            index_t rh_major        = ys_to_rhs_major[i];
            index_t rh_minor        = ys_to_rhs_minor[i];
            top_dim_ids(ndim_p + i) = rh_major_minor_to_hidden_ids[rh_major][rh_minor];
        }
    }

    //
    const auto ps_ys_to_xs_adaptor_encoding =
        make_tuple(trans, num_tran, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top);

    // descriptor: [y0, y1, ...] to [d]
    Lengths y_lengths;
    index_t d_length = 1;

    for(index_t i = 0; i < ndim_y; ++i)
    {
        index_t rh_major = ys_to_rhs_major[i];
        index_t rh_minor = ys_to_rhs_minor[i];
        index_t y_length = rh_major_minor_to_hidden_lengths[rh_major][rh_minor];
        y_lengths(i)     = y_length;
        d_length *= y_length;
    }

    auto tran = make_tuple(IndexTransformEnum::UnMerge,
                           MetaData{to_array<index_t, ndim_y>(y_lengths)},
                           NumDim{1},
                           Dims{0},
                           NumDim{ndim_y},
                           make_sequential_index<kMaxNumDim>(1, ndim_y + 1));

    const auto ys_to_d_adaptor_encoding = make_tuple(
        make_tuple(tran), 1, Dims{0}, 1, make_sequential_index<kMaxNumDim>(1, ndim_y + 1), ndim_y);

    return make_tuple(ps_ys_to_xs_adaptor_encoding,
                      ys_to_d_adaptor_encoding,
                      d_length,
                      rh_major_minor_to_hidden_ids);
}

// FIXME: this is nasty. Move it inside TileDistributionEncoding::Detail
template <typename RhMajorMinor2AdaptorHiddenIdss> // Tuple<Sequence<...>, ...>
struct TileDistributionDetail
{
    static constexpr auto rh_major_minor_to_adaptor_hidden_idss_ =
        to_array_of_array(RhMajorMinor2AdaptorHiddenIdss{});
};

} // namespace detail

// this returns a constexpr TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto make_tile_distribution(StaticTileDistributionEncoding_)
{
    using DstrEncode = remove_cvref_t<StaticTileDistributionEncoding_>;

    constexpr auto adaptor_impl =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto ps_ys_to_xs_adaptor_impl          = adaptor_impl.template At<0>();
    constexpr auto ys_to_d_adaptor_impl              = adaptor_impl.template At<1>();
    constexpr index_t d_length                       = adaptor_impl.template At<2>();
    constexpr auto rh_major_minor_to_hidden_ids_impl = adaptor_impl.template At<3>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(ps_ys_to_xs_adaptor_impl);

    constexpr auto ys_to_d_adaptor = CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(ys_to_d_adaptor_impl);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, d_length);

    //
    constexpr index_t ndim_rh_major = DstrEncode::Detail::ndim_rh_major_;
    constexpr auto ndims_rhs_minor  = DstrEncode::Detail::ndims_rhs_minor_;

    constexpr auto rh_major_minor_to_hidden_ids =
        TO_TUPLE_OF_SEQUENCE(rh_major_minor_to_hidden_ids_impl, ndim_rh_major, ndims_rhs_minor);

    return TileDistribution<
        remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
        remove_cvref_t<decltype(ys_to_d_descriptor)>,
        remove_cvref_t<DstrEncode>,
        detail::TileDistributionDetail<remove_cvref_t<decltype(rh_major_minor_to_hidden_ids)>>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}

// this returns a static TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto make_static_tile_distribution(StaticTileDistributionEncoding_)
{
    using DstrEncode = remove_cvref_t<StaticTileDistributionEncoding_>;

    constexpr auto adaptor_impl =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto ps_ys_to_xs_adaptor_impl          = adaptor_impl.template At<0>();
    constexpr auto ys_to_d_adaptor_impl              = adaptor_impl.template At<1>();
    constexpr index_t d_length                       = adaptor_impl.template At<2>();
    constexpr auto rh_major_minor_to_hidden_ids_impl = adaptor_impl.template At<3>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(ps_ys_to_xs_adaptor_impl);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(ys_to_d_adaptor_impl);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, Number<d_length>{});

    //
    constexpr index_t ndim_rh_major = DstrEncode::Detail::ndim_rh_major_;
    constexpr auto ndims_rhs_minor  = DstrEncode::Detail::ndims_rhs_minor_;

    constexpr auto rh_major_minor_to_hidden_ids =
        TO_TUPLE_OF_SEQUENCE(rh_major_minor_to_hidden_ids_impl, ndim_rh_major, ndims_rhs_minor);

    return TileDistribution<
        remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
        remove_cvref_t<decltype(ys_to_d_descriptor)>,
        remove_cvref_t<DstrEncode>,
        detail::TileDistributionDetail<remove_cvref_t<decltype(rh_major_minor_to_hidden_ids)>>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}

} // namespace tile_program
} // namespace ck
