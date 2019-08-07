#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "tensor_coordinate.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1 0
#endif

namespace ck {

// user need to make sure alignment requirement is satisfied when setting DataPerAccesss > 1
template <class Float,
          class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class DimAccessOrder,
          index_t DataPerAccess>
__device__ void threadwise_generic_tensor_slice_copy_v1(
    SrcDesc,
    const Float* __restrict__ p_src,
    Array<index_t, SrcDesc::GetNumOfDimension()> src_multi_id_begin,
    DstDesc,
    Float* __restrict__ p_dst,
    Array<index_t, DstDesc::GetNumOfDimension()> dst_multi_id_begin,
    SliceLengths,
    DimAccessOrder,
    Number<DataPerAccess>)
{
    constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    static_assert(nDim == SrcDesc::GetNumOfDimension() && nDim == DstDesc::GetNumOfDimension() &&
                      nDim == SliceLengths::GetSize() && nDim == DimAccessOrder::GetSize(),
                  "wrong! # of dimensions not the same");

    static_assert(is_valid_sequence_map<DimAccessOrder>::value, "wrong! map is not valid");

    // TODO: do more sanity-check here, something like:
    // constexpr auto src_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // constexpr auto dst_strides_in_access_order =
    //     SrcDesc::ReorderGivenNew2Old(DimAccessOrder{}).GetStride(Number<nDim-1>{});

    // // check src/dst stride on the lowest access dimension
    // static_assert((DataPerAccess == 1 || src_strides_in_access_order.Back() == 1) &&
    //                   (DataPerAccess == 1 || dst_strides_in_access_order.Back() == 1),
    //               "wrong! src/dst stride on the lowest access dimension needs to be 1 for "
    //               "vectorized read/write");

    constexpr auto slice_lengths_in_access_order =
        SliceLengths::ReorderGivenNew2Old(DimAccessOrder{});

    // check slice length on the lowest access dimension
    static_assert(slice_lengths_in_access_order.Back() % DataPerAccess == 0,
                  "wrong! slice length on the lowest access dimension should be evenly divided by "
                  "DataPerAccess");

    constexpr index_t num_access_on_lowest_access_dimension =
        slice_lengths_in_access_order.Back() / DataPerAccess;

    constexpr auto access_lengths = slice_lengths_in_access_order.Modify(
        Number<nDim - 1>{}, Number<num_access_on_lowest_access_dimension>{});

    using vector_t = typename vector_type<Float, DataPerAccess>::MemoryType;

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1
    static_ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        constexpr index_t itmp = access_multi_id.Back() * DataPerAccess;

        constexpr auto data_multi_id_in_access_order =
            access_multi_id.Modify(Number<nDim - 1>{}, Number<itmp>{});

        constexpr auto data_multi_id =
            data_multi_id_in_access_order.ReorderGivenOld2New(DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
    });
#else
    ford<decltype(access_lengths)>{}([&](auto access_multi_id) {
        auto data_multi_id_in_access_order      = access_multi_id;
        data_multi_id_in_access_order(nDim - 1) = access_multi_id[nDim - 1] * DataPerAccess;

        const auto data_multi_id =
            reorder_array_given_old2new(data_multi_id_in_access_order, DimAccessOrder{});

        const index_t src_index =
            SrcDesc::GetOffsetFromMultiIndex(src_multi_id_begin + data_multi_id);

        const index_t dst_index =
            DstDesc::GetOffsetFromMultiIndex(dst_multi_id_begin + data_multi_id);

        *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_index]);
    });
#endif
}

// This threadwise copy allow vector access of src and dst.
// It allows the dimensions of vector access to be different on src and dst.
// It also allows the vector size to be different on src and dst.
// It also allows order of access to be different on src and dst.
// It use register as buffer to hold all data moving from src to dst.
// It is designed for copying small amount of data, and src and dst are
// device memory or LDS.
// When copying large amout of data, let's hope compiler will reduce register
// used for the buffer.
template <class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class SrcDimAccessOrder,
          class DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct ThreadwiseGenericTensorSliceCopy_v1r1
{
    static constexpr index_t nDim = SliceLengths::GetSize();

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v1r1(
        Array<index_t, nDim> src_slice_origin, Array<index_t, nDim> dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::GetSize() &&
                          nDim == SrcDimAccessOrder::GetSize() &&
                          nDim == DstDimAccessOrder::GetSize(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDimAccessOrder>::value &&
                          is_valid_sequence_map<DstDimAccessOrder>::value,
                      "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcVectorAccessDim] % SrcDataPerAccess == 0 &&
                          SliceLengths{}[DstVectorAccessDim] % DstDataPerAccess == 0,
                      "wrong! cannot evenly divide");

        // check vectorized memory access
        constexpr auto src_vector_access_dim = Number<SrcVectorAccessDim>{};
        constexpr auto dst_vector_access_dim = Number<DstVectorAccessDim>{};

        static_if<!SrcDesc::ContainMultipleOriginalDimensions(src_vector_access_dim)>{}(
            [&](auto fwd) {
                static_assert(
                    (fwd(SrcDesc{}).GetStride(src_vector_access_dim) == 1 || SrcDataPerAccess == 1),
                    "wrong! vectorized access is allowed only if stride == 1");
            })
            .Else([&](auto fwd) {
                static_assert(
                    (fwd(SrcDesc{}).GetLastOriginalDimensionStride(src_vector_access_dim) == 1 ||
                     SrcDataPerAccess == 1),
                    "wrong! vectorized access is allowed only if stride == 1");
            });

        static_if<!DstDesc::ContainMultipleOriginalDimensions(dst_vector_access_dim)>{}(
            [&](auto fwd) {
                static_assert(
                    (fwd(DstDesc{}).GetStride(dst_vector_access_dim) == 1 || DstDataPerAccess == 1),
                    "wrong! vectorized access is allowed only if stride == 1");
            })
            .Else([&](auto fwd) {
                static_assert(
                    (fwd(DstDesc{}).GetLastOriginalDimensionStride(dst_vector_access_dim) == 1 ||
                     DstDataPerAccess == 1),
                    "wrong! vectorized access is allowed only if stride == 1");
            });
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v1r1()
        : ThreadwiseGenericTensorSliceCopy_v1r1(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(Array<index_t, nDim> src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(Array<index_t, nDim> dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <class TData>
    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        constexpr auto buffer_desc = make_ConstantTensorDescriptor_packed(SliceLengths{});

        TData p_buffer_[buffer_desc.GetElementSpace()];
        TData* p_buffer = p_buffer_;

        // copy data from src into buffer
        {
            using vector_t = typename vector_type<TData, SrcDataPerAccess>::MemoryType;

            constexpr auto src_vector_access_dim = Number<SrcVectorAccessDim>{};
            constexpr auto src_data_per_access   = Number<SrcDataPerAccess>{};

            constexpr auto src_access_lengths = SliceLengths::Modify(
                src_vector_access_dim,
                SliceLengths::Get(src_vector_access_dim) / src_data_per_access);

            static_ford<decltype(src_access_lengths), SrcDimAccessOrder>{}([&](auto src_access_id) {
                constexpr auto src_data_begin_id = src_access_id.Modify(
                    src_vector_access_dim,
                    src_access_id[src_vector_access_dim] * src_data_per_access);

                const index_t src_offset =
                    SrcDesc::GetOffsetFromMultiIndex(mSrcSliceOrigin + src_data_begin_id);

                // load vector from src
                const vector_t vector_data = *reinterpret_cast<const vector_t*>(&p_src[src_offset]);

                // unpack vector into buffer
                static_for<0, SrcDataPerAccess, 1>{}([&](auto i) {
                    constexpr auto scalar_id =
                        typename uniform_sequence_gen<nDim, 0>::type{}.Modify(src_vector_access_dim,
                                                                              i);

                    constexpr index_t buffer_offset =
                        buffer_desc.GetOffsetFromMultiIndex(src_data_begin_id + scalar_id);

                    p_buffer[buffer_offset] = reinterpret_cast<const TData*>(&vector_data)[i];
                });
            });
        }

        // copy data from buffer to dst
        {
            using vector_t = typename vector_type<TData, DstDataPerAccess>::MemoryType;

            constexpr auto dst_vector_access_dim = Number<DstVectorAccessDim>{};
            constexpr auto dst_data_per_access   = Number<DstDataPerAccess>{};

            constexpr auto dst_access_lengths = SliceLengths::Modify(
                dst_vector_access_dim,
                SliceLengths::Get(dst_vector_access_dim) / dst_data_per_access);

            static_ford<decltype(dst_access_lengths), DstDimAccessOrder>{}([&](auto dst_access_id) {
                constexpr auto dst_data_begin_id = dst_access_id.Modify(
                    dst_vector_access_dim,
                    dst_access_id[dst_vector_access_dim] * dst_data_per_access);

                vector_t vector_data;

                // pack vector from buffer
                static_for<0, DstDataPerAccess, 1>{}([&](auto i) {
                    constexpr auto scalar_id =
                        typename uniform_sequence_gen<nDim, 0>::type{}.Modify(dst_vector_access_dim,
                                                                              i);

                    constexpr index_t buffer_offset =
                        buffer_desc.GetOffsetFromMultiIndex(dst_data_begin_id + scalar_id);

                    reinterpret_cast<TData*>(&vector_data)[i] = p_buffer[buffer_offset];
                });

                const index_t dst_offset =
                    DstDesc::GetOffsetFromMultiIndex(mDstSliceOrigin + dst_data_begin_id);

                // store vector into dst
                *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) = vector_data;
            });
        }
    }

    private:
    Array<index_t, nDim> mSrcSliceOrigin;
    Array<index_t, nDim> mDstSliceOrigin;
};

// This threadwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimensions of vector access should be the same on src and dst.
// The dimension access order should be the same on src and dst.
// It is designed for cases, where one of src and dst is register, and
// the other is device memory or LDS
template <class SrcDesc,
          class DstDesc,
          class SliceLengths,
          class DimAccessOrder,
          index_t VectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct ThreadwiseGenericTensorSliceCopy_v1r2
{
    static constexpr index_t nDim = SliceLengths::GetSize();

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v1r2(
        Array<index_t, nDim> src_slice_origin, Array<index_t, nDim> dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::GetSize() &&
                          nDim == DimAccessOrder::GetSize(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<DimAccessOrder>::value, "wrong! map is not valid");

        static_assert(
            SliceLengths{}[VectorAccessDim] % math::lcm(SrcDataPerAccess, DstDataPerAccess) == 0,
            "wrong! cannot evenly divide");

        // check vectorized memory access
        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        static_if<!SrcDesc::ContainMultipleOriginalDimensions(vector_access_dim)>{}([&](auto fwd) {
            static_assert(
                (fwd(SrcDesc{}).GetStride(vector_access_dim) == 1 || SrcDataPerAccess == 1),
                "wrong! vectorized access is allowed only if stride == 1");
        }).Else([&](auto fwd) {
            static_assert((fwd(SrcDesc{}).GetLastOriginalDimensionStride(vector_access_dim) == 1 ||
                           SrcDataPerAccess == 1),
                          "wrong! vectorized access is allowed only if stride == 1");
        });

        static_if<!DstDesc::ContainMultipleOriginalDimensions(vector_access_dim)>{}([&](auto fwd) {
            static_assert(
                (fwd(DstDesc{}).GetStride(vector_access_dim) == 1 || DstDataPerAccess == 1),
                "wrong! vectorized access is allowed only if stride == 1");
        }).Else([&](auto fwd) {
            static_assert((fwd(DstDesc{}).GetLastOriginalDimensionStride(vector_access_dim) == 1 ||
                           DstDataPerAccess == 1),
                          "wrong! vectorized access is allowed only if stride == 1");
        });
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v1r2()
        : ThreadwiseGenericTensorSliceCopy_v1r2(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(Array<index_t, nDim> src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(Array<index_t, nDim> dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <class TData>
    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        using src_vector_t = typename vector_type<TData, SrcDataPerAccess>::MemoryType;
        using dst_vector_t = typename vector_type<TData, DstDataPerAccess>::MemoryType;

        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerAccess>{};
        constexpr auto dst_data_per_access = Number<DstDataPerAccess>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerAccess, DstDataPerAccess)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        static_ford<decltype(long_vector_access_lengths), DimAccessOrder>{}([&](
            auto long_vector_access_id) {
            // data id w.r.t slicing-window
            constexpr auto long_vector_data_begin_id = long_vector_access_id.Modify(
                vector_access_dim, long_vector_access_id[vector_access_dim] * long_vector_size);

            // buffer to hold a long-vector
            TData p_long_vector[long_vector_size];

            // load data from src to the long-vector buffer
            static_for<0, long_vector_size / src_data_per_access, 1>{}([&](auto i) {
                constexpr auto scalar_id = typename uniform_sequence_gen<nDim, 0>::type{}.Modify(
                    vector_access_dim, i * src_data_per_access);

                const index_t src_offset = SrcDesc::GetOffsetFromMultiIndex(
                    mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id));

                constexpr index_t buffer_offset = i * src_data_per_access;

                *reinterpret_cast<src_vector_t*>(&p_long_vector[buffer_offset]) =
                    *reinterpret_cast<const src_vector_t*>(&p_src[src_offset]);
            });

            // store data from the long-vector buffer to dst
            static_for<0, long_vector_size / dst_data_per_access, 1>{}([&](auto i) {
                constexpr auto scalar_id = typename uniform_sequence_gen<nDim, 0>::type{}.Modify(
                    vector_access_dim, i * dst_data_per_access);

                constexpr index_t buffer_offset = i * dst_data_per_access;

                const index_t dst_offset = DstDesc::GetOffsetFromMultiIndex(
                    mDstSliceOrigin + (long_vector_data_begin_id + scalar_id));

                *reinterpret_cast<dst_vector_t*>(&p_dst[dst_offset]) =
                    *reinterpret_cast<dst_vector_t*>(&p_long_vector[buffer_offset]);
            });
        });
    }

    template <class TData>
    __device__ void Run_non_static(const TData* p_src, TData* p_dst) const
    {
        using src_vector_t = typename vector_type<TData, SrcDataPerAccess>::MemoryType;
        using dst_vector_t = typename vector_type<TData, DstDataPerAccess>::MemoryType;

        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerAccess>{};
        constexpr auto dst_data_per_access = Number<DstDataPerAccess>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerAccess, DstDataPerAccess)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), DimAccessOrder>{}(
            [&](auto long_vector_access_id) {
                // data id w.r.t slicing-window
                auto long_vector_data_begin_id = long_vector_access_id;
                long_vector_data_begin_id(vector_access_dim) =
                    long_vector_size * long_vector_access_id[vector_access_dim];

                // buffer to hold a long-vector
                TData p_long_vector[long_vector_size];

                // load data from src to the long-vector buffer
                for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * src_data_per_access;

                    const index_t src_offset = SrcDesc::GetOffsetFromMultiIndex(
                        mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id));

                    const index_t buffer_offset = i * src_data_per_access;

                    *reinterpret_cast<src_vector_t*>(&p_long_vector[buffer_offset]) =
                        *reinterpret_cast<const src_vector_t*>(&p_src[src_offset]);
                }

                // store data from the long-vector buffer to dst
                for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * dst_data_per_access;

                    const index_t buffer_offset = i * dst_data_per_access;

                    const index_t dst_offset = DstDesc::GetOffsetFromMultiIndex(
                        mDstSliceOrigin + (long_vector_data_begin_id + scalar_id));

                    *reinterpret_cast<dst_vector_t*>(&p_dst[dst_offset]) =
                        *reinterpret_cast<dst_vector_t*>(&p_long_vector[buffer_offset]);
                }
            });
    }

    private:
    Array<index_t, nDim> mSrcSliceOrigin;
    Array<index_t, nDim> mDstSliceOrigin;
};

template <class SrcDesc,
          class DstDesc,
          class SrcCoordinate,
          class DstCoordinate,
          class SliceLengths>
struct ThreadwiseGenericTensorSliceCopy_v2
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v2(SrcCoordinate src_slice_origin,
                                                             DstCoordinate dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v2()
        : ThreadwiseGenericTensorSliceCopy_v2(make_zero_array<index_t, nDim>(),
                                              make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(SrcCoordinate src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(DstCoordinate dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <class TDesc, class Seq>
    struct IsolateMergedDimSliceLengthsHack
    {
        template <class IDim>
        __device__ constexpr index_t operator()(IDim idim) const
        {
            return TDesc::ContainMultipleOriginalDimensions(idim) ? Seq{}[idim] : 1;
        }
    };

    template <class TData>
    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        constexpr auto buffer_desc = make_ConstantTensorDescriptor_packed(SliceLengths{});

        TData p_buffer_[buffer_desc.GetElementSpace()];
        TData* p_buffer = p_buffer_;

        // hacks to isolate merged dimension from normal dimensions, and calculate their offset
        // seperately
        // SrcMergedDimSliceLengthsHack has entry same as SliceLengths on src merged dimensions,
        // but 1 on normal dimensions;
        // SrcNormalDimSliceLengthsHack has entry same as SliceLengths on src normal dimensions,
        // but 1 on merged dimensions;
        using SrcMergedDimSliceLengthsHack =
            typename sequence_gen<SliceLengths::GetSize(),
                                  IsolateMergedDimSliceLengthsHack<SrcDesc, SliceLengths>>::type;

        using SrcNormalDimSliceLengthsHack =
            decltype((SliceLengths{} + Number<1>{}) - SrcMergedDimSliceLengthsHack{});

        static_ford<SrcMergedDimSliceLengthsHack>{}([&](auto merged_dim_data_id_) {
            constexpr auto merged_dim_data_id = decltype(merged_dim_data_id_){};

            const TData* p_src_tmp = p_src + (mSrcSliceOrigin + merged_dim_data_id).GetOffset();

            static_ford<SrcNormalDimSliceLengthsHack>{}([&](auto normal_dim_data_id_) {
                constexpr auto normal_dim_data_id = decltype(normal_dim_data_id_){};

                constexpr index_t buffer_offset =
                    buffer_desc.GetOffsetFromMultiIndex(merged_dim_data_id + normal_dim_data_id);

                constexpr index_t src_normal_offset =
                    SrcDesc::GetOffsetFromMultiIndex(normal_dim_data_id);

                p_buffer[buffer_offset] = p_src_tmp[src_normal_offset];
            });
        });

        // DstMergedDimSliceLengthsHack has entry same as SliceLengths on dst merged dimensions,
        // but 1 on normal dimensions;
        // DstNormalDimSliceLengthsHack has entry same as SliceLengths on dst normal dimensions,
        // but 1 on merged dimensions;
        using DstMergedDimSliceLengthsHack =
            typename sequence_gen<SliceLengths::GetSize(),
                                  IsolateMergedDimSliceLengthsHack<DstDesc, SliceLengths>>::type;

        using DstNormalDimSliceLengthsHack =
            decltype((SliceLengths{} + Number<1>{}) - DstMergedDimSliceLengthsHack{});

        static_ford<DstMergedDimSliceLengthsHack>{}([&](auto merged_dim_data_id_) {
            constexpr auto merged_dim_data_id = decltype(merged_dim_data_id_){};

            TData* p_dst_tmp = p_dst + (mDstSliceOrigin + merged_dim_data_id).GetOffset();

            static_ford<DstNormalDimSliceLengthsHack>{}([&](auto normal_dim_data_id_) {
                constexpr auto normal_dim_data_id = decltype(normal_dim_data_id_){};

                constexpr index_t buffer_offset =
                    buffer_desc.GetOffsetFromMultiIndex(merged_dim_data_id + normal_dim_data_id);

                constexpr index_t dst_normal_offset =
                    DstDesc::GetOffsetFromMultiIndex(normal_dim_data_id);

                p_dst_tmp[dst_normal_offset] = p_buffer[buffer_offset];
            });
        });
    }

    // T can be Sequence or Array
    template <class T, bool PositiveDirection>
    __device__ void MoveSrcSlicingWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += step_sizes;
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <class T, bool PositiveDirection>
    __device__ void MoveDstSlicingWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoordinate mSrcSliceOrigin;
    DstCoordinate mDstSliceOrigin;
};

} // namespace ck
#endif
