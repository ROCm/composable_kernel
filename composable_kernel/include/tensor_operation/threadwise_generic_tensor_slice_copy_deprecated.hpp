#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_DEPRECATED_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_DEPRECATED_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "tensor_view.hpp"
#include "tensor_coordinate_deprecated.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R1 0
#endif

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2 0
#endif

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1 0
#endif

#ifndef CK_USE_AMD_INTRINSIC
#define CK_USE_AMD_INTRINSIC 1
#endif

#ifndef CK_USE_AMD_INTRINSIC_BUFFER_LOAD_STORE
#define CK_USE_AMD_INTRINSIC_BUFFER_LOAD_STORE 1
#endif

namespace ck {

// This threadwise copy allow vector access of src and dst.
// It allows the dimensions of vector access to be different on src and dst.
// It also allows the vector size to be different on src and dst.
// It also allows order of access to be different on src and dst.
// It use register as buffer to hold all data moving from src to dst.
// It is designed for copying small amount of data, and src and dst are
// device memory or LDS.
// When copying large amout of data, let's hope compiler will reduce register
// used for the buffer.
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
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

    template <typename TData>
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

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R1
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
#else
            ford<decltype(src_access_lengths), SrcDimAccessOrder>{}([&](auto src_access_id) {
                auto src_data_begin_id = src_access_id;
                src_data_begin_id(src_vector_access_dim) =
                    src_access_id[src_vector_access_dim] * src_data_per_access;

                const index_t src_offset =
                    SrcDesc::GetOffsetFromMultiIndex(mSrcSliceOrigin + src_data_begin_id);

                // load vector from src
                const vector_t vector_data = *reinterpret_cast<const vector_t*>(&p_src[src_offset]);

                // unpack vector into buffer
                for(index_t i = 0; i < SrcDataPerAccess; ++i)
                {
                    auto scalar_id                   = make_zero_array<index_t, nDim>();
                    scalar_id(src_vector_access_dim) = i;

                    const index_t buffer_offset =
                        buffer_desc.GetOffsetFromMultiIndex(src_data_begin_id + scalar_id);

                    p_buffer[buffer_offset] = reinterpret_cast<const TData*>(&vector_data)[i];
                }
            });
#endif
        }

        // copy data from buffer to dst
        {
            using vector_t = typename vector_type<TData, DstDataPerAccess>::MemoryType;

            constexpr auto dst_vector_access_dim = Number<DstVectorAccessDim>{};
            constexpr auto dst_data_per_access   = Number<DstDataPerAccess>{};

            constexpr auto dst_access_lengths = SliceLengths::Modify(
                dst_vector_access_dim,
                SliceLengths::Get(dst_vector_access_dim) / dst_data_per_access);

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R1
            static_ford<decltype(dst_access_lengths), DstDimAccessOrder>{}([&](auto dst_access_id) {
                constexpr auto dst_data_begin_id = dst_access_id.Modify(
                    dst_vector_access_dim,
                    dst_access_id[dst_vector_access_dim] * dst_data_per_access);

                vector_t vector_data{};

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
#else
            ford<decltype(dst_access_lengths), DstDimAccessOrder>{}([&](auto dst_access_id) {
                auto dst_data_begin_id = dst_access_id;
                dst_data_begin_id(dst_vector_access_dim) =
                    dst_access_id[dst_vector_access_dim] * dst_data_per_access;

                vector_t vector_data{};

                // pack vector from buffer
                for(index_t i = 0; i < DstDataPerAccess; ++i)
                {
                    auto scalar_id                   = make_zero_array<index_t, nDim>();
                    scalar_id(dst_vector_access_dim) = i;

                    const index_t buffer_offset =
                        buffer_desc.GetOffsetFromMultiIndex(dst_data_begin_id + scalar_id);

                    reinterpret_cast<TData*>(&vector_data)[i] = p_buffer[buffer_offset];
                }

                const index_t dst_offset =
                    DstDesc::GetOffsetFromMultiIndex(mDstSliceOrigin + dst_data_begin_id);

                // store vector into dst
                *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) = vector_data;
            });
#endif
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
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
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

    template <typename TData>
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

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2
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
#else
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
#endif
    }

    private:
    Array<index_t, nDim> mSrcSliceOrigin;
    Array<index_t, nDim> mDstSliceOrigin;
};

// This version use TensorCoordinate_deprecated
// This threadwise copy allow vector access of src and dst.
// It allows the dimensions of vector access to be different on src and dst.
// It also allows the vector size to be different on src and dst.
// It also allows order of access to be different on src and dst.
// It use register as buffer to hold all data moving from src to dst.
// It is designed for copying small amount of data, and src and dst are
// device memory or LDS.
// When copying large amout of data, let's hope compiler will reduce register
// used for the buffer.
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct ThreadwiseGenericTensorSliceCopy_v2r1
{
    static constexpr index_t nDim = SliceLengths::GetSize();

    using Index = MultiIndex<nDim>;

    using SrcCoordinate = typename TensorCoordinate_deprecated<SrcDesc>::type;
    using DstCoordinate = typename TensorCoordinate_deprecated<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v2r1(const Index& src_slice_origin,
                                                               const Index& dst_slice_origin)
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

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v2r1()
        : ThreadwiseGenericTensorSliceCopy_v2r1(make_zero_array<index_t, nDim>(),
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

    template <typename TDesc, class Lengths>
    struct IsolateMergedDimLengths
    {
        template <typename IDim>
        __device__ constexpr index_t operator()(IDim idim) const
        {
            return TDesc::ContainMultipleOriginalDimensions(idim) ? Lengths{}[idim] : 1;
        }
    };

    template <typename TData,
              address_space_t SrcAddressSpace = address_space_t::generic,
              address_space_t DstAddressSpace = address_space_t::generic>
    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        constexpr auto buffer_desc = make_ConstantTensorDescriptor_packed(SliceLengths{});

        TData p_buffer_[buffer_desc.GetElementSpace()];
        TData* p_buffer = p_buffer_;

        // copy data from src into buffer
        {
            using src_vector_t = typename vector_type<TData, SrcDataPerAccess>::MemoryType;

            constexpr auto src_vector_access_dim = Number<SrcVectorAccessDim>{};
            constexpr auto src_data_per_access   = Number<SrcDataPerAccess>{};

            constexpr auto src_access_lengths = SliceLengths::Modify(
                src_vector_access_dim,
                SliceLengths::Get(src_vector_access_dim) / src_data_per_access);

            // Offset w.r.t merged dimensions need to be calculated at run-time. Offset w.r.t
            // normal dimensions is known at compile time.
            // Below is a hack to isolate merged dimension id from normal dimension id, so the
            // corresponding offset can be calculated seperately at run-time and compile-time.
            // src_merged_dim_access_lengths has the same value as src_access_lengths on src's
            // merged dimensions, and has value = 1 on normal dimensions;
            // src_merged_dim_access_lengths has the same value as src_access_lengths on src's
            // normal dimensions, and has value = 1 on merged dimensions;
            constexpr auto src_merged_dim_access_lengths = typename sequence_gen<
                nDim,
                IsolateMergedDimLengths<SrcDesc, decltype(src_access_lengths)>>::type{};

            constexpr auto src_normal_dim_access_lengths =
                src_access_lengths + Number<1>{} - src_merged_dim_access_lengths;

            ford<decltype(src_merged_dim_access_lengths), SrcDimAccessOrder>{}([&](
                auto src_merged_dim_access_id) {

                auto src_merged_dim_data_id = src_merged_dim_access_id;
                src_merged_dim_data_id(src_vector_access_dim) =
                    src_merged_dim_access_id[src_vector_access_dim] * src_data_per_access;

                // offset w.r.t. merged dimension need be computed at run-time,
                const index_t src_merged_offset =
                    (mSrcSliceOrigin + src_merged_dim_data_id).GetOffset();

                ford<decltype(src_normal_dim_access_lengths), SrcDimAccessOrder>{}([&](
                    auto src_normal_dim_access_id) {

                    auto src_normal_dim_data_id = src_normal_dim_access_id;
                    src_normal_dim_data_id(src_vector_access_dim) =
                        src_normal_dim_access_id[src_vector_access_dim] * src_data_per_access;

                    // offset w.r.t. normal dimension is known at compile-time
                    const index_t src_normal_offset =
                        SrcDesc::GetOffsetFromMultiIndex(src_normal_dim_data_id);

                    src_vector_t vector_data;

                    // Read vector from src.
                    //   1. Source code version can take src of all kinds of memory-space
                    //   2. Inline asm versions using global_load or buffer_load can only take
                    //      src from global-memory
                    //
                    // Commemt for loading from global-memory:
                    //   When
                    //     1) using source code, in order for compiler to emit optimal
                    //        load instruction, or
                    //     2) using inline asm (global_load or buffer_load), in order
                    //        for inline asm to be valid,
                    //   following assumptions need to be satisfied:
                    //     1. p_src need to be block-invariant (assumption)
                    //     2. src_normal_offset must be calculatd at compile time (guaranteed)
                    //     3. src_merged_offset can be runtime value (no assumption imposed)
                    static_if<SrcAddressSpace == address_space_t::global>{}([&](auto) {
#if CK_USE_AMD_INTRINSIC && CK_USE_AMD_INTRINSIC_BUFFER_LOAD_STORE
                        vector_data = __buffer_load<TData, SrcDataPerAccess>(
                            p_src, src_merged_offset, src_normal_offset);
#else
                        vector_data = *reinterpret_cast<const src_vector_t*>(
                            &p_src[src_normal_offset + src_merged_offset]);
#endif
                    }).Else([&](auto) {
                        // src can be all kinds of memory-space.
                        vector_data = *reinterpret_cast<const src_vector_t*>(
                            &p_src[src_normal_offset + src_merged_offset]);
                    });

                    // unpack vector into buffer
                    for(index_t i = 0; i < SrcDataPerAccess; ++i)
                    {
                        auto scalar_id                   = make_zero_array<index_t, nDim>();
                        scalar_id(src_vector_access_dim) = i;

                        const index_t buffer_offset = buffer_desc.GetOffsetFromMultiIndex(
                            src_merged_dim_data_id + src_normal_dim_data_id + scalar_id);

                        p_buffer[buffer_offset] = reinterpret_cast<const TData*>(&vector_data)[i];
                    }
                });
            });
        }

        // copy data from buffer into dst
        {
            using dst_vector_t = typename vector_type<TData, DstDataPerAccess>::MemoryType;

            constexpr auto dst_vector_access_dim = Number<DstVectorAccessDim>{};
            constexpr auto dst_data_per_access   = Number<DstDataPerAccess>{};

            constexpr auto dst_access_lengths = SliceLengths::Modify(
                dst_vector_access_dim,
                SliceLengths::Get(dst_vector_access_dim) / dst_data_per_access);

            constexpr auto dst_merged_dim_access_lengths = typename sequence_gen<
                nDim,
                IsolateMergedDimLengths<DstDesc, decltype(dst_access_lengths)>>::type{};

            constexpr auto dst_normal_dim_access_lengths =
                dst_access_lengths + Number<1>{} - dst_merged_dim_access_lengths;

            ford<decltype(dst_merged_dim_access_lengths), DstDimAccessOrder>{}(
                [&](auto dst_merged_dim_access_id) {

                    auto dst_merged_dim_data_id = dst_merged_dim_access_id;
                    dst_merged_dim_data_id(dst_vector_access_dim) =
                        dst_merged_dim_access_id[dst_vector_access_dim] * dst_data_per_access;

                    // offset w.r.t. merged dimension need be computed at run-time,
                    const index_t dst_merged_offset =
                        (mDstSliceOrigin + dst_merged_dim_data_id).GetOffset();

                    ford<decltype(dst_normal_dim_access_lengths), DstDimAccessOrder>{}([&](
                        auto dst_normal_dim_access_id) {

                        auto dst_normal_dim_data_id = dst_normal_dim_access_id;
                        dst_normal_dim_data_id(dst_vector_access_dim) =
                            dst_normal_dim_access_id[dst_vector_access_dim] * dst_data_per_access;

                        dst_vector_t vector_data;

                        // pack vector from buffer
                        for(index_t i = 0; i < DstDataPerAccess; ++i)
                        {
                            auto scalar_id                   = make_zero_array<index_t, nDim>();
                            scalar_id(dst_vector_access_dim) = i;

                            const index_t buffer_offset = buffer_desc.GetOffsetFromMultiIndex(
                                dst_merged_dim_data_id + dst_normal_dim_data_id + scalar_id);

                            reinterpret_cast<TData*>(&vector_data)[i] = p_buffer[buffer_offset];
                        }

                        // offset w.r.t. normal dimension is known at compile-time
                        const index_t dst_normal_offset =
                            DstDesc::GetOffsetFromMultiIndex(dst_normal_dim_data_id);

                        // Write vector into dst.
                        //   1. Source code version can take dst of all kinds of memory-space
                        //   2. Inline asm versions using global_store or buffer_store can only take
                        //      dst from global-memory
                        //
                        // Commemt for storing into global-memory:
                        //   When
                        //     1) using source code, in order for compiler to emit optimal
                        //        store instruction, or
                        //     2) using inline asm (global_store or buffer_store), in order
                        //        for inline asm to be valid,
                        //   following assumptions need to be satisfied:
                        //     1. p_dst need to be block-invariant (assumption)
                        //     2. dst_normal_offset must be calculatd at compile time (guaranteed)
                        //     3. dst_merged_offset can be runtime value (no assumption imposed)
                        static_if<DstAddressSpace == address_space_t::global>{}([&](auto) {
#if CK_USE_AMD_INTRINSIC && CK_USE_AMD_INTRINSIC_BUFFER_LOAD_STORE
                            __buffer_store<TData, DstDataPerAccess>(
                                vector_data, p_dst, dst_merged_offset, dst_normal_offset);
#else
                            *reinterpret_cast<dst_vector_t*>(
                                &p_dst[dst_normal_offset + dst_merged_offset]) = vector_data;
#endif
                        }).Else([&](auto) {
                            // dst can be all kinds of memory-space
                            *reinterpret_cast<dst_vector_t*>(
                                &p_dst[dst_normal_offset + dst_merged_offset]) = vector_data;
                        });
                    });
                });
        }
    }

    // T can be Sequence or Array
    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += step_sizes;
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoordinate mSrcSliceOrigin;
    DstCoordinate mDstSliceOrigin;
};

// this version use TensorView and TensorCoordinate_deprecated
template <typename SrcTensor,
          typename DstTensor,
          typename SliceLengths,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct ThreadwiseGenericTensorSliceCopy_v3r1
{
    static constexpr index_t nDim = SrcTensor::GetNumOfDimension();
    using data_type               = remove_cv_t<typename SrcTensor::data_type>;

    using SrcCoordinate = typename SrcTensor::coordinate_type;
    using DstCoordinate = typename DstTensor::coordinate_type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v3r1(SrcTensor src,
                                                               SrcCoordinate src_slice_origin,
                                                               DstTensor dst,
                                                               DstCoordinate dst_slice_origin)
        : mSrc{src},
          mDst{dst},
          mSrcSlice{src.Slice(src_slice_origin, SliceLengths{})},
          mDstSlice{dst.Slice(dst_slice_origin, SliceLengths{})}
    {
        static_assert(nDim == SrcTensor::GetNumOfDimension() &&
                          nDim == DstTensor::GetNumOfDimension() &&
                          nDim == SliceLengths::GetSize() && nDim == SrcDimAccessOrder::GetSize() &&
                          nDim == DstDimAccessOrder::GetSize(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDimAccessOrder>::value &&
                          is_valid_sequence_map<DstDimAccessOrder>::value,
                      "wrong! map is not valid");

        static_assert(is_same<remove_cv_t<typename SrcTensor::data_type>,
                              remove_cv_t<typename DstTensor::data_type>>{},
                      "wrong! type conversion is not supported yet");

        static_assert(decltype(mSrcSlice)::IsVectorizationAllowed(Number<SrcVectorAccessDim>{},
                                                                  Number<SrcDataPerAccess>{}) &&
                          decltype(mDstSlice)::IsVectorizationAllowed(Number<DstVectorAccessDim>{},
                                                                      Number<DstDataPerAccess>{}),
                      "wrong! vectorized access is not allowed");
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v3r1()
        : ThreadwiseGenericTensorSliceCopy_v3r1(
              SrcTensor{}, SrcCoordinate{}, DstTensor{}, DstCoordinate{})
    {
    }

    __device__ void Run() const
    {
        // buffer
        constexpr auto buffer_desc = make_ConstantTensorDescriptor_packed(SrcTensor::GetLengths());
        data_type p_buffer[buffer_desc.GetElementSpace()];
        auto buffer = make_TensorView(buffer_desc, p_buffer);

        // copy data from src into buffer
        {
            using src_vector_t = typename vector_type<data_type, SrcDataPerAccess>::MemoryType;

            constexpr auto src_vector_access_dim = Number<SrcVectorAccessDim>{};
            constexpr auto src_data_per_access   = Number<SrcDataPerAccess>{};

            auto src_slice_vectorized =
                mSrcSlice.Vectorize(src_vector_access_dim, src_data_per_access);

            ford<decltype(src_slice_vectorized.GetLengths()), SrcDimAccessOrder>{}(
                [&](auto src_vector_id) {
                    // load vector from src
                    const src_vector_t vector_data = src_slice_vectorized[src_vector_id];

                    // unpack vector into buffer
                    auto src_scalar_id = src_vector_id;
                    src_scalar_id(src_vector_access_dim) *= src_data_per_access;

                    for(index_t i = 0; i < SrcDataPerAccess; ++i)
                    {
                        auto id                   = make_zero_array<index_t, nDim>();
                        id(src_vector_access_dim) = i;

                        buffer(src_scalar_id + id) =
                            reinterpret_cast<const data_type*>(&vector_data)[i];
                    }
                });
        }

        // copy data from buffer into dst
        {
            using dst_vector_t = typename vector_type<data_type, DstDataPerAccess>::MemoryType;

            constexpr auto dst_vector_access_dim = Number<DstVectorAccessDim>{};
            constexpr auto dst_data_per_access   = Number<DstDataPerAccess>{};

            auto dst_slice_vectorized =
                mDstSlice.Vectorize(dst_vector_access_dim, dst_data_per_access);

            ford<decltype(dst_slice_vectorized.GetLengths()), DstDimAccessOrder>{}(
                [&](auto dst_vector_id) {

                    dst_vector_t vector_data{};

                    // pack vector from buffer
                    auto dst_scalar_id = dst_vector_id;
                    dst_scalar_id(dst_vector_access_dim) *= dst_data_per_access;

                    for(index_t i = 0; i < DstDataPerAccess; ++i)
                    {
                        auto id                   = make_zero_array<index_t, nDim>();
                        id(dst_vector_access_dim) = i;

                        reinterpret_cast<data_type*>(&vector_data)[i] = buffer[dst_scalar_id + id];
                    }

                    // write vector into dst
                    dst_slice_vectorized(dst_vector_id) = vector_data;
                });
        }
    }

    // T can be Sequence or Array
    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        mSrc.MoveSliceWindow(mSrcSlice, step_sizes, integral_constant<bool, PositiveDirection>{});
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection>)
    {
        mDst.MoveSliceWindow(mDstSlice, step_sizes, integral_constant<bool, PositiveDirection>{});
    }

    private:
    using SrcSlice = decltype(SrcTensor{}.Slice(make_zero_array<index_t, nDim>(), SliceLengths{}));
    using DstSlice = decltype(DstTensor{}.Slice(make_zero_array<index_t, nDim>(), SliceLengths{}));

    SrcTensor mSrc;
    DstTensor mDst;
    SrcSlice mSrcSlice;
    DstSlice mDstSlice;
};

} // namespace ck
#endif
