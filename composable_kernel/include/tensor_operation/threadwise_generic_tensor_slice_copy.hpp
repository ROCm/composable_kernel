#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"

namespace ck {

// This threadwise copy allow vector access of src and dst.
// It allows the vector size to be different on src and dst.
// The dimensions of vector access should be the same on src and dst.
// The dimension access order should be the same on src and dst.
// Will do valid mapping check on src data: Read 0 if src data has a invalid mapping
// Will do valid mapping check on dst data: No write if dst data has a invalid mapping
template <typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SrcDstDimAccessOrder,
          index_t SrcDstVectorReadWriteDim,
          index_t SrcDataPerRead,
          index_t DstDataPerWrite,
          AddressSpace SrcAddressSpace     = AddressSpace::Generic,
          AddressSpace DstAddressSpace     = AddressSpace::Generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::Set,
          index_t SrcDataStride            = 1,
          index_t DstDataStride            = 1>
struct ThreadwiseGenericTensorSliceCopy_v4r2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = typename TensorCoordinate<SrcDesc>::type;
    using DstCoord = typename TensorCoordinate<DstDesc>::type;

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2(const Index& src_slice_origin,
                                                               const Index& dst_slice_origin)
        : mSrcSliceOrigin(src_slice_origin), mDstSliceOrigin(dst_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SrcDstDimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<SrcDstDimAccessOrder>{}, "wrong! map is not valid");

        static_assert(SliceLengths{}[SrcDstVectorReadWriteDim] %
                              math::lcm(SrcDataPerRead, DstDataPerWrite) ==
                          0,
                      "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory read/write is allowed on src and dst
    }

    __device__ constexpr ThreadwiseGenericTensorSliceCopy_v4r2()
        : ThreadwiseGenericTensorSliceCopy_v4r2(make_zero_array<index_t, nDim>(),
                                                make_zero_array<index_t, nDim>())
    {
    }

    __device__ void SetSrcSliceOrigin(SrcCoord src_slice_origin)
    {
        mSrcSliceOrigin = src_slice_origin;
    }

    __device__ void SetDstSliceOrigin(DstCoord dst_slice_origin)
    {
        mDstSliceOrigin = dst_slice_origin;
    }

    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<SrcDstVectorReadWriteDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerRead>{};
        constexpr auto dst_data_per_access = Number<DstDataPerWrite>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerRead, DstDataPerWrite)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), SrcDstDimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            // buffer to hold a src long-vector
            SrcData p_src_long_vector[long_vector_size];

#if 1
            // zero out buffer
            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_src_long_vector[i] = 0;
            }
#endif

            // load data from src to the long-vector buffer
            for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * src_data_per_access;

                const index_t buffer_offset = i * src_data_per_access;

                const auto src_coord = mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check src data's valid mapping situation, only check the first data in this src
                //   vector. It's user's responsiblity to make sure all data in the src vector
                //   has the valid/invalid mapping situation
                transfer_data<SrcData,
                              SrcDataPerRead,
                              SrcAddressSpace,
                              AddressSpace::Vgpr,
                              InMemoryDataOperation::Set,
                              SrcDataStride,
                              1>(p_src,
                                 src_coord.GetOffset(),
                                 src_coord.IsOffsetValidAssumingUpperIndexIsValid(),
                                 SrcDesc::GetElementSpace(),
                                 p_src_long_vector,
                                 buffer_offset,
                                 true,
                                 long_vector_size);
            }

            // SrcData to DstData conversion
            DstData p_dst_long_vector[long_vector_size];

            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i]);
            }

            // store data from the long-vector buffer to dst
            for(index_t i = 0; i < long_vector_size / dst_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * dst_data_per_access;

                const index_t buffer_offset = i * dst_data_per_access;

                const auto dst_coord = mDstSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check dst data's valid mapping situation, only check the first data in this dst
                //   vector. It's user's responsiblity to make sure all data in the dst vector
                //   has the valid/invalid mapping situation
                transfer_data<DstData,
                              DstDataPerWrite,
                              AddressSpace::Vgpr,
                              DstAddressSpace,
                              DstInMemOp,
                              1,
                              DstDataStride>(p_dst_long_vector,
                                             buffer_offset,
                                             true,
                                             long_vector_size,
                                             p_dst,
                                             dst_coord.GetOffset(),
                                             dst_coord.IsOffsetValidAssumingUpperIndexIsValid(),
                                             DstDesc::GetElementSpace());
            }
        });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveSrcSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mSrcSliceOrigin += to_array(step_sizes);
        }).Else([&](auto) { mSrcSliceOrigin -= step_sizes; });
    }

    template <typename T, bool PositiveDirection>
    __device__ void MoveDstSliceWindow(const T& step_sizes_,
                                       integral_constant<bool, PositiveDirection>)
    {
        const auto step_sizes = to_array(step_sizes_);

        static_if<PositiveDirection>{}([&](auto) {
            mDstSliceOrigin += step_sizes;
        }).Else([&](auto) { mDstSliceOrigin -= step_sizes; });
    }

    private:
    SrcCoord mSrcSliceOrigin;
    DstCoord mDstSliceOrigin;
};

} // namespace ck
#endif
