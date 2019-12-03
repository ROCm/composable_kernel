#ifndef CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_THREADWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_coordinate.hpp"

namespace ck {

// This version use multi-index transformation
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
          index_t DstDataPerAccess,
          AddressSpace SrcAddressSpace     = AddressSpace::generic,
          AddressSpace DstAddressSpace     = AddressSpace::generic,
          InMemoryDataOperation DstInMemOp = InMemoryDataOperation::none>
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
                          nDim == DimAccessOrder::Size(),
                      "wrong! # of dimensions not the same");

        static_assert(is_valid_sequence_map<DimAccessOrder>{}, "wrong! map is not valid");

        static_assert(
            SliceLengths{}[VectorAccessDim] % math::lcm(SrcDataPerAccess, DstDataPerAccess) == 0,
            "wrong! cannot evenly divide");

        // TODO:: sanity-check if vectorized memory access is allowed on src and dst
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

    // Will do padding check on src data: Read 0 if src data is in padding area.
    // Will do padding check on dst data: No write if dst data is in paddin area.
    template <typename SrcData, typename DstData>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerAccess>{};
        constexpr auto dst_data_per_access = Number<DstDataPerAccess>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerAccess, DstDataPerAccess)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        ford<decltype(long_vector_access_lengths), DimAccessOrder>{}([&](
            auto long_vector_access_id) {

            // data id w.r.t slicing-window
            auto long_vector_data_begin_id = long_vector_access_id;
            long_vector_data_begin_id(vector_access_dim) =
                long_vector_size * long_vector_access_id[vector_access_dim];

            // buffer to hold a src long-vector
            SrcData p_src_long_vector[long_vector_size];

            // zero out buffer
            for(index_t i = 0; i < long_vector_size; ++i)
            {
                p_src_long_vector[i] = 0;
            }

            // load data from src to the long-vector buffer
            for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
            {
                auto scalar_id               = make_zero_array<index_t, nDim>();
                scalar_id(vector_access_dim) = i * src_data_per_access;

                const index_t buffer_offset = i * src_data_per_access;

                const auto src_coord = mSrcSliceOrigin + (long_vector_data_begin_id + scalar_id);

                // Check src vector's padding situation, only check the first data in this src
                //   vector. It's user's responsiblity to make sure all data in the src vector
                //   has the same padding situation
                if(src_coord.IsUpperIndexMappedToValidOffset())
                {
                    move_data<SrcData,
                              SrcDataPerAccess,
                              SrcAddressSpace,
                              AddressSpace::vgpr,
                              InMemoryDataOperation::none>(
                        p_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset);
                }
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

                // Check dst vector's padding situation, only check the first data in this dst
                //   vector. It's user's responsiblity to make sure all data in the dst vector
                //   has the same padding situation
                if(dst_coord.IsUpperIndexMappedToValidOffset())
                {
                    move_data<DstData,
                              DstDataPerAccess,
                              AddressSpace::vgpr,
                              DstAddressSpace,
                              DstInMemOp>(
                        p_dst_long_vector, buffer_offset, p_dst, dst_coord.GetOffset());
                }
            }
        });
    }

    // Modify Length to 1, if Mask is set to false
    // Used for isolating linear dimension from non-linear dimensions
    template <index_t... Lengths, index_t... Mask>
    __device__ static constexpr auto mask_lengths(Sequence<Lengths...>, Sequence<Mask...>)
    {
        return Sequence<(Mask ? Lengths : 1)...>{};
    }

    // Will do padding check on src data: Read 0 if src data is in padding area.
    // Will do padding check on dst data: No write if dst data is in paddin area.
    // This version is optimized for address calculation of src tensor
    // TODO: this function is not compiled to expected ISA
    template <typename SrcData, typename DstData>
    __device__ void Run_optimized_src_address_calculation(const SrcData* p_src,
                                                          DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerAccess>{};
        constexpr auto dst_data_per_access = Number<DstDataPerAccess>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerAccess, DstDataPerAccess)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        // separate linear dimensions from non-linear dimensions
        constexpr auto src_linear_dim_mask    = SrcDesc::GetLinearDimensionMask();
        constexpr auto src_nonlinear_dim_mask = SrcDesc::GetNonLinearDimensionMask();

        static_assert(src_linear_dim_mask.At(VectorAccessDim) ||
                          long_vector_size == SrcDataPerAccess,
                      "Warning! VectorAccessDim is not SrcDesc's linear dimension, performance "
                      "would drop");

        // separate steps into linear and non-linear components, accoording to src tensor
        constexpr auto linear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, src_linear_dim_mask);

        constexpr auto nonlinear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, src_nonlinear_dim_mask);

        // loop over src's non-linear dimensions
        ford<decltype(nonlinear_long_vector_access_lengths)>{}([&](
            auto nonlinear_dim_long_vector_access_id) {

            // calculate step-sizes along src's nonlinear dimensions
            auto nonlinear_dim_data_steps = nonlinear_dim_long_vector_access_id;
            nonlinear_dim_data_steps(vector_access_dim) =
                long_vector_size * nonlinear_dim_long_vector_access_id[vector_access_dim];

            // move src cooridnate along nonlinear dimensions
            // this coordinate contains run-time per-thread offset
            const auto src_nonlinear_coord = mSrcSliceOrigin + nonlinear_dim_data_steps;

            // loop over src's linear dimensions
            ford<decltype(linear_long_vector_access_lengths)>{}([&](
                auto linear_dim_long_vector_access_id) {

                // step-sizes along src's linear dimensions
                auto linear_dim_data_steps = linear_dim_long_vector_access_id;
                linear_dim_data_steps(vector_access_dim) =
                    long_vector_size * linear_dim_long_vector_access_id[vector_access_dim];

                // buffer to hold a long-vector
                SrcData p_src_long_vector[long_vector_size];

                // zero out buffer
                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_src_long_vector[i] = 0;
                }

                // Loop over VectorAccessDim, and load data from src to the
                //   long-vector buffer.
                // If VectorAccessDim is src's linear dimension, then src's
                //   offset-diff due to this looping is known at compile-time. If
                //   VectorAccessDim is src's nonlinear dimension, then src's
                //   offset-diff due to this looping is only known at run-time. For best
                //   performance, VectorAccessDim, should be src's linear dimension
                for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * src_data_per_access;

                    const index_t buffer_offset = i * src_data_per_access;

                    // move src cooridnate along linear dimensions
                    const auto src_coord =
                        src_nonlinear_coord + (linear_dim_data_steps + scalar_id);

#if CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF // tweaking
                    // this is src compile-time offset
                    const index_t src_linear_offset =
                        src_nonlinear_coord.CalculateOffsetDiff(linear_dim_data_steps + scalar_id);
#else
                    // this is src compile-time offset
                    const index_t src_linear_offset =
                        src_coord.GetOffset() - src_nonlinear_coord.GetOffset();
#endif

                    // Check src vector's padding situation, only check the first data in
                    //   this src vector. It's user's responsiblity to make sure all data in
                    //   the src vector has the same padding situation
                    if(src_coord.IsUpperIndexMappedToValidOffset())
                    {
                        move_data<SrcData,
                                  SrcDataPerAccess,
                                  SrcAddressSpace,
                                  AddressSpace::vgpr,
                                  InMemoryDataOperation::none>(p_src,
                                                               src_nonlinear_coord.GetOffset() +
                                                                   src_linear_offset,
                                                               p_src_long_vector,
                                                               buffer_offset);
                    }
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

                    // dst offset is calculated here, without explicitly separating into
                    //   compile-time and per-thread component
                    const auto dst_coord = mDstSliceOrigin + (nonlinear_dim_data_steps +
                                                              linear_dim_data_steps + scalar_id);

                    // Check dst vector's padding situation, only check the first data in
                    //   this dst vector. It's user's responsiblity to make sure all data in
                    //   the dst vector has the same padding situation
                    if(dst_coord.IsUpperIndexMappedToValidOffset())
                    {
                        move_data<DstData,
                                  DstDataPerAccess,
                                  AddressSpace::vgpr,
                                  DstAddressSpace,
                                  DstInMemOp>(
                            p_dst_long_vector, buffer_offset, p_dst, dst_coord.GetOffset());
                    }
                }
            });
        });
    }

    // Will do padding check on src data: Read 0 if src data is in padding area.
    // Will do padding check on dst data: No write if dst data is in paddin area.
    // This version is optimized for address calculation of dst tensor
    // TODO: this function is not compiled to expected ISA
    template <typename SrcData, typename DstData>
    __device__ void Run_optimized_dst_address_calculation(const SrcData* p_src,
                                                          DstData* p_dst) const
    {
        constexpr auto vector_access_dim = Number<VectorAccessDim>{};

        constexpr auto src_data_per_access = Number<SrcDataPerAccess>{};
        constexpr auto dst_data_per_access = Number<DstDataPerAccess>{};

        constexpr auto long_vector_size = Number<math::lcm(SrcDataPerAccess, DstDataPerAccess)>{};

        constexpr auto long_vector_access_lengths = SliceLengths::Modify(
            vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size);

        // separate linear dimensions from non-linear dimensions
        constexpr auto dst_linear_dim_mask    = DstDesc::GetLinearDimensionMask();
        constexpr auto dst_nonlinear_dim_mask = DstDesc::GetNonLinearDimensionMask();

        static_assert(dst_linear_dim_mask.At(VectorAccessDim) ||
                          long_vector_size == DstDataPerAccess,
                      "Warning! VectorAccessDim is not DstDesc's linear dimension, performance "
                      "would drop");

        // separate steps into linear and non-linear components, accoording to dst tensor
        constexpr auto linear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, dst_linear_dim_mask);

        constexpr auto nonlinear_long_vector_access_lengths =
            mask_lengths(long_vector_access_lengths, dst_nonlinear_dim_mask);

        // loop over dst's non-linear dimensions
        ford<decltype(nonlinear_long_vector_access_lengths)>{}([&](
            auto nonlinear_dim_long_vector_access_id) {

            // calculate step-sizes along dst's nonlinear dimensions
            auto nonlinear_dim_data_steps = nonlinear_dim_long_vector_access_id;
            nonlinear_dim_data_steps(vector_access_dim) =
                long_vector_size * nonlinear_dim_long_vector_access_id[vector_access_dim];

            // move dst cooridnate along nonlinear dimensions
            // this coordinate contains run-time per-thread offset
            const auto dst_nonlinear_coord = mDstSliceOrigin + nonlinear_dim_data_steps;

            // loop over dst's linear dimensions
            ford<decltype(linear_long_vector_access_lengths)>{}([&](
                auto linear_dim_long_vector_access_id) {

                // step-sizes along dst's linear dimensions
                auto linear_dim_data_steps = linear_dim_long_vector_access_id;
                linear_dim_data_steps(vector_access_dim) =
                    long_vector_size * linear_dim_long_vector_access_id[vector_access_dim];

                // buffer to hold a long-vector
                SrcData p_src_long_vector[long_vector_size];

                // zero out buffer
                for(index_t i = 0; i < long_vector_size; ++i)
                {
                    p_src_long_vector[i] = 0;
                }

                // Loop over VectorAccessDim, and load data from src to the
                //   long-vector buffer.
                // If VectorAccessDim is dst's linear dimension, then dst's
                //   offset-diff due to this looping is known at compile-time. If
                //   VectorAccessDim is dst's nonlinear dimension, then dst's
                //   offset-diff due to this looping is only known at run-time. For best
                //   performance, VectorAccessDim, should be dst's linear dimension
                for(index_t i = 0; i < long_vector_size / src_data_per_access; ++i)
                {
                    auto scalar_id               = make_zero_array<index_t, nDim>();
                    scalar_id(vector_access_dim) = i * src_data_per_access;

                    const index_t buffer_offset = i * src_data_per_access;

                    // src offset is calculated here, without explicitly separating into
                    //   compile-time and per-thread component
                    const auto src_coord = mSrcSliceOrigin + (nonlinear_dim_data_steps +
                                                              linear_dim_data_steps + scalar_id);

                    // Check src vector's padding situation, only check the first data in
                    //   this src vector. It's user's responsiblity to make sure all data in
                    //   the src vector has the same padding situation
                    if(src_coord.IsUpperIndexMappedToValidOffset())
                    {
                        move_data<SrcData,
                                  SrcDataPerAccess,
                                  SrcAddressSpace,
                                  AddressSpace::vgpr,
                                  InMemoryDataOperation::none>(
                            p_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset);
                    }
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

                    // move dst cooridnate along linear dimensions
                    const auto dst_coord =
                        dst_nonlinear_coord + (linear_dim_data_steps + scalar_id);

#if CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF // tweaking
                    // this is dst compile-time offset
                    const index_t dst_linear_offset =
                        dst_nonlinear_coord.CalculateOffsetDiff(linear_dim_data_steps + scalar_id);
#else
                    // this is dst compile-time offset
                    const index_t dst_linear_offset =
                        dst_coord.GetOffset() - dst_nonlinear_coord.GetOffset();
#endif

                    // Check dst vector's padding situation, only check the first data in
                    //   this dst vector. It's user's responsiblity to make sure all data in
                    //   the dst vector has the same padding situation
                    if(dst_coord.IsUpperIndexMappedToValidOffset())
                    {
                        move_data<DstData,
                                  DstDataPerAccess,
                                  AddressSpace::vgpr,
                                  DstAddressSpace,
                                  DstInMemOp>(p_dst_long_vector,
                                              buffer_offset,
                                              p_dst,
                                              dst_nonlinear_coord.GetOffset() + dst_linear_offset);
                    }
                }
            });
        });
    }

    __device__ static constexpr bool HasWorkingOptimizedAddressCalculation()
    {
#if CK_EXPERIMENTAL_THREADWISE_COPY_V4R2_USE_OPTIMIZED_ADDRESS_CACLULATION // tweaking
        return true;
#else
        return false;
#endif
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
