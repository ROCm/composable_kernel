// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

namespace ck {

// Helper function to compare multi-indices element by element
template <typename Idx1, typename Idx2>
constexpr bool indices_equal(const Idx1& idx1, const Idx2& idx2)
{
    if constexpr (Idx1::Size() != Idx2::Size()) {
        return false;
    } else {
        bool equal = true;
        static_for<0, Idx1::Size(), 1>{}([&](auto i) {
            equal = equal && (idx1.At(i) == idx2.At(i));
        });
        return equal;
    }
}

template <typename SpaceFillingCurveSrc, typename SpaceFillingCurveDst, index_t DstIdx, index_t SearchIdx, index_t MaxIdx>
struct LinearIndexFinder
{
    static constexpr index_t find()
    {
        if constexpr (SearchIdx >= MaxIdx) {
            return index_t(-1); // Not found
        } else {
            constexpr auto dst_md = SpaceFillingCurveDst::GetIndex(Number<DstIdx>{});
            constexpr auto src_md = SpaceFillingCurveSrc::GetIndex(Number<SearchIdx>{});
            
            if constexpr (indices_equal(dst_md, src_md)) {
                return SearchIdx;
            } else {
                return LinearIndexFinder<SpaceFillingCurveSrc, SpaceFillingCurveDst, DstIdx, SearchIdx + 1, MaxIdx>::find();
            }
        }
    }
};

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          bool PackedInput = false,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3_pass_through
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t nDim = SliceLengths::Size();

    static constexpr bool SerpentineAccessPattern = true;

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3_pass_through(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation& element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx)),
          element_op_{element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");

        // For now, SrcData must be float and DstData must be ck::bhalf_t
        static_assert(std::is_same_v<SrcData, float>,
                      "wrong! SrcData must be float");
        static_assert(std::is_same_v<DstData, ck::bhalf_t>,
                      "wrong! DstData must be bhalf_t");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurveDst = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPattern>;

        using SpaceFillingCurveSrc = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    false>;

        static_assert(DstScalarPerVector == SpaceFillingCurveDst::ScalarPerVector,
                    "wrong!DstScalarPerVector != SpaceFillingCurveDst::ScalarPerVector");

        constexpr auto num_access = SpaceFillingCurveDst::GetNumOfAccess();

        static_assert(std::is_same_v<DimAccessOrder, Sequence<0, 1, 2, 3, 4, 5, 6, 7>>,
                "wrong! DimAccessOrder must be the identity sequence <0, 1, 2, 3, 4, 5, 6, 7>");
          
        static_assert(1 == SpaceFillingCurveDst::ScalarPerVector, "wrong!1 != SpaceFillingCurve::ScalarPerVector");
        static_assert(1 == DstScalarPerVector, "wrong!1 != DstScalarPerVector");

        static_assert(SpaceFillingCurveDst::GetNumOfAccess() == SpaceFillingCurveSrc::GetNumOfAccess(),
                  "wrong! SpaceFillingCurveDst and SpaceFillingCurveSrc must have the same number of access.");

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr index_t idx_src_1d = LinearIndexFinder<SpaceFillingCurveSrc, SpaceFillingCurveDst, idx_1d.value, 0, num_access>::find();
            static_assert(idx_src_1d != index_t(-1), "wrong! Cannot find linear index.");
            
            // Map linear index to the packed BF16 index
            constexpr index_t idx_src_1d_packed = idx_src_1d / 2;
            constexpr index_t pair_index = idx_src_1d % 2; 

            constexpr auto idx_md_src = SpaceFillingCurveSrc::GetIndex(Number<idx_src_1d_packed>{});
            constexpr index_t src_offset = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_src);

            union
            {
                float fp32;
                ck::bhalf2_t bf16x2;
            } converter;
            converter.fp32 = src_buf[Number<src_offset>{}];

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, ck::bhalf_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                converter.bf16x2[pair_index]);

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurveDst::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPattern>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    DstCoord dst_coord_;
    const ElementwiseOperation element_op_;
}; // namespace ThreadwiseTensorSliceTransfer_v1r3_pass_through

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3_packed_cast 
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    // Currently these must match, either true or false for both.
    // Otherwise we increase the register usage and end-up using the scratch memory.
    // The linear pattern seems to be better tahn serpentine.
    // Howeverm for LDS writes, serpentine pattern could be better.
    static constexpr bool SerpentineAccessPatternDst = true;
    static constexpr bool SerpentineAccessPatternSrc = true;

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3_packed_cast(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation&)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");

        // Assert that elementwise op is pass through.
        static_assert(
            std::is_same_v<remove_cvref_t<ElementwiseOperation>, ck::tensor_operation::element_wise::PassThrough>,
            "wrong! ElementwiseOperation must be PassThrough");

        // For now, SrcData must be float and DstData must be ck::bhalf_t
        static_assert(std::is_same_v<SrcData, float>,
                      "wrong! SrcData must be float");
        static_assert(std::is_same_v<DstData, ck::bhalf_t>,
                      "wrong! DstData must be bhalf_t");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        // Note: We cannot easily have different SFC for src and dst because this will lead to increased register usage.
        using SpaceFillingCurveDst = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPatternDst>;

        using SpaceFillingCurveSrc = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPatternSrc>;
          
        static_assert(1 == SpaceFillingCurveDst::ScalarPerVector,
                    "wrong!DstScalarPerVector != SpaceFillingCurveDst::ScalarPerVector");

        static_assert(SpaceFillingCurveDst::GetNumOfAccess() == SpaceFillingCurveSrc::GetNumOfAccess(),
                  "wrong! SpaceFillingCurveDst and SpaceFillingCurveSrc must have the same number of access.");

        constexpr auto num_access = SpaceFillingCurveDst::GetNumOfAccess();
        constexpr index_t num_pairs = num_access / 2;
        constexpr bool has_odd_element = (num_access % 2 == 1);

        // TODO: Enable also odd number of elements.
        static_assert(!has_odd_element, "wrong!Slice should have even number of elements.");
      
        ck::float2_t float2_buffer;
        static_for<0, num_pairs, 1>{}([&](auto i_pair) 
        {
            // First of the pair of the float values
            constexpr auto idx_1d_dst_0 = I2 * i_pair;
            constexpr index_t idx_1d_src_0 = SerpentineAccessPatternDst != SerpentineAccessPatternSrc
                ? LinearIndexFinder<SpaceFillingCurveSrc, SpaceFillingCurveDst, idx_1d_dst_0.value, 0, num_access>::find()
                : idx_1d_dst_0.value;
            static_assert(idx_1d_src_0 != index_t(-1), "wrong! Cannot find first linear index.");
            constexpr auto idx_md_src_0 = SpaceFillingCurveSrc::GetIndex(Number<idx_1d_src_0>{});
            constexpr index_t src_offset_0 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_src_0);

            // Second pf the pair of the float values
            constexpr auto idx_1d_dst_1 = I2 * i_pair + I1;
            constexpr index_t idx_1d_src_1 = SerpentineAccessPatternDst != SerpentineAccessPatternSrc
                ? LinearIndexFinder<SpaceFillingCurveSrc, SpaceFillingCurveDst, idx_1d_dst_1.value, 0, num_access>::find()
                : idx_1d_dst_1.value;
            static_assert(idx_1d_src_1 != index_t(-1), "wrong! Cannot find first linear index.");
            constexpr auto idx_md_src_1 = SpaceFillingCurveSrc::GetIndex(Number<idx_1d_src_1>{});
            constexpr index_t src_offset_1 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_src_1);

            if constexpr (src_offset_1 - src_offset_0 == 1)
            {
                // Load two consecutive float values from the src buffer
                float2_buffer = src_buf.template GetAsType<ck::float2_t>(Number<src_offset_0>{});
            }
            else 
            {
                // Load the two float values one by one
                float2_buffer= {src_buf[Number<src_offset_0>{}], src_buf[Number<src_offset_1>{}]};
            }

            const ck::bhalf2_t packed_value= bf16x2_convert_rne<ck::bhalf2_t, float>(float2_buffer[0], float2_buffer[1]);

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // Store the packed value into the dst buffer one by one.
            const auto dst_offset_0 = dst_coord_.GetOffset();
            dst_buf.template Update<DstInMemOp, ck::bhalf_t>(
                dst_offset_0,
                is_dst_valid,
                packed_value[0]);

            // Move to the next dst coordinate
            constexpr auto forward_step_0 = SpaceFillingCurveDst::GetForwardStep(idx_1d_dst_0);
                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step_0));

            const auto dst_offset_1 = dst_coord_.GetOffset();
            dst_buf.template Update<DstInMemOp, ck::bhalf_t>(
                dst_offset_1,
                is_dst_valid,
                packed_value[1]);
            
            // Move to next dst coordinate, unless this was the last pair
            if constexpr(i_pair.value != num_pairs - 1)
            {
                
                constexpr auto forward_step_1 = SpaceFillingCurveDst::GetForwardStep(idx_1d_dst_1);
                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step_1));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }


    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPatternDst>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }
private:
    DstCoord dst_coord_;
};

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3_buffered_packed_cast 
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr bool SerpentineAccessPattern = false;

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3_buffered_packed_cast(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation&)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");

        // Assert that elementwise op is pass through.
        static_assert(
            std::is_same_v<remove_cvref_t<ElementwiseOperation>, ck::tensor_operation::element_wise::PassThrough>,
            "wrong! ElementwiseOperation must be PassThrough");

        // For now, SrcData must be float and DstData must be ck::bhalf_t
        static_assert(std::is_same_v<SrcData, float>,
                      "wrong! SrcData must be float");
        static_assert(std::is_same_v<DstData, ck::bhalf_t>,
                      "wrong! DstData must be bhalf_t");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer>
    __device__ void RunRead(const SrcBuffer& src_buf)
    {

        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});
        constexpr auto src_desc = remove_cvref_t<SrcDesc>{};

        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>,
                                                    SerpentineAccessPattern>;
          
        static_assert(1 == SpaceFillingCurve::ScalarPerVector, "wrong!1 != SpaceFillingCurve::ScalarPerVector");

        constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();
        constexpr index_t num_pairs = num_access / 2;
        constexpr bool has_odd_element = (num_access % 2 == 1);

        // TODO: Enable also odd number of elements.
        static_assert(!has_odd_element, "wrong!Slice should have even number of elements.");
      
        static_for<0, num_pairs, 1>{}([&](auto i_pair) 
        {
            constexpr auto idx_1d_0 = I2 * i_pair;
            constexpr auto idx_1d_1 = I2 * i_pair + I1;
            constexpr auto idx_md_0 = SpaceFillingCurve::GetIndex(idx_1d_0);
            constexpr auto idx_md_1 = SpaceFillingCurve::GetIndex(idx_1d_1);
        
            constexpr index_t src_offset_0 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_0);
            constexpr index_t src_offset_1 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_1);

            const float val_0 = src_buf[Number<src_offset_0>{}];
            const float val_1 = src_buf[Number<src_offset_1>{}]; 

            const ck::bhalf2_t packed_value= bf16x2_convert_rne<ck::bhalf2_t, float>(val_0, val_1);

            // Store the packed value into the thread scratch buffer
            thread_scratch_.template SetAsType<ck::bhalf2_t>(Number<idx_1d_0>{}, packed_value);
        });
    }

    template <typename DstBuffer>
    __device__ void RunWrite(const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPattern>;
          
        static_assert(1 == SpaceFillingCurve::ScalarPerVector, "wrong!1 != SpaceFillingCurve::ScalarPerVector");

        constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();
        static_assert(num_access == buffer_size_, "wrong!num_access != buffer_size_");

        static_for<0, num_access, 1>{}([&](auto idx_1d) 
        {
            const ck::bhalf_t val = thread_scratch_.template GetAsType<ck::bhalf_t>(Number<idx_1d>{});
    
            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            dst_buf.template Update<DstInMemOp, ck::bhalf_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                val);

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        RunRead<SrcSliceOriginIdx,SrcBuffer>(src_buf);
        RunWrite(dst_desc, dst_buf);
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SerpentineAccessPattern>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    static constexpr index_t ScratchVectorSize = 2;

    __device__ static constexpr index_t GetBufferSize()
    {
        return reduce_on_sequence(SliceLengths{}, math::multiplies{}, Number<1>{});
    }
private:
    DstCoord dst_coord_;

    static constexpr auto buffer_size_ = GetBufferSize();
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                        ck::bhalf_t,
                                        buffer_size_ / ScratchVectorSize,
                                        ScratchVectorSize,
                                        true> thread_scratch_;
};

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3_vectorized
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr bool SnakedAccess = true;

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3_vectorized(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation&)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");

        // Assert that elementwise op is pass through.
        static_assert(
            std::is_same_v<remove_cvref_t<ElementwiseOperation>, ck::tensor_operation::element_wise::PassThrough>,
            "wrong! ElementwiseOperation must be PassThrough");

        // For now, SrcData must be float and DstData must be ck::bhalf_t
        static_assert(std::is_same_v<SrcData, float>,
                      "wrong! SrcData must be float");
        static_assert(std::is_same_v<DstData, ck::bhalf_t>,
                      "wrong! DstData must be bhalf_t");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SnakedAccess>;
          
        static_assert(2 == SpaceFillingCurve::ScalarPerVector, "wrong!2 != SpaceFillingCurve::ScalarPerVector");
        ck::bhalf2_t dst_vector;
        using dst_vector_t = ck::bhalf2_t;

        constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();
        static_for<0, num_access, 1>{}([&](auto idx_1d) 
        {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);
            constexpr auto idx_src_0 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md);
            constexpr auto idx_src_1 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md + src_scalar_step_in_vector);

            const float val_0 = src_buf[Number<idx_src_0>{}];
            const float val_1 = src_buf[Number<idx_src_1>{}]; 
            dst_vector = bf16x2_convert_rne<ck::bhalf2_t, float>(val_0, val_1);

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector);
            
            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);
                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }


    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>,
                                                    SnakedAccess>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }
private:
    DstCoord dst_coord_;
};

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation& element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx)),
          element_op_{element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        // TODO: Use SpaceFillingCurve::ScalarsPerAccess instread of DstScalarPerVector?
        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");
        typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;
        using dst_vector_t = typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            // copy data from src_buf into dst_vector
            // TODO: It's a hack here to use \p dst_scalar_step_in_vector. Use SpaceFillingCurve?
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                DstData v;

                // apply element-wise operation
                element_op_(v, src_buf[Number<src_offset>{}]);

                dst_vector.template AsType<DstData>()(i) = v;
            });

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    DstCoord dst_coord_;
    const ElementwiseOperation element_op_;
}; // namespace ThreadwiseTensorSliceTransfer_v1r3

/**
 * @brief Helper structure that facilitates transfer of source (grid) data to destination threads.
 *
 * @details The following assumptions are made:
 *  - For Source (Grid) Data:
 *     1. The source tensor descriptor SrcDesc is not known at compile-time.
 *     2. The source buffer is a dynamic buffer.
 *     3. The source slice origin index src_slice_origin_idx is not known at compile-time.
 *  - For Destination (Thread) Data:
 *     1. The destination tensor descriptor DstDesc is known at compile-time.
 *     2. The destination buffer dst_buf is a static buffer.
 *     3. The destination slice origin index dst_slice_origin_idx is known at compile-time.
 *
 * @tparam SrcData The data type of the source tensor.
 * @tparam DstData The data type of the destination tensor.
 * @tparam SrcDesc The descriptor type of the source tensor.
 * @tparam DstDesc The descriptor type of the destination tensor.
 * @tparam SliceLengths The lengths of the slice to be transferred.
 * @tparam DimAccessOrder The order of dimension access for the space-filling curve.
 * @tparam SrcVectorDim The dimension along which vectorized access is performed in the source
 * tensor.
 * @tparam SrcScalarPerVector The number of scalar elements per vector in the source tensor.
 * @tparam SrcScalarStrideInVector Not used.
 * @tparam SrcResetCoordinateAfterRun controls whether source coordinate is restored after each Run
 * or rolled back one step in MoveSrcSliceWindow
 * @tparam InvalidElementAsNaN Whether to fill invalid elements with NaN (only applicable for
 * floating-point types).
 *
 */
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t SrcScalarStrideInVector,
          bool SrcResetCoordinateAfterRun,
          bool InvalidElementAsNaN                                        = false,
          typename enable_if<DstDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v2
{
    static_assert((InvalidElementAsNaN && !ck::is_integral<DstData>::value) ||
                      (!InvalidElementAsNaN),
                  "Filling invalid element as NaN is only for floating point types");

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __device__ constexpr ThreadwiseTensorSliceTransfer_v2(const SrcDesc& src_desc,
                                                          const Index& src_slice_origin_idx)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin_idx))
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");

        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t> ||
                     is_same_v<remove_cvref_t<SrcData>, f4x2_pk_t>)
        {
            static_assert(SrcScalarPerVector % PackedSize == 0, "pk data N cannot be 1");
        }
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    template <typename SrcBuffer, typename DstBuffer, typename DstSliceOriginIdx>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf)
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! DstDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! DstSliceOrigin need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value &&
            "wrong! inconsistent type");

        // DstDesc and dst_slice_origin_idx are known at compile-time
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto dst_slice_origin_idx = DstSliceOriginIdx{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        // loop over tensor and copy
        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            typename vector_type_maker<SrcData, SrcScalarPerVector / PackedSize>::type src_vector;

            using src_vector_t =
                typename vector_type_maker<SrcData, SrcScalarPerVector / PackedSize>::type::type;
            constexpr auto src_data_idx = SpaceFillingCurve::GetIndex(idx_1d);

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf into src_vector
            src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset() / PackedSize,
                                                   is_src_valid);

            // copy data from src_vector into dst_buf
            static_for<0, SrcScalarPerVector / PackedSize, 1>{}([&](auto i) {
                constexpr index_t dst_offset =
                    dst_desc.CalculateOffset(to_multi_index(dst_slice_origin_idx) + src_data_idx +
                                             i * src_scalar_step_in_vector);

                if constexpr(InvalidElementAsNaN)
                {
                    dst_buf(Number<dst_offset>{}) =
                        is_src_valid
                            ? type_convert<DstData>(src_vector.template AsType<SrcData>()[i])
                            : NumericLimits<DstData>::QuietNaN();
                }
                else
                {
                    dst_buf(Number<dst_offset>{}) =
                        type_convert<DstData>(src_vector.template AsType<SrcData>()[i]);
                }
            });

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    src_desc, src_coord_, make_tensor_coordinate_step(src_desc, forward_step));
            }
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowStepHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowStepHack& src_move_slice_window_step_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(
            src_desc, adjusted_step_idx, src_move_slice_window_step_hack);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    private:
    SrcCoord src_coord_;
}; // namespace ck

template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t SrcScalarStrideInVector,
          bool SrcResetCoordinateAfterRun,
          index_t scale_gather_num,
          bool InvalidElementAsNaN                                        = false,
          typename enable_if<DstDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v2_gather
{
    static_assert((InvalidElementAsNaN && !ck::is_integral<DstData>::value) ||
                      (!InvalidElementAsNaN),
                  "Filling invalid element as NaN is only for floating point types");

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __device__ constexpr ThreadwiseTensorSliceTransfer_v2_gather(
        const SrcDesc& src_desc,
        const Index& src_slice_origin_idx,
        const StaticallyIndexedArray<index_t, scale_gather_num>& scale_gather_offsets)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin_idx)),
          scale_gather_offsets_(scale_gather_offsets)
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");

        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
        {
            static_assert(SrcScalarPerVector % PackedSize == 0, "pk data N cannot be 1");
        }
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        auto adjusted_origin_idx = [&]() {
            Index idx;

            static_for<0, nDim, 1>{}(
                [&](auto i) { idx(i) = i.value == 0 ? 0 : src_slice_origin_idx[Number<i>{}]; });

            return idx;
        }();

        src_coord_ = make_tensor_coordinate(src_desc, adjusted_origin_idx);
    }

    template <typename SrcBuffer, typename DstBuffer, typename DstSliceOriginIdx>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf)
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! DstDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! DstSliceOrigin need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value &&
            "wrong! inconsistent type");

        // DstDesc and dst_slice_origin_idx are known at compile-time
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto dst_slice_origin_idx = DstSliceOriginIdx{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        // loop over tensor and copy
        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, scale_gather_num, 1>{}([&](auto gather_idx) {
            constexpr auto current_dst_origin =
                to_multi_index(dst_slice_origin_idx) + make_multi_index(gather_idx, 0);

            static_for<0, num_access, 1>{}([&](auto idx_1d) {
                typename vector_type_maker<SrcData, SrcScalarPerVector / PackedSize>::type
                    src_vector;

                using src_vector_t =
                    typename vector_type_maker<SrcData,
                                               SrcScalarPerVector / PackedSize>::type::type;
                constexpr auto src_data_idx = SpaceFillingCurve::GetIndex(idx_1d);

                const bool is_src_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc,
                                                                                src_coord_);

                // copy data from src_buf into src_vector
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    src_buf.template Get<src_vector_t>(src_coord_.GetOffset() / PackedSize +
                                                           scale_gather_offsets_(gather_idx),
                                                       is_src_valid);

                // copy data from src_vector into dst_buf
                static_for<0, SrcScalarPerVector / PackedSize, 1>{}([&](auto i) {
                    constexpr index_t dst_offset =
                        dst_desc.CalculateOffset(to_multi_index(dst_slice_origin_idx) +
                                                 src_data_idx + i * src_scalar_step_in_vector);
                    constexpr auto full_dst_offset =
                        dst_desc.CalculateOffset(current_dst_origin) + dst_offset;

                    if constexpr(InvalidElementAsNaN)
                    {
                        dst_buf(full_dst_offset) =
                            is_src_valid
                                ? type_convert<DstData>(src_vector.template AsType<SrcData>()[i])
                                : NumericLimits<DstData>::QuietNaN();
                    }
                    else
                    {
                        dst_buf(Number<full_dst_offset>{}) =
                            type_convert<DstData>(src_vector.template AsType<SrcData>()[i]);
                    }
                });

                if constexpr(idx_1d.value != num_access - 1)
                {
                    constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                    move_tensor_coordinate(
                        src_desc, src_coord_, make_tensor_coordinate_step(src_desc, forward_step));
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowStepHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowStepHack& src_move_slice_window_step_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(
            src_desc, adjusted_step_idx, src_move_slice_window_step_hack);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    private:
    SrcCoord src_coord_;
    StaticallyIndexedArray<index_t, scale_gather_num> scale_gather_offsets_;
}; // namespace ck

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
template <typename SliceLengths,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
struct ThreadwiseTensorSliceTransfer_v3
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3(const SrcDesc& src_desc,
                                                          const Index& src_slice_origin,
                                                          const DstDesc& dst_desc,
                                                          const Index& dst_slice_origin)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin))
    {
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, typename SrcStepHacks>
    __device__ void
    RunRead(const SrcDesc& src_desc, const SrcBuffer& src_buf, const SrcStepHacks& src_step_hacks)
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value,
            "wrong! SrcBuffer and SrcData data type are inconsistent");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward steps
        const auto src_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(
                    src_desc, forward_step_idx, src_step_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward steps
        const auto src_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(
                    src_desc, backward_step_idx, src_step_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_src_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate src data index
            constexpr auto src_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_src_access_idx[i]
                                                      : ordered_src_access_lengths[i] - 1 -
                                                            ordered_src_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                       src_scalar_per_access;
            }();

            vector_type_maker_t<SrcData, SrcScalarPerVector> src_tmp_vector;

            using src_vector_t = typename decltype(src_tmp_vector)::type;

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf to src_tmp_vector
            src_tmp_vector.template AsType<src_vector_t>()(Number<0>{}) =
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid);

            // copy data from src_tmp_vector to buffer_
            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(src_data_idx + i * src_scalar_step_in_vector);

                buffer_(Number<buffer_offset>{}) = src_tmp_vector.template AsType<SrcData>()[i];
            });

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_src_access_idx[i] < ordered_src_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_src_access_idx[j] == ordered_src_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_forward_steps[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_backward_steps[src_dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    template <typename DstBuffer, typename DstStepHacks>
    __device__ void
    RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf, const DstStepHacks& dst_step_hacks)
    {
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // src scalar per access on each dim
        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward steps
        const auto dst_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(
                    dst_desc, forward_step_idx, dst_step_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward steps
        const auto dst_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(
                    dst_desc, backward_step_idx, dst_step_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_idx[i]
                                                      : ordered_dst_access_lengths[i] - 1 -
                                                            ordered_dst_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                       dst_scalar_per_access;
            }();

            vector_type_maker_t<DstData, DstScalarPerVector> dst_tmp_vector;

            // copy data from buffer_ to dst_tmp_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(dst_data_idx + i * dst_scalar_step_in_vector);

                dst_tmp_vector.template AsType<DstData>()(i) =
                    type_convert<DstData>(buffer_[Number<buffer_offset>{}]);
            });

            using dst_vector_t = typename decltype(dst_tmp_vector)::type;

            // copy data from dst_tmp_vector to dst_buf
            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_tmp_vector.template AsType<dst_vector_t>()[Number<0>{}]);

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_dst_access_idx[i] < ordered_dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_dst_access_idx[j] == ordered_dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_forward_steps[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_backward_steps[dst_dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    template <typename SrcBuffer>
    __device__ void RunRead(const SrcDesc& src_desc, const SrcBuffer& src_buf)
    {
        constexpr index_t ntransform_src = SrcDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_src, 0>::type{};

        constexpr auto src_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunRead(src_desc, src_buf, src_step_hacks);
    }

    template <typename DstBuffer>
    __device__ void RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunWrite(dst_desc, dst_buf, dst_step_hacks);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_src_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate src data index after last iteration in RunRead(), if it has not being reset by
        // RunRead()
        constexpr auto src_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_src_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                   src_scalar_per_access;
        }();

        //
        constexpr auto reset_src_data_step = [&]() {
            Index reset_src_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_src_data_step_(i) = -src_data_idx[i]; });

            return reset_src_data_step_;
        }();

        return reset_src_data_step;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_dst_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate dst data index after last iteration in RunWrite(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                   dst_scalar_per_access;
        }();

        //
        constexpr auto reset_dst_data_step = [&]() {
            Index reset_dst_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_dst_data_step_(i) = -dst_data_idx[i]; });

            return reset_dst_data_step_;
        }();

        return reset_dst_data_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowStepHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowStepHack& src_move_slice_window_step_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(
            src_desc, adjusted_step_idx, src_move_slice_window_step_hack);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }
    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    static constexpr auto buffer_desc_ =
        make_naive_tensor_descriptor_packed(sequence_to_tuple_of_number(SliceLengths{}));

    static constexpr auto buffer_size_ = buffer_desc_.GetElementSpaceSize();

    StaticBuffer<AddressSpaceEnum::Vgpr, SrcData, buffer_size_, true> buffer_;

    SrcCoord src_coord_;
    DstCoord dst_coord_;
};

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is DynamicBuffer
//     3. src_ref_idx is known at run-time
//     4. SrcRefToOriginDisplacement is known at compile-time
//     5. use #-step
//   2. dst:
//     1. DstDesc is known at compile-time
//     2. DstBuffer is StaticBuffer
//     3. DstOriginIdx is known at compile-time
//     4. use direct address calculation
//   3. vector access on src
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t SrcScalarStrideInVector,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v4
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __device__ constexpr ThreadwiseTensorSliceTransfer_v4(const Index& src_ref_idx)
        : src_ref_coord_(make_tensor_coordinate(SrcDesc{}, src_ref_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t> ||
                     is_same_v<remove_cvref_t<SrcData>, f4x2_pk_t>)
        {
            static_assert(SrcScalarPerVector % PackedSize == 0, "pk data N cannot be 1");
        }
    }

    template <typename SrcRefToOriginDisplacement,
              typename DstOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcRefToOriginDisplacement&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value &&
                is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        static_assert(DstBuffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcRefToOriginDisplacement>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstOriginIdx>>::value,
                      "wrong! SrcOriginToRefDistance and DstOriginToRefDistance need to be known "
                      "at compile-time");

        // SrcDesc and DstDesc are known at compile-time
        constexpr auto src_desc = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc = remove_cvref_t<DstDesc>{};

        // SrcOriginToRefDisttance and DstOriginToRefDistance are known at compile-time
        constexpr auto src_ref_to_origin_disp_idx = to_multi_index(SrcRefToOriginDisplacement{});
        constexpr auto dst_origin_idx             = to_multi_index(DstOriginIdx{});

        // scalar per access of each dim
        constexpr auto src_scalar_per_access = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<SrcScalarPerVector>{};
                }
                else
                {
                    return Number<1>{};
                }
            },
            Number<nDim>{});

        // scalar step (if steping on SrcVectorDim) of each dim
        constexpr auto src_scalar_step_in_vector = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<1>{};
                }
                else
                {
                    return Number<0>{};
                }
            },
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
#if 0
            // TODO: unable to compile
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                container_reorder_given_old2new(ordered_access_idx, dim_access_order) *
                src_scalar_per_access;
#else
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                ordered_access_idx.ReorderGivenOld2New(dim_access_order) * src_scalar_per_access;
#endif
            // src coordinate
            constexpr auto src_ref_to_data_disp_idx =
                src_ref_to_origin_disp_idx + data_to_origin_disp_idx;

            constexpr auto src_ref_to_data_disp_coord_step =
                make_tensor_coordinate_step(src_desc, src_ref_to_data_disp_idx);

            auto src_data_coord = src_ref_coord_;

            move_tensor_coordinate(src_desc, src_data_coord, src_ref_to_data_disp_coord_step);

            vector_type_maker_t<SrcData, SrcScalarPerVector / PackedSize> src_tmp_vector;

            using src_vector_t = typename decltype(src_tmp_vector)::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_data_coord);

            // copy data from src_buf into src_tmp_vector
            if constexpr(SrcBuffer::IsDynamicBuffer())
            {
                src_tmp_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    src_buf.template Get<src_vector_t>(src_data_coord.GetOffset() / PackedSize,
                                                       is_src_valid);
            }
            else if constexpr(SrcBuffer::IsStaticBuffer())
            {
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t src_offset = src_desc.CalculateOffset(
                        src_ref_to_origin_disp_idx + data_to_origin_disp_idx +
                        i * src_scalar_step_in_vector);

                    src_tmp_vector.template AsType<SrcData>()(i) = src_buf[Number<src_offset>{}];
                });
            }

            if constexpr(is_same<remove_cvref_t<SrcData>, pk_i4_t>::value)
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;

                constexpr index_t pack_size = 8;

                static_assert(SrcScalarPerVector % pack_size == 0, "");

                using src_v_t = typename vector_type_maker_t<SrcData, pack_size / PackedSize>::type;
                using dst_v_t = typename vector_type_maker_t<DstData, pack_size>::type;

                static_for<0, SrcScalarPerVector / pack_size, 1>{}([&](auto i) {
                    ck::tensor_operation::element_wise::PassThroughPack8{}(
                        dst_tmp_vector.template AsType<dst_v_t>()(i),
                        src_tmp_vector.template AsType<src_v_t>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
            else if constexpr(is_same<remove_cvref_t<SrcData>, f8_t>::value &&
                              is_same<remove_cvref_t<DstData>, half_t>::value &&
                              SrcScalarPerVector % 2 == 0)
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;

                constexpr index_t pack_size = 2;

                using dst_v_t = typename vector_type_maker_t<DstData, pack_size>::type;
                using src_v_t = typename vector_type_maker_t<SrcData, pack_size>::type;
                static_for<0, SrcScalarPerVector / pack_size, 1>{}([&](auto i) {
                    ck::tensor_operation::element_wise::PassThroughPack2{}(
                        dst_tmp_vector.template AsType<dst_v_t>()(i),
                        src_tmp_vector.template AsType<src_v_t>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
            else
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector / PackedSize> dst_tmp_vector;

                // TODO: if SrcData and DstData are vetor type, then static_cast may not compile
                static_for<0, SrcScalarPerVector / PackedSize, 1>{}([&](auto i) {
                    dst_tmp_vector.template AsType<DstData>()(i) =
                        type_convert<DstData>(src_tmp_vector.template AsType<SrcData>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector / PackedSize, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
        });
    }

    // Fuse scale
    template <typename SrcRefToOriginDisplacement,
              typename DstOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcRefToOriginDisplacement&,
                        const SrcBuffer& src_buf,
                        const DstData& scale,
                        const DstDesc&,
                        const DstOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value &&
                is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        static_assert(DstBuffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcRefToOriginDisplacement>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstOriginIdx>>::value,
                      "wrong! SrcOriginToRefDistance and DstOriginToRefDistance need to be known "
                      "at compile-time");

        // SrcDesc and DstDesc are known at compile-time
        constexpr auto src_desc = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc = remove_cvref_t<DstDesc>{};

        // SrcOriginToRefDisttance and DstOriginToRefDistance are known at compile-time
        constexpr auto src_ref_to_origin_disp_idx = to_multi_index(SrcRefToOriginDisplacement{});
        constexpr auto dst_origin_idx             = to_multi_index(DstOriginIdx{});

        // scalar per access of each dim
        constexpr auto src_scalar_per_access = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<SrcScalarPerVector>{};
                }
                else
                {
                    return Number<1>{};
                }
            },
            Number<nDim>{});

        // scalar step (if steping on SrcVectorDim) of each dim
        constexpr auto src_scalar_step_in_vector = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<1>{};
                }
                else
                {
                    return Number<0>{};
                }
            },
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
#if 0
            // TODO: unable to compile
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                container_reorder_given_old2new(ordered_access_idx, dim_access_order) *
                src_scalar_per_access;
#else
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                ordered_access_idx.ReorderGivenOld2New(dim_access_order) * src_scalar_per_access;
#endif
            // src coordinate
            constexpr auto src_ref_to_data_disp_idx =
                src_ref_to_origin_disp_idx + data_to_origin_disp_idx;

            constexpr auto src_ref_to_data_disp_coord_step =
                make_tensor_coordinate_step(src_desc, src_ref_to_data_disp_idx);

            auto src_data_coord = src_ref_coord_;

            move_tensor_coordinate(src_desc, src_data_coord, src_ref_to_data_disp_coord_step);

            vector_type_maker_t<SrcData, SrcScalarPerVector / PackedSize> src_tmp_vector;

            using src_vector_t = typename decltype(src_tmp_vector)::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_data_coord);

            // copy data from src_buf into src_tmp_vector
            if constexpr(SrcBuffer::IsDynamicBuffer())
            {
                src_tmp_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    src_buf.template Get<src_vector_t>(src_data_coord.GetOffset() / PackedSize,
                                                       is_src_valid);
            }
            else if constexpr(SrcBuffer::IsStaticBuffer())
            {
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t src_offset = src_desc.CalculateOffset(
                        src_ref_to_origin_disp_idx + data_to_origin_disp_idx +
                        i * src_scalar_step_in_vector);

                    src_tmp_vector.template AsType<SrcData>()(i) = src_buf[Number<src_offset>{}];
                });
            }

            if constexpr(is_same<remove_cvref_t<SrcData>, pk_i4_t>::value)
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;
                vector_type<DstData, 2> scale_vector;
                scale_vector.template AsType<DstData>()(Number<0>{}) = scale;
                scale_vector.template AsType<DstData>()(Number<1>{}) = scale;

                constexpr index_t pack_size = 8;

                static_assert(SrcScalarPerVector % pack_size == 0, "");

                using src_v_t = typename vector_type_maker_t<SrcData, pack_size / PackedSize>::type;
                using dst_v_t = typename vector_type_maker_t<DstData, pack_size>::type;
                using scale_v_t = typename vector_type_maker_t<DstData, 2>::type;

                static_for<0, SrcScalarPerVector / pack_size, 1>{}([&](auto i) {
                    ck::tensor_operation::element_wise::DequantPack8{}(
                        dst_tmp_vector.template AsType<dst_v_t>()(i),
                        src_tmp_vector.template AsType<src_v_t>()[i],
                        scale_vector.template AsType<scale_v_t>()[Number<0>{}]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
            else if constexpr(is_same<remove_cvref_t<SrcData>, f8_t>::value &&
                              is_same<remove_cvref_t<DstData>, half_t>::value &&
                              SrcScalarPerVector % 2 == 0)
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;

                constexpr index_t pack_size = 2;

                using dst_v_t = typename vector_type_maker_t<DstData, pack_size>::type;
                using src_v_t = typename vector_type_maker_t<SrcData, pack_size>::type;
                static_for<0, SrcScalarPerVector / pack_size, 1>{}([&](auto i) {
                    ck::tensor_operation::element_wise::PassThroughPack2{}(
                        dst_tmp_vector.template AsType<dst_v_t>()(i),
                        src_tmp_vector.template AsType<src_v_t>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
            else
            {
                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;

                // TODO: if SrcData and DstData are vetor type, then static_cast may not compile
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    dst_tmp_vector.template AsType<DstData>()(i) =
                        type_convert<DstData>(src_tmp_vector.template AsType<SrcData>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            }
        });
    }

    template <typename SrcSliceMoveStepIdx>
    __device__ void MoveSrcSliceWindow(const SrcDesc&,
                                       const SrcSliceMoveStepIdx& src_slice_move_step_idx)
    {
        constexpr auto src_desc = SrcDesc{};

        const auto src_slice_move_step_iter =
            make_tensor_coordinate_step(src_desc, to_multi_index(src_slice_move_step_idx));

        move_tensor_coordinate(SrcDesc{}, src_ref_coord_, src_slice_move_step_iter);
    }
    __device__ void SetSrcCoord(const Index& src_ref_idx)
    {
        src_ref_coord_ = make_tensor_coordinate(SrcDesc{}, src_ref_idx);
    }

    private:
    SrcCoord src_ref_coord_;
};

/**
 * @brief Threadwise data transfer
 *
 * Do NOT involve any tensor coordinates with StaticBuffer
 *
 */
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_StaticToStatic
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __device__ constexpr ThreadwiseTensorSliceTransfer_StaticToStatic(
        const ElementwiseOperation& element_op)
        : element_op_{element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    template <typename SrcSliceOriginIdx,
              typename DstSliceOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! SliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer() && DstBuffer::IsStaticBuffer(),
                      "wrong! Buffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});
        constexpr auto dst_slice_origin_idx = to_multi_index(DstSliceOriginIdx{});

        // scalar per access on each dim
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        if constexpr(is_same<remove_cvref_t<SrcData>, pk_i4_t>::value)
        {
            static_for<0, num_access, 1>{}([&](auto idx_1d) {
                typename vector_type_maker<SrcData, DstScalarPerVector / PackedSize>::type
                    src_tmp_vector;

                constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

                // copy data from src_buf into dst_vector
                static_for<0, DstScalarPerVector / PackedSize, 1>{}([&](auto i) {
                    constexpr index_t src_offset = src_desc.CalculateOffset(
                        src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                    src_tmp_vector.template AsType<SrcData>()(i) = src_buf[Number<src_offset>{}];
                });

                // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
                // DstData)
                vector_type_maker_t<DstData, DstScalarPerVector> dst_tmp_vector;

                constexpr index_t pack_size = 8;

                static_assert(DstScalarPerVector % pack_size == 0, "");

                using src_v_t = typename vector_type_maker_t<SrcData, pack_size / PackedSize>::type;
                using dst_v_t = typename vector_type_maker_t<DstData, pack_size>::type;

                static_for<0, DstScalarPerVector / pack_size, 1>{}([&](auto i) {
                    ck::tensor_operation::element_wise::PassThroughPack8{}(
                        dst_tmp_vector.template AsType<dst_v_t>()(i),
                        src_tmp_vector.template AsType<src_v_t>()[i]);
                });

                // copy data from dst_tmp_vector into dst_buf
                static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                    dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
                });
            });
        }
        else
        {
            static_for<0, num_access, 1>{}([&](auto idx_1d) {
                constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

                // copy data from src_buf into dst_vector
                static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t src_offset = src_desc.CalculateOffset(
                        src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                    constexpr index_t dst_offset = dst_desc.CalculateOffset(
                        dst_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                    DstData v;

                    // apply element-wise operation
                    element_op_(v, src_buf[Number<src_offset>{}]);

                    // apply type convert
                    dst_buf(Number<dst_offset>{}) = v;
                });
            });
        }
    }

    ElementwiseOperation element_op_;
};

// Specialized for gfx11
// A single Wave32 is composed by double row
// Data exchange allowed between these two rows
// This RowLane Dst buf will be filled from two Src buf
// SrcA:    From specific thread buffer hold by This RowLane on This Row
// SrcB:    From specific thread buffer hold by This RowLane on The other Row
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          uint32_t LowEightRowlaneIdx,
          uint32_t HighEightRowLaneIdx,
          bool IntraRowSwizzlePerm,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow(const Index& src_idx)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
        ignore = src_idx;
    }

    template <typename SrcSliceOriginIdx,
              typename DstSliceOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! SliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer() && DstBuffer::IsStaticBuffer(),
                      "wrong! Buffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});
        constexpr auto dst_slice_origin_idx = to_multi_index(DstSliceOriginIdx{});

        // scalar per access on each dim
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            // copy data from src_buf into dst_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                // src_desc error, non constexpr, caused by merge transform
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                constexpr index_t dst_offset = dst_desc.CalculateOffset(
                    dst_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                SrcData v_this_row, v_theother_row;
                // int type temp value due to intrinsic requirement
                int temp = 0;

                // apply element-wise operation
                element_op_(v_this_row, src_buf[Number<src_offset>{}]);

                // apply intra-row permute.
                if constexpr(IntraRowSwizzlePerm)
                {
                    temp = __builtin_amdgcn_permlane16(
                        temp, type_convert_sp<int>(v_this_row), 0xb3a29180, 0xf7e6d5c4, 1, 0);
                    v_this_row = type_convert_sp<SrcData>(temp);
                }

                // apply inter-row permute.
                temp           = __builtin_amdgcn_permlanex16(temp,
                                                    type_convert_sp<int>(v_this_row),
                                                    LowEightRowlaneIdx,
                                                    HighEightRowLaneIdx,
                                                    1,
                                                    0);
                v_theother_row = type_convert_sp<SrcData>(temp);

                if(get_thread_local_1d_id() % 32 < 16)
                {
                    // apply type convert
                    dst_buf(Number<dst_offset>{}) = type_convert_sp<DstData>(v_this_row);
                    dst_buf(Number<dst_offset + DstScalarPerVector>{}) =
                        type_convert_sp<DstData>(v_theother_row);
                }
                else
                {
                    // apply type convert
                    dst_buf(Number<dst_offset + DstScalarPerVector>{}) =
                        type_convert_sp<DstData>(v_this_row);
                    dst_buf(Number<dst_offset>{}) = type_convert_sp<DstData>(v_theother_row);
                }
            });
        });
    }
    ElementwiseOperation element_op_{};
};

// Specialized for gfx12
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          bool IntraRowSwizzlePerm,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow(const Index& src_idx)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
        ignore = src_idx;
    }

    template <typename SrcSliceOriginIdx,
              typename DstSliceOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! SliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer() && DstBuffer::IsStaticBuffer(),
                      "wrong! Buffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});
        constexpr auto dst_slice_origin_idx = to_multi_index(DstSliceOriginIdx{});

        // scalar per access on each dim
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            // copy data from src_buf into dst_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                // src_desc error, non constexpr, caused by merge transform
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                constexpr index_t dst_offset = dst_desc.CalculateOffset(
                    dst_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                SrcData v_this_row;
                // int type temp value due to intrinsic requirement
                int temp = 0;

                // apply element-wise operation
                element_op_(v_this_row, src_buf[Number<src_offset>{}]);

                // apply intra-row permute.
                if constexpr(IntraRowSwizzlePerm)
                {
                    temp = __builtin_amdgcn_permlane16(
                        temp, type_convert_sp<int>(v_this_row), 0xb3a29180, 0xf7e6d5c4, 1, 0);
                    v_this_row = type_convert_sp<SrcData>(temp);
                }

                // apply type convert
                dst_buf(Number<dst_offset>{}) = type_convert_sp<DstData>(v_this_row);
            });
        });
    }
    ElementwiseOperation element_op_{};
};

} // namespace ck
