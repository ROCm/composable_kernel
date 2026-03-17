// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor/static_tensor.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

namespace ck {

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
//   5. Dequantization happened between read and write.
template <typename SliceLengths,
          typename ScaleSliceLengths,
          typename SrcElementwiseOperation,
          typename ScaleElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename ScaleData,
          typename DstData,
          typename SrcDesc,
          typename ScaleDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t ScaleScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t ScaleScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun, // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r1_dequant
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using Helper = ThreadwiseTransferHelper_Serpentine;

    using SrcCoord   = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using ScaleCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord   = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r1_dequant(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const ScaleDesc& scale_desc,
        const Index& scale_slice_origin,
        const ScaleElementwiseOperation& scale_element_op,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          scale_coord_(make_tensor_coordinate(scale_desc, scale_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          src_element_op_(src_element_op),
          scale_element_op_(scale_element_op),
          dst_element_op_(dst_element_op)
    {
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetScaleSliceOrigin(const ScaleDesc& scale_desc,
                                        const Index& scale_slice_origin_idx)
    {
        scale_coord_ = make_tensor_coordinate(scale_desc, scale_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcBuffer& src_buf,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value,
            "wrong! SrcBuffer and SrcData data type are inconsistent");

        // scalar per access on each dim
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward and backward steps
        const auto src_forward_steps = Helper::ComputeForwardSteps(src_desc, src_scalar_per_access);
        const auto src_backward_steps =
            Helper::ComputeBackwardSteps(src_desc, src_scalar_per_access);

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            constexpr auto forward_sweep =
                Helper::ComputeForwardSweep(ordered_src_access_idx, ordered_src_access_lengths);

            constexpr auto src_data_idx = Helper::ComputeDataIndex(ordered_src_access_idx,
                                                                   ordered_src_access_lengths,
                                                                   forward_sweep,
                                                                   src_dim_access_order,
                                                                   src_scalar_per_access);

            constexpr auto src_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<src_data_idx[i]>{}; }, Number<src_data_idx.Size()>{});

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            // copy data from src_buf into src_vector_container
            auto src_vector_container = src_vector_type{
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid)};

            // copy data from src_vector_container into src_thread_scratch_
            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<src_vector_t>(
                    src_data_idx_seq,
                    src_vector_container.template AsType<src_vector_t>()[Helper::I0]);

            constexpr auto move_on_dim =
                Helper::ComputeMoveOnDim(ordered_src_access_idx, ordered_src_access_lengths);

            // move src coord
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

    template <typename ScaleBuffer>
    __device__ void RunScaleRead(const ScaleDesc& scale_desc, const ScaleBuffer& scale_buf)
    {
        static_assert(ScaleBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          ScaleBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename ScaleBuffer::type>, remove_cvref_t<ScaleData>>::value,
            "wrong! ScaleBuffer and ScaleData data type are inconsistent");

        // scalar per access on each dim
        constexpr auto scale_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, ScaleScalarPerVector>{}, Number<nDim>{});

        constexpr auto scale_access_lengths = SliceLengths{} / scale_scalar_per_access;

        constexpr auto scale_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_scale_access_lengths =
            container_reorder_given_new2old(scale_access_lengths, scale_dim_access_order);

        // make forward and backward steps
        const auto scale_forward_steps =
            Helper::ComputeForwardSteps(scale_desc, scale_scalar_per_access);
        const auto scale_backward_steps =
            Helper::ComputeBackwardSteps(scale_desc, scale_scalar_per_access);

        // loop over tensor and copy
        static_ford<decltype(ordered_scale_access_lengths)>{}([&](auto ordered_scale_access_idx) {
            constexpr auto forward_sweep =
                Helper::ComputeForwardSweep(ordered_scale_access_idx, ordered_scale_access_lengths);

            constexpr auto scale_data_idx = Helper::ComputeDataIndex(ordered_scale_access_idx,
                                                                     ordered_scale_access_lengths,
                                                                     forward_sweep,
                                                                     scale_dim_access_order,
                                                                     scale_scalar_per_access);

            constexpr auto scale_data_idx_seq =
                generate_sequence_v2([&](auto i) { return Number<scale_data_idx[i]>{}; },
                                     Number<scale_data_idx.Size()>{});

            const bool is_scale_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                scale_desc, scale_coord_);

            using scale_vector_type = vector_type_maker_t<ScaleData, ScaleScalarPerVector>;
            using scale_vector_t    = typename scale_vector_type::type;

            // copy data from scale_buf into scale_vector_container
            auto scale_vector_container = scale_vector_type{
                scale_buf.template Get<scale_vector_t>(scale_coord_.GetOffset(), is_scale_valid)};

            // copy data from scale_vector_container into scale_thread_scratch_
            scale_thread_scratch_.template SetAsType<scale_vector_t>(
                scale_data_idx_seq,
                scale_vector_container.template AsType<scale_vector_t>()[Helper::I0]);

            constexpr auto move_on_dim =
                Helper::ComputeMoveOnDim(ordered_scale_access_idx, ordered_scale_access_lengths);

            // move scale coord
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(scale_desc,
                                               scale_coord_,
                                               scale_forward_steps[scale_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(scale_desc,
                                               scale_coord_,
                                               scale_backward_steps[scale_dim_access_order[i]]);
                    }
                }
            });
        });
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
#if !CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        static_ford<SliceLengths>{}([&](auto idx) {
            // convert from SrcData to DstData here
            dst_thread_scratch_(idx) =
                type_convert<DstData>(src_thread_scratch_tuple_[thread_scratch_id][idx]);
        });
#else
        // sub-dword transpose between src_thread_scratch_ and dst_thread_scratch_
        // TODO make this logic more generic for more sub-dword datatype
        if constexpr(SrcVectorDim != DstVectorDim &&
                     ((is_same<half_t, remove_cvref_t<SrcData>>::value &&
                       is_same<half_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 2 == 0 && DstScalarPerVector % 2 == 0) ||
                      (is_same<int8_t, remove_cvref_t<SrcData>>::value &&
                       is_same<int8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0)))
        {
            // each transpose does
            // DstScalarPerVector # of src vectors in src_thread_scratch_
            // SrcScalarPerVector # of dst vectors in dst_thread_scratch_
            constexpr index_t num_src_vector = Number<DstScalarPerVector>{};
            constexpr index_t num_dst_vector = Number<SrcScalarPerVector>{};

            // Assume SrcVectorDim is not the same as DstVectorDim, so we do transpose
            // TODO: make this logic generic for all scenario
            static_assert(SrcVectorDim != DstVectorDim, "wrong");

            constexpr auto src_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

            constexpr auto dst_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

            constexpr auto scalar_per_access = generate_sequence(
                detail::lambda_scalar_per_access_for_src_and_dst<SrcVectorDim,
                                                                 SrcScalarPerVector,
                                                                 DstVectorDim,
                                                                 DstScalarPerVector>{},
                Number<nDim>{});

            constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

            static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
                constexpr auto data_idx = access_idx * scalar_per_access;

                constexpr auto data_idx_seq = generate_sequence_v2(
                    [&](auto i) { return Number<data_idx[i]>{}; }, Number<nDim>{});

                using src_vector_t = vector_type_maker_t<SrcData, SrcScalarPerVector>;
                using dst_vector_t = vector_type_maker_t<DstData, DstScalarPerVector>;

                // get DstScalarPerVector # of read-only references to src vectors from
                // src_thread_scratch_
                const auto src_vector_refs = generate_tie(
                    [&](auto i) -> const src_vector_t& {
                        // i increment corresponds to movement in DstVectorDim
                        return src_thread_scratch_tuple_[thread_scratch_id].GetVectorTypeReference(
                            data_idx_seq + i * dst_scalar_step_in_vector);
                    },
                    Number<num_src_vector>{});

                // get SrcScalarPerVector # of references to dst vectors from dst_thread_scratch_
                auto dst_vector_refs = generate_tie(
                    [&](auto i) -> dst_vector_t& {
                        // i increment corresponds to movement in SrcVectorDim
                        return dst_thread_scratch_.GetVectorTypeReference(
                            data_idx_seq + i * src_scalar_step_in_vector);
                    },
                    Number<num_dst_vector>{});

                // do data transpose
                transpose_vectors<SrcData, DstScalarPerVector, SrcScalarPerVector>{}(
                    src_vector_refs, dst_vector_refs);
            });
        }

        // Do fast numeric convert
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access_for_src_and_dst<SrcVectorDim,
                                                             SrcScalarPerVector,
                                                             DstVectorDim,
                                                             DstScalarPerVector>{},
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
        using src_vector_t    = typename src_vector_type::type;

        using src_converted_vector_type = vector_type_maker_t<DstData, SrcScalarPerVector>;
        using src_converted_vector_t    = typename src_converted_vector_type::type;
        // Vector-wise type convert
        static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
            auto src_vector_container = src_vector_type{
                src_thread_scratch_tuple_[thread_scratch_id].template GetAsType<src_vector_t>(
                    access_idx)};

            auto src_converted_vector_container =
                src_converted_vector_type{fast_numeric_converter(src_vector_container)};

            src_converted_thread_scratch_.template SetAsType<src_converted_vector_t>(
                access_idx,
                src_converted_vector_container
                    .template AsType<src_converted_vector_t>()[Helper::I0]);
        });

        // Element-scale operation, expect packed multiplication
        static_ford<SliceLengths>{}([&](auto idx) {
            DstData dst_v;
            constexpr auto scale_idx = Sequence<Helper::I0, idx.At(1), Helper::I0>{};
            src_element_op_(dst_v,
                            src_converted_thread_scratch_[idx] * scale_thread_scratch_[scale_idx]);
            dst_thread_scratch_(idx) = dst_v;
        });
#endif
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // if there is transpose, it's done here
        // TODO move this elsewhere
        TransferDataFromSrcThreadScratchToDstThreadScratch(thread_scratch_id);

        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        // src scalar per access on each dim
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward and backward steps
        const auto dst_forward_steps = Helper::ComputeForwardSteps(dst_desc, dst_scalar_per_access);
        const auto dst_backward_steps =
            Helper::ComputeBackwardSteps(dst_desc, dst_scalar_per_access);

        // loop over tensor and copy
        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            constexpr auto forward_sweep =
                Helper::ComputeForwardSweep(ordered_dst_access_idx, ordered_dst_access_lengths);

            constexpr auto dst_data_idx = Helper::ComputeDataIndex(ordered_dst_access_idx,
                                                                   ordered_dst_access_lengths,
                                                                   forward_sweep,
                                                                   dst_dim_access_order,
                                                                   dst_scalar_per_access);

            constexpr auto dst_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<dst_data_idx[i]>{}; }, Number<dst_data_idx.Size()>{});

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            using dst_vector_type = vector_type_maker_t<DstData, DstScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            // copy data from dst_thread_scratch_ into dst_vector_container
            auto dst_vector_container = dst_vector_type{
                dst_thread_scratch_.template GetAsType<dst_vector_t>(dst_data_idx_seq)};

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                DstData dst_v;

                // apply DstElementwiseOperation
                dst_element_op_(dst_v, dst_vector_container.template AsType<DstData>()[i]);

                dst_vector_container.template AsType<DstData>()(i) = dst_v;
            });

            // copy data from dst_vector_container to dst_buf
            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>()[Helper::I0]);

            constexpr auto move_on_dim =
                Helper::ComputeMoveOnDim(ordered_dst_access_idx, ordered_dst_access_lengths);

            // move dst coord
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

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        return Helper::ComputeCoordinateResetStep<SliceLengths,
                                                  SrcVectorDim,
                                                  SrcScalarPerVector,
                                                  SrcDimAccessOrder>();
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        return Helper::ComputeCoordinateResetStep<SliceLengths,
                                                  DstVectorDim,
                                                  DstScalarPerVector,
                                                  DstDimAccessOrder>();
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        Helper::MoveSliceWindow<SrcDesc, SrcCoord, SrcResetCoordinateAfterRun>(
            src_desc, src_coord_, src_slice_origin_step_idx, GetSrcCoordinateResetStep);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        Helper::MoveSliceWindow<DstDesc, DstCoord, DstResetCoordinateAfterRun>(
            dst_desc, dst_coord_, dst_slice_origin_step_idx, GetDstCoordinateResetStep);
    }

    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        return Helper::
            ComputeThreadScratchDescriptor<SliceLengths, SrcVectorDim, SrcScalarPerVector>();
    }

    __device__ static constexpr auto GetScaleThreadScratchDescriptor()
    {
        return Helper::
            ComputeThreadScratchDescriptor<SliceLengths, SrcVectorDim, ScaleScalarPerVector>();
    }

    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        return Helper::
            ComputeThreadScratchDescriptor<SliceLengths, DstVectorDim, DstScalarPerVector>();
    }

    private:
    static constexpr auto src_thread_scratch_desc_ = decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto scale_thread_scratch_desc_ =
        decltype(GetScaleThreadScratchDescriptor()){};
    static constexpr auto dst_thread_scratch_desc_ = decltype(GetDstThreadScratchDescriptor()){};

    // Registers, contain raw data loaded from global buffer
    using SrcThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             SrcData,
                                                             SrcScalarPerVector,
                                                             decltype(src_thread_scratch_desc_),
                                                             true>;

    // Registers, contain fast converted data
    using SrcThreadConvertedScratch =
        StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                        DstData,
                                        SrcScalarPerVector,
                                        decltype(src_thread_scratch_desc_),
                                        true>;

    // Registers, contain scale data
    using ScaleThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                               ScaleData,
                                                               ScaleScalarPerVector,
                                                               decltype(scale_thread_scratch_desc_),
                                                               true>;

    // Registers, contain dequantized data
    using DstThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             DstScalarPerVector,
                                                             decltype(dst_thread_scratch_desc_),
                                                             true>;

    using FastTypeConverter = tensor_operation::element_wise::
        FastNumericArrayConverter<SrcData, DstData, SrcScalarPerVector>;

    StaticallyIndexedArray<SrcThreadScratch, NumThreadScratch> src_thread_scratch_tuple_;
    SrcThreadConvertedScratch src_converted_thread_scratch_;
    ScaleThreadScratch scale_thread_scratch_;

    DstThreadScratch dst_thread_scratch_;
    FastTypeConverter fast_numeric_converter;

    SrcCoord src_coord_;
    ScaleCoord scale_coord_;
    DstCoord dst_coord_;
    const SrcElementwiseOperation src_element_op_;
    // Note: scale_element_op_ appears unused but is retained for API completeness
    const ScaleElementwiseOperation scale_element_op_;
    const DstElementwiseOperation dst_element_op_;
};

} // namespace ck
