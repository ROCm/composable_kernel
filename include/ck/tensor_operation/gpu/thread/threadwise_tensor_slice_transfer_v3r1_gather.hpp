// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/utility/is_detected.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

namespace ck {

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
template <typename SliceLengths,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector_,
          index_t DstScalarPerVector_,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun, // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
          typename IndexType,
          index_t GatherDim        = 1,
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r1_gather
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using Helper = ThreadwiseTransferHelper_Serpentine;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr auto SrcScalarPerVector = Number<SrcScalarPerVector_ / PackedSize>{};
    static constexpr auto DstScalarPerVector = Number<DstScalarPerVector_ / PackedSize>{};

    static constexpr index_t gather_num = SliceLengths{}.At(Number<GatherDim>{});

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r1_gather(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const DstElementwiseOperation& dst_element_op,
        const StaticallyIndexedArray<IndexType, gather_num>& gather_offsets)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          src_element_op_(src_element_op),
          dst_element_op_(dst_element_op),
          gather_offsets_(gather_offsets)
    {
        if constexpr((packed_size_v<SrcData>) > 1)
        {
            static_assert(is_same_v<remove_cvref_t<SrcData>, remove_cvref_t<DstData>>,
                          "SrcData != DstData");

            static_assert(
                SrcScalarPerVector_ % PackedSize == 0 && DstScalarPerVector_ % PackedSize == 0,
                "SrcScalarPerVector_ and DstScalarPerVector_ cannot be 1 for packed data type");

            static_assert(SrcVectorDim == DstVectorDim,
                          "Packed data type does not support transpose");
        }
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {

        auto adjusted_origin_idx = [&]() {
            Index idx;
            static_for<0, nDim, 1>{}([&](auto i) {
                idx(i) = i.value == GatherDim ? 0 : src_slice_origin_idx[Number<i>{}];
            });
            return idx;
        }();
        src_coord_ = make_tensor_coordinate(src_desc, adjusted_origin_idx);
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
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        static_assert(SliceLengths::At(SrcVectorDim) % (SrcScalarPerVector_) == 0,
                      "SliceLengths[SrcVectorDim] must be divisible by SrcScalarPerVector");

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};
        constexpr auto ordered_gather_dim   = src_dim_access_order[GatherDim];
        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward and backward steps
        const auto src_forward_steps = Helper::ComputeForwardSteps(src_desc, src_scalar_per_access);
        const auto src_backward_steps =
            Helper::ComputeBackwardSteps(src_desc, src_scalar_per_access);

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep =
                Helper::ComputeForwardSweep(ordered_src_access_idx, ordered_src_access_lengths);

            // calculate src data index
            constexpr auto src_data_idx = Helper::ComputeDataIndex(ordered_src_access_idx,
                                                                   ordered_src_access_lengths,
                                                                   forward_sweep,
                                                                   src_dim_access_order,
                                                                   src_scalar_per_access);

            constexpr auto src_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<src_data_idx[i]>{}; }, Number<src_data_idx.Size()>{});

            auto gather_offset =
                gather_offsets_(ordered_src_access_idx[Number<ordered_gather_dim>{}]);

            const IndexType ld_offset = src_coord_.GetOffset() / PackedSize + gather_offset;
            src_oob_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<bool>(src_data_idx_seq, true);

            using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            auto src_vector_container =
                src_vector_type{src_buf.template Get<src_vector_t>(ld_offset, true)};

            using dst_vector_type = vector_type_maker_t<DstData, SrcScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;
            dst_vector_type op_r_v;

            constexpr auto get_elem_op_vec_len = []() {
                if constexpr(is_detected<is_pack8_invocable_t, decltype(src_element_op_)>::value)
                {
                    if constexpr(decltype(src_element_op_)::is_pack8_invocable)
                        return math::min(8, SrcScalarPerVector);
                }
                else if constexpr(is_detected<is_pack4_invocable_t,
                                              decltype(src_element_op_)>::value)
                {
                    if constexpr(decltype(src_element_op_)::is_pack4_invocable)
                        return math::min(4, SrcScalarPerVector);
                }
                else if constexpr(is_detected<is_pack2_invocable_t,
                                              decltype(src_element_op_)>::value)
                {
                    if constexpr(decltype(src_element_op_)::is_pack2_invocable)
                        return math::min(2, SrcScalarPerVector);
                }
                else
                {
                    return 1;
                }
            };

            constexpr index_t elem_op_vec_len = get_elem_op_vec_len();

            using src_elem_op_vec_t = typename vector_type<SrcData, elem_op_vec_len>::type;
            using dst_elem_op_vec_t = typename vector_type<DstData, elem_op_vec_len>::type;

            static_for<0, SrcScalarPerVector / elem_op_vec_len, 1>{}([&](auto idx) {
                // apply the src elementwise op and convert to DstData under the hood if needed
                src_element_op_(op_r_v.template AsType<dst_elem_op_vec_t>()(idx),
                                src_vector_container.template AsType<src_elem_op_vec_t>()[idx]);
            });

            // copy data from src_vector_container into src_thread_scratch_
            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<dst_vector_t>(
                    src_data_idx_seq, op_r_v.template AsType<dst_vector_t>()[Helper::I0]);

            // Gather-specific: skip gather dimension during coordinate movement
            auto move_on_dim = [&]() constexpr {
                auto move_on_dim_ =
                    Helper::ComputeMoveOnDim(ordered_src_access_idx, ordered_src_access_lengths);

                static_for<0, nDim, 1>{}(
                    [&](auto i) { move_on_dim_(i) &= i.value != ordered_gather_dim; });

                return move_on_dim_;
            }();

            // move src coord
            static_for<0, nDim, 1>{}([&](auto i) {
                if(move_on_dim[i])
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

    template <typename SeqIdx, index_t ThreadScratchId = 0>
    __device__ constexpr auto
    GetSrcThreadScratchIdx(Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        using vector_t = typename vector_type_maker<SrcData, SrcScalarPerVector>::type::type;
        return src_thread_scratch_tuple_(thread_scratch_id).template GetAsType<vector_t>(SeqIdx{});
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
#if !CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        static_ford<SliceLengths>{}([&](auto idx) {
            dst_thread_scratch_(idx) = src_thread_scratch_tuple_[thread_scratch_id][idx];
        });
#else

        // OOB Check
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

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

            using vector_t = typename vector_type_maker<DstData, SrcScalarPerVector>::type::type;

            auto op_r = src_thread_scratch_tuple_(thread_scratch_id)
                            .template GetAsType<vector_t>(src_data_idx_seq);

            auto op_r_v = op_r;

            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<vector_t>(src_data_idx_seq, op_r_v);
        });

        // sub-dword transpose between src_thread_scratch_ and dst_thread_scratch_
        // TODO make this logic more generic for more sub-dword datatype
        if constexpr(SrcVectorDim != DstVectorDim &&
                     ((is_same<half_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 2 == 0 && DstScalarPerVector % 2 == 0) ||
                      (is_same<int8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0) ||
                      (is_same<f8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0)))
        {
            static_assert(!is_same_v<remove_cvref_t<SrcData>, pk_i4_t>,
                          "in-register transpose is not supported for pk_i4_t");
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

                using src_vector_t = vector_type_maker_t<DstData, SrcScalarPerVector>;
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
                transpose_vectors<DstData, DstScalarPerVector, SrcScalarPerVector>{}(
                    src_vector_refs, dst_vector_refs);
            });
        }
        else
        {
            constexpr auto packed_per_access = generate_sequence(
                detail::lambda_scalar_per_access<SrcVectorDim, PackedSize>{}, Number<nDim>{});

            constexpr auto packed_access_lengths = SliceLengths{} / packed_per_access;

            static_ford<decltype(packed_access_lengths)>{}([&](auto idx) {
                dst_thread_scratch_(idx) = src_thread_scratch_tuple_[thread_scratch_id][idx];
            });
        }
#endif
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // if there is transpose, it's done here
        // if there is oob check, it's done here
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
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector_>{}, Number<nDim>{});

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
                dst_coord_.GetOffset() / PackedSize,
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

    // Gather-specific: src coordinate reset zeroes the gather dimension
    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        constexpr auto ordered_access_lengths_minus_1 = generate_tuple(
            [&](auto i) { return Number<ordered_src_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
        constexpr auto forward_sweep =
            Helper::ComputeForwardSweep(ordered_access_lengths_minus_1, ordered_src_access_lengths);

        constexpr auto src_data_idx = [&]() {
            MultiIndex<nDim> ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_src_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                   src_scalar_per_access;
        }();

        // Gather-specific: don't reset the gather dimension
        constexpr auto reset_src_data_step = [&]() {
            MultiIndex<nDim> reset_src_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) {
                reset_src_data_step_(i) = i.value == GatherDim ? 0 : -src_data_idx[i];
            });

            return reset_src_data_step_;
        }();
        return reset_src_data_step;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        return Helper::ComputeCoordinateResetStep<SliceLengths,
                                                  DstVectorDim,
                                                  DstScalarPerVector_,
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
            ComputeThreadScratchDescriptor<SliceLengths, SrcVectorDim, SrcScalarPerVector_>();
    }

    __device__ static constexpr auto GetSrcOOBThreadScratchDescriptor()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        return make_naive_tensor_descriptor_packed(src_access_lengths);
    }

    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        return Helper::
            ComputeThreadScratchDescriptor<SliceLengths, DstVectorDim, DstScalarPerVector_>();
    }

    private:
    static constexpr auto src_thread_scratch_desc_ = decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto src_oob_thread_scratch_desc_ =
        decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto dst_thread_scratch_desc_ = decltype(GetDstThreadScratchDescriptor()){};

    using SrcThreadScratch =
        StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                        DstData, // apply data_convert with SrcThreadScratch
                                        SrcScalarPerVector,
                                        decltype(src_thread_scratch_desc_),
                                        true>;

    using SrcOOBThreadScratch =
        StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                        bool, // apply data_convert with SrcThreadScratch
                                        1,
                                        decltype(src_oob_thread_scratch_desc_),
                                        true>;

    using DstThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             DstScalarPerVector,
                                                             decltype(dst_thread_scratch_desc_),
                                                             true>;

    StaticallyIndexedArray<SrcThreadScratch, NumThreadScratch> src_thread_scratch_tuple_;
    StaticallyIndexedArray<SrcOOBThreadScratch, NumThreadScratch> src_oob_thread_scratch_tuple_;

    DstThreadScratch dst_thread_scratch_;

    SrcCoord src_coord_;
    DstCoord dst_coord_;
    const SrcElementwiseOperation src_element_op_;
    const DstElementwiseOperation dst_element_op_;
    StaticallyIndexedArray<IndexType, gather_num> gather_offsets_;
};

} // namespace ck
