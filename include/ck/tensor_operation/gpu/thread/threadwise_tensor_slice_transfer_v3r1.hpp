// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_coordinate.hpp"
#include "ck/tensor/thread_private_tensor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
// FIXME: remove
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor/static_tensor.hpp"

namespace ck {

namespace detail {
// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t DstVectorDim,
          index_t DstScalarPerVector>
struct lambda_scalar_per_access_for_src_and_dst
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if(i == SrcVectorDim && i == DstVectorDim)
        {
            return math::lcm(SrcScalarPerVector, DstScalarPerVector);
        }
        else if(i == SrcVectorDim)
        {
            return SrcScalarPerVector;
        }
        else if(i == DstVectorDim)
        {
            return DstScalarPerVector;
        }
        else
        {
            return 1;
        }
    }
};

} // namespace detail

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
template <typename SliceLengths,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcDataTmp,
          typename DstDataTmp,
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
          bool DstResetCoordinateAfterRun, // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r1
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcData = remove_cvref_t<SrcDataTmp>;
    using DstData = remove_cvref_t<DstDataTmp>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    // scalar per access on each dim
    // TODO: don't use lambda_scalar_per_access
    static constexpr auto src_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});
    static constexpr auto dst_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

    using SrcSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   SrcDimAccessOrder,
                                                   remove_cv_t<decltype(src_scalar_per_access)>>;

    using DstSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   DstDimAccessOrder,
                                                   remove_cv_t<decltype(dst_scalar_per_access)>>;

    __host__ __device__ constexpr ThreadwiseTensorSliceTransfer_v3r1()
        : src_coord_{}, dst_coord_{}, src_element_op_{}, dst_element_op_{}
    {
    }

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r1(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          src_element_op_(src_element_op),
          dst_element_op_(dst_element_op)
    {
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
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

        constexpr auto num_access = SrcSpaceFillingCurve::GetNumOfAccess();

        // loop over space-filling curve
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            // calculate src data index
            constexpr auto src_data_idx = SrcSpaceFillingCurve::GetIndexTupleOfNumber(iAccess);

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_top_index_is_valid(src_desc, src_coord_);

            using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            // copy data from src_buf into src_vector_container
            auto src_vector_container = src_vector_type{
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid)};

            // copy data from src_vector_container into src_thread_scratch_
#if 1 // debug
            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<src_vector_t>(
                    src_data_idx, src_vector_container.template AsType<src_vector_t>()[I0]);
#else
            src_thread_scratch_tuple_(thread_scratch_id)
                .template Set<src_vector_t>(
                    src_data_idx, src_vector_container.template AsType<src_vector_t>()[I0]);
#endif

            // move src coordinate
            if constexpr(iAccess.value != num_access - 1)
            {
                constexpr auto step = SrcSpaceFillingCurve::GetForwardStep(iAccess);

                move_tensor_coordinate(src_desc, src_coord_, step);
            }
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step = GetSrcCoordinateResetStep();

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
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

        static_ford<SliceLengths>{}([&](auto idx) {
            // apply the src elementwise op and convert to DstData under the hood if needed
            DstData dst_v;
            src_element_op_(dst_v, src_thread_scratch_tuple_[thread_scratch_id][idx]);
            dst_thread_scratch_(idx) = dst_v;
        });
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

        constexpr auto num_access = DstSpaceFillingCurve::GetNumOfAccess();

        // loop over tensor and copy
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            // calculate src data index
            constexpr auto dst_data_idx = DstSpaceFillingCurve::GetIndexTupleOfNumber(iAccess);

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_top_index_is_valid(dst_desc, dst_coord_);

            using dst_vector_type = vector_type_maker_t<DstData, DstScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            // copy data from dst_thread_scratch_ into dst_vector_container
            auto dst_vector_container =
                dst_vector_type{dst_thread_scratch_.template GetAsType<dst_vector_t>(dst_data_idx)};

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
                dst_vector_container.template AsType<dst_vector_t>()[I0]);

            // move dst coord
            if constexpr(iAccess.value != num_access - 1)
            {
                constexpr auto step = DstSpaceFillingCurve::GetForwardStep(iAccess);

                move_tensor_coordinate(dst_desc, dst_coord_, step);
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            move_tensor_coordinate(dst_desc, dst_coord_, GetDstCoordinateResetStep());
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto num_access = SrcSpaceFillingCurve::GetNumOfAccess();

        if constexpr(num_access == 0)
        {
            return typename SrcSpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SrcSpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto num_access = DstSpaceFillingCurve::GetNumOfAccess();

        if constexpr(num_access == 0)
        {
            return typename DstSpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                DstSpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step_idx);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step_idx);
    }

    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(src_access_lengths), Number<SrcScalarPerVector>{});

        // 1st stage of transforms
        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(src_access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(src_access_lengths_and_vector_length[i],
                                   src_access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(src_access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        // 1st stage of transforms
        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(dst_access_lengths), Number<DstScalarPerVector>{});

        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(dst_access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(dst_access_lengths_and_vector_length[i],
                                   dst_access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(dst_access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    private:
    static constexpr auto src_thread_scratch_desc_ = decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto dst_thread_scratch_desc_ = decltype(GetDstThreadScratchDescriptor()){};

#if 1 // debug
    using SrcThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             SrcData,
                                                             SrcScalarPerVector,
                                                             decltype(src_thread_scratch_desc_),
                                                             true>;
#else
    using SrcThreadScratch = ThreadPrivateTensor<SrcData, decltype(src_thread_scratch_desc_)>;
#endif

    using DstThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             DstScalarPerVector,
                                                             decltype(dst_thread_scratch_desc_),
                                                             true>;

    StaticallyIndexedArray<SrcThreadScratch, NumThreadScratch> src_thread_scratch_tuple_;

    DstThreadScratch dst_thread_scratch_;

    SrcCoord src_coord_;
    DstCoord dst_coord_;
    const SrcElementwiseOperation src_element_op_;
    const DstElementwiseOperation dst_element_op_;
};

} // namespace ck
