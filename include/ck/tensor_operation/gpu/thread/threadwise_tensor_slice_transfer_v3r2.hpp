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
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence
          typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          typename SrcsScalarPerVector,         // Sequence
          typename DstsScalarPerVector,         // Sequence
          typename SrcsScalarStrideInVector,    // Sequence
          typename DstsScalarStrideInVector,    // Sequence
          typename SrcsResetCoordinateAfterRun, // control whether to move back src coordinate after
                                                // each RunRead(),  will be fused with
                                                // MoveSrcSliceWindow to save addr computation
          typename DstsResetCoordinateAfterRun, // control whether to move back dst coordinate after
                                                // each RunWrite(),  will be fused with
                                                // MoveDstSliceWindow to save addr computation
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    static constexpr index_t nSrc = SrcDescs::Size();
    static constexpr index_t nDst = DstDescs::Size();

    using Helper = ThreadwiseTransferHelper_Serpentine;

    // return a tuple of coordiantes for a tuple of tensor
    template <typename Descs,
              typename Indices,
              enable_if_t<Descs::Size() == Indices::Size(), bool> = false>
    static constexpr auto MakeCoordinates(const Descs& descs, const Indices& indices)
    {
        return generate_tuple([&](auto i) { return make_tensor_coordinate(descs[i], indices[i]); },
                              Number<Descs::Size()>{});
    }

    using SrcCoords = decltype(MakeCoordinates(SrcDescs{}, StaticallyIndexedArray<Index, nSrc>{}));
    using DstCoords = decltype(MakeCoordinates(DstDescs{}, StaticallyIndexedArray<Index, nDst>{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r2(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_slice_origins,
        const ElementwiseOperation& element_op)
        : src_coords_(MakeCoordinates(src_descs, src_slice_origins)),
          dst_coords_(MakeCoordinates(dst_descs, dst_slice_origins)),
          element_op_(element_op)
    {
    }

    template <typename Indices, enable_if_t<SrcDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetSrcSliceOrigins(const SrcDescs& src_descs,
                                       const Indices& src_slice_origin_idxs)
    {
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            src_coords_(src_i) =
                make_tensor_coordinate(src_descs.At(src_i), src_slice_origin_idxs[src_i]);
        });
    }

    template <typename Indices, enable_if_t<DstDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetDstSliceOrigins(const DstDescs& dst_descs,
                                       const Indices& dst_slice_origin_idxs)
    {
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            dst_coords_(dst_i) =
                make_tensor_coordinate(dst_descs.At(dst_i), dst_slice_origin_idxs[dst_i]);
        });
    }

    template <typename SrcBuffers, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDescs& src_descs,
                            const SrcBuffers& src_bufs,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // scalar per access on each dim
        constexpr auto src_scalar_per_access_tuple = generate_tuple(
            [&](auto src_i) {
                return generate_sequence(
                    detail::lambda_scalar_per_access<SrcVectorDim,
                                                     SrcsScalarPerVector::At(src_i)>{},
                    Number<nDim>{});
            },
            Number<nSrc>{});

        constexpr auto src_access_lengths_tuple = generate_tuple(
            [&](auto src_i) {
                return SliceLengths{} / src_scalar_per_access_tuple.At(src_i);
                static_assert(
                    SliceLengths::At(SrcVectorDim) % SrcsScalarPerVector::At(src_i) == 0,
                    "SliceLengths[SrcVectorDim] must be divisible by SrcsScalarPerVector");
            },
            Number<nSrc>{});

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths_tuple = generate_tuple(
            [&](auto src_i) {
                return container_reorder_given_new2old(src_access_lengths_tuple.At(src_i),
                                                       src_dim_access_order);
            },
            Number<nSrc>{});

        // make forward and backward steps
        const auto src_forward_steps_tuple = generate_tuple(
            [&](auto src_i) {
                return Helper::ComputeForwardSteps(src_descs.At(src_i),
                                                   src_scalar_per_access_tuple.At(src_i));
            },
            Number<nSrc>{});

        const auto src_backward_steps_tuple = generate_tuple(
            [&](auto src_i) {
                return Helper::ComputeBackwardSteps(src_descs.At(src_i),
                                                    src_scalar_per_access_tuple.At(src_i));
            },
            Number<nSrc>{});

        // loop over tensor and copy
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            static_ford<remove_cvref_t<decltype(ordered_src_access_lengths_tuple.At(src_i))>>{}(
                [&](auto ordered_src_access_idx) {
                    // judge move forward or move backward
                    constexpr auto forward_sweep = Helper::ComputeForwardSweep(
                        ordered_src_access_idx, ordered_src_access_lengths_tuple.At(src_i));

                    // calculate src data index
                    constexpr auto src_data_idx =
                        Helper::ComputeDataIndex(ordered_src_access_idx,
                                                 ordered_src_access_lengths_tuple.At(src_i),
                                                 forward_sweep,
                                                 src_dim_access_order,
                                                 src_scalar_per_access_tuple.At(src_i));

                    constexpr auto src_data_idx_seq =
                        generate_sequence_v2([&](auto i) { return Number<src_data_idx[i]>{}; },
                                             Number<src_data_idx.Size()>{});

                    const bool is_src_valid =
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_descs.At(src_i), src_coords_.At(src_i));

                    using src_vector_type = vector_type_maker_t<tuple_element_t<src_i, SrcDatas>,
                                                                SrcsScalarPerVector::At(src_i)>;
                    using src_vector_t    = typename src_vector_type::type;

                    // copy data from src_buf into src_vector_container
                    auto src_vector_container =
                        src_vector_type{src_bufs.At(src_i).template Get<src_vector_t>(
                            src_coords_.At(src_i).GetOffset(), is_src_valid)};

                    // copy data from src_vector_container into src_thread_scratch_
                    src_thread_scratch_tuple_(thread_scratch_id)
                        .At(src_i)
                        .template SetAsType<src_vector_t>(
                            src_data_idx_seq,
                            src_vector_container.template AsType<src_vector_t>()[Helper::I0]);

                    constexpr auto move_on_dim = Helper::ComputeMoveOnDim(
                        ordered_src_access_idx, ordered_src_access_lengths_tuple.At(src_i));

                    // move src coord
                    static_for<0, nDim, 1>{}([&](auto i) {
                        if constexpr(move_on_dim[i])
                        {
                            if constexpr(forward_sweep[i])
                            {
                                move_tensor_coordinate(
                                    src_descs.At(src_i),
                                    src_coords_.At(src_i),
                                    src_forward_steps_tuple.At(src_i)[src_dim_access_order[i]]);
                            }
                            else
                            {
                                move_tensor_coordinate(
                                    src_descs.At(src_i),
                                    src_coords_.At(src_i),
                                    src_backward_steps_tuple.At(src_i)[src_dim_access_order[i]]);
                            }
                        }
                    });
                });
        });

        static_for<0, nSrc, 1>{}([&](auto src_i) {
            // move src coordinate back to slice origin (or not)
            if constexpr(SrcsResetCoordinateAfterRun::At(src_i))
            {
                const auto src_reset_step = make_tensor_coordinate_step(
                    src_descs.At(src_i), GetSrcCoordinateResetStep<src_i>());

                move_tensor_coordinate(src_descs.At(src_i), src_coords_.At(src_i), src_reset_step);
            }
        });
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
        // TODO: Add support for CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        // (it requires to add Elementwise support in transpose_vectors)
        if constexpr(nSrc == 1 && nDst == 1)
        {
            // Fast path: direct element transfer, no generate_tie/unpack2 overhead
            static_ford<SliceLengths>{}([&](auto idx) {
                element_op_(dst_thread_scratch_tuple_.At(Number<0>{})(idx),
                            src_thread_scratch_tuple_[thread_scratch_id].At(Number<0>{})[idx]);
            });
        }
        else
        {
            // General path: use generate_tie + unpack2 for multi-src/dst
            static_ford<SliceLengths>{}([&](auto idx) {
                const auto src_data_refs = generate_tie(
                    [&](auto src_i) -> const auto& {
                        return src_thread_scratch_tuple_[thread_scratch_id].At(src_i)[idx];
                    },
                    Number<nSrc>{});

                auto dst_data_refs = generate_tie(
                    [&](auto dst_i) -> auto& { return dst_thread_scratch_tuple_.At(dst_i)(idx); },
                    Number<nDst>{});
                unpack2(element_op_, dst_data_refs, src_data_refs);
            });
        }
    }

    template <typename DstBuffers, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDescs& dst_descs,
                             DstBuffers& dst_bufs,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // if there is transpose, it's done here
        // TODO move this elsewhere
        TransferDataFromSrcThreadScratchToDstThreadScratch(thread_scratch_id);

        // src scalar per access on each dim
        constexpr auto dst_scalar_per_access_tuple = generate_tuple(
            [&](auto dst_i) {
                return generate_sequence(
                    detail::lambda_scalar_per_access<DstVectorDim,
                                                     DstsScalarPerVector::At(dst_i)>{},
                    Number<nDim>{});
            },
            Number<nDst>{});

        constexpr auto dst_access_lengths_tuple = generate_tuple(
            [&](auto dst_i) { return SliceLengths{} / dst_scalar_per_access_tuple.At(dst_i); },
            Number<nDst>{});

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths_tuple = generate_tuple(
            [&](auto dst_i) {
                return container_reorder_given_new2old(dst_access_lengths_tuple.At(dst_i),
                                                       dst_dim_access_order);
            },
            Number<nDst>{});

        // make forward and backward steps
        const auto dst_forward_steps_tuple = generate_tuple(
            [&](auto dst_i) {
                return Helper::ComputeForwardSteps(dst_descs.At(dst_i),
                                                   dst_scalar_per_access_tuple.At(dst_i));
            },
            Number<nDst>{});

        const auto dst_backward_steps_tuple = generate_tuple(
            [&](auto dst_i) {
                return Helper::ComputeBackwardSteps(dst_descs.At(dst_i),
                                                    dst_scalar_per_access_tuple.At(dst_i));
            },
            Number<nDst>{});

        // loop over tensor and copy
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            static_ford<remove_cvref_t<decltype(ordered_dst_access_lengths_tuple.At(dst_i))>>{}(
                [&](auto ordered_dst_access_idx) {
                    // judge move forward or move backward
                    constexpr auto forward_sweep = Helper::ComputeForwardSweep(
                        ordered_dst_access_idx, ordered_dst_access_lengths_tuple.At(dst_i));

                    // calculate dst data index
                    constexpr auto dst_data_idx =
                        Helper::ComputeDataIndex(ordered_dst_access_idx,
                                                 ordered_dst_access_lengths_tuple.At(dst_i),
                                                 forward_sweep,
                                                 dst_dim_access_order,
                                                 dst_scalar_per_access_tuple.At(dst_i));

                    constexpr auto dst_data_idx_seq =
                        generate_sequence_v2([&](auto i) { return Number<dst_data_idx[i]>{}; },
                                             Number<dst_data_idx.Size()>{});

                    const bool is_dst_valid =
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            dst_descs.At(dst_i), dst_coords_.At(dst_i));

                    using dst_vector_type = vector_type_maker_t<tuple_element_t<dst_i, DstDatas>,
                                                                DstsScalarPerVector::At(dst_i)>;
                    using dst_vector_t    = typename dst_vector_type::type;

                    // copy data from dst_thread_scratch_ into dst_vector_container
                    auto dst_vector_container = dst_vector_type{
                        dst_thread_scratch_tuple_.At(dst_i).template GetAsType<dst_vector_t>(
                            dst_data_idx_seq)};

                    constexpr InMemoryDataOperationEnum DstInMemOp =
                        static_cast<InMemoryDataOperationEnum>(DstInMemOps::At(dst_i.value));

                    // copy data from dst_vector_container to dst_buf
                    dst_bufs.At(dst_i).template Update<DstInMemOp, dst_vector_t>(
                        dst_coords_.At(dst_i).GetOffset(),
                        is_dst_valid,
                        dst_vector_container.template AsType<dst_vector_t>()[Helper::I0]);

                    constexpr auto move_on_dim = Helper::ComputeMoveOnDim(
                        ordered_dst_access_idx, ordered_dst_access_lengths_tuple.At(dst_i));

                    // move dst coord
                    static_for<0, nDim, 1>{}([&](auto i) {
                        if constexpr(move_on_dim[i])
                        {
                            if constexpr(forward_sweep[i])
                            {
                                move_tensor_coordinate(
                                    dst_descs.At(dst_i),
                                    dst_coords_.At(dst_i),
                                    dst_forward_steps_tuple.At(dst_i)[dst_dim_access_order[i]]);
                            }
                            else
                            {
                                move_tensor_coordinate(
                                    dst_descs.At(dst_i),
                                    dst_coords_.At(dst_i),
                                    dst_backward_steps_tuple.At(dst_i)[dst_dim_access_order[i]]);
                            }
                        }
                    });
                });
        });

        // move dst coordinate back to slice origin (or not)
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            if constexpr(DstsResetCoordinateAfterRun::At(dst_i))
            {
                const auto dst_reset_step = make_tensor_coordinate_step(
                    dst_descs.At(dst_i), GetDstCoordinateResetStep<dst_i>());

                move_tensor_coordinate(dst_descs.At(dst_i), dst_coords_.At(dst_i), dst_reset_step);
            }
        });
    }

    template <index_t src_i>
    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        return Helper::ComputeCoordinateResetStep<SliceLengths,
                                                  SrcVectorDim,
                                                  SrcsScalarPerVector::At(src_i),
                                                  SrcDimAccessOrder>();
    }

    template <index_t dst_i>
    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        return Helper::ComputeCoordinateResetStep<SliceLengths,
                                                  DstVectorDim,
                                                  DstsScalarPerVector::At(dst_i),
                                                  DstDimAccessOrder>();
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDescs& src_descs,
                                       const Index& src_slice_origin_step_idx)
    {
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            // if src coord was not reset by RunRead(), then need to adjust the step here
            const auto adjusted_step_idx =
                SrcsResetCoordinateAfterRun::At(src_i)
                    ? src_slice_origin_step_idx
                    : src_slice_origin_step_idx + GetSrcCoordinateResetStep<src_i>();

            // is it OK to construct a new step every time?
            const auto adjusted_step =
                make_tensor_coordinate_step(src_descs.At(src_i), adjusted_step_idx);

            move_tensor_coordinate(src_descs.At(src_i), src_coords_.At(src_i), adjusted_step);
        });
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDescs& dst_descs,
                                       const Index& dst_slice_origin_step_idx)
    {
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            // if dst coord was not reset by RunWrite(), then need to adjust the step here
            const auto adjusted_step_idx =
                DstsResetCoordinateAfterRun::At(dst_i)
                    ? dst_slice_origin_step_idx
                    : dst_slice_origin_step_idx + GetDstCoordinateResetStep<dst_i>();

            // is it OK to construct a new step every time?
            const auto adjusted_step =
                make_tensor_coordinate_step(dst_descs.At(dst_i), adjusted_step_idx);

            move_tensor_coordinate(dst_descs.At(dst_i), dst_coords_.At(dst_i), adjusted_step);
        });
    }

    template <index_t src_i>
    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        return Helper::ComputeThreadScratchDescriptor<SliceLengths,
                                                      SrcVectorDim,
                                                      SrcsScalarPerVector::At(src_i)>();
    }

    template <index_t dst_i>
    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        return Helper::ComputeThreadScratchDescriptor<SliceLengths,
                                                      DstVectorDim,
                                                      DstsScalarPerVector::At(dst_i)>();
    }

    __device__ static constexpr auto MakeSrcThreadScratchTuple()
    {
        return generate_tuple(
            [&](auto src_i) {
                constexpr auto src_thread_scratch_desc =
                    decltype(GetSrcThreadScratchDescriptor<src_i>()){};
                using SrcThreadScratch =
                    StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                    tuple_element_t<src_i, SrcDatas>,
                                                    SrcsScalarPerVector::At(src_i),
                                                    decltype(src_thread_scratch_desc),
                                                    true>;
                return SrcThreadScratch{};
            },
            Number<nSrc>{});
    }

    __device__ static constexpr auto MakeDstThreadScratchTuple()
    {
        return generate_tuple(
            [&](auto dst_i) {
                constexpr auto dst_thread_scratch_desc =
                    decltype(GetDstThreadScratchDescriptor<dst_i>()){};
                using DstThreadScratch =
                    StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                    tuple_element_t<dst_i, DstDatas>,
                                                    DstsScalarPerVector::At(dst_i),
                                                    decltype(dst_thread_scratch_desc),
                                                    true>;
                return DstThreadScratch{};
            },
            Number<nDst>{});
    }

    private:
    using SrcThreadScratchTuple = decltype(MakeSrcThreadScratchTuple());
    using DstThreadScratchTuple = decltype(MakeDstThreadScratchTuple());

    StaticallyIndexedArray<SrcThreadScratchTuple, NumThreadScratch> src_thread_scratch_tuple_;

    DstThreadScratchTuple dst_thread_scratch_tuple_;

    SrcCoords src_coords_;
    DstCoords dst_coords_;
    const ElementwiseOperation element_op_;
};

} // namespace ck
