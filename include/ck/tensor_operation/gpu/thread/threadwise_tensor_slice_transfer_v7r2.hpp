// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/utility/is_detected.hpp"

namespace ck {

// Thread-level multi-source, multi-destination tensor slice data movement
// Assume:
//   1. All sources and destinations are DynamicBuffer
//   2. Same VectorDim and ScalerPerVector for all sources and destinations
//   3. DstInMemOps are per destination tensor
//   4. ThreadTransferSrcResetCoordinateAfterRunFlags are per source tensor
//   5. ThreadTransferDstResetCoordinateAfterRunFlags are per destination tensor
//   6. Does not need to know src_descs and dst_descs at compile-time
//   7. Does not need to know src_slice_origins and dst_slice_origins at compile-time,
//
// Does following things to avoid scratch memory issue
//   1. Use StaticallyIndexedArray or vector_type instead of C array for thread buffer
//   2. Pass tensor descritpors by reference (or tuple of references)
//   3. Does not keep reference to tensor descriptor
//   4. Does not construct new tensor coordinate when call Run()
template <typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence<InMemoryDataOperationEnum ...>
          typename SliceLengths,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          typename SrcResetCoordinateAfterRunFlags, // Sequence<bool ...>
          typename DstResetCoordinateAfterRunFlags, // Sequence<bool ...>
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v7r2
{
    static constexpr auto I0 = Number<0>{};

    static constexpr index_t nDim = SliceLengths::Size();

    static constexpr index_t nSrc = SrcDescs::Size();
    static constexpr index_t nDst = DstDescs::Size();

    using Index = MultiIndex<nDim>;

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

    // scalar per access on each dim
    // FIXME: don't use lambda_scalar_per_access
    static constexpr auto src_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

    using SrcSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   SrcDimAccessOrder,
                                                   remove_cv_t<decltype(src_scalar_per_access)>>;

    static constexpr auto dst_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

    using DstSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   DstDimAccessOrder,
                                                   remove_cv_t<decltype(dst_scalar_per_access)>>;

    __device__ constexpr ThreadwiseTensorSliceTransfer_v7r2(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_slice_origins,
        const ElementwiseOperation& element_op)
        : src_coords_(MakeCoordinates(src_descs, src_slice_origins)),
          dst_coords_(MakeCoordinates(dst_descs, dst_slice_origins)),
          element_op_(element_op)
    {
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! cannot evenly divide");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! cannot evenly divide");
    }

    template <typename Indices, enable_if_t<SrcDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetSrcSliceOrigins(const SrcDescs& src_descs,
                                       const Indices& src_slice_origin_idxs)
    {
        static_for<0, nSrc, 1>{}([&](auto i) {
            src_coords_(i) = make_tensor_coordinate(src_descs[i], src_slice_origin_idxs[i]);
        });
    }

    template <typename Indices, enable_if_t<DstDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetDstSliceOrigins(const DstDescs& dst_descs,
                                       const Indices& dst_slice_origin_idxs)
    {
        static_for<0, nDst, 1>{}([&](auto i) {
            dst_coords_(i) = make_tensor_coordinate(dst_descs[i], dst_slice_origin_idxs[i]);
        });
    }

    template <typename DataTypes, index_t ScalarPerVector>
    __device__ static auto generate_vectors()
    {
        auto data_types = DataTypes{};

        constexpr index_t num = data_types.Size();

        return generate_tuple(
            [&](auto i) {
                using DataType = remove_cvref_t<decltype(data_types[i])>;

                return vector_type_maker_t<DataType, ScalarPerVector>{};
            },
            Number<num>{});
    }

    template <typename DataTypes, index_t ScalarPerVector>
    __device__ static auto generate_oob_vectors()
    {
        auto data_types = DataTypes{};

        constexpr index_t num = data_types.Size();

        return generate_tuple(
            [&](auto i) {
                ignore = i;
                return vector_type_maker_t<bool, ScalarPerVector>{};
            },
            Number<num>{});
    }

    // SrcDescs: Tuple<const SrcDesc0&, const SrcDesc1&, ...>
    // SrcBuffers: Tuple<const SrcBuffer0&, const SrcBuffer1&, ...>
    template <typename SrcBuffers,
              index_t ThreadScratchId                                   = 0,
              enable_if_t<SrcDescs::Size() == SrcBuffers::Size(), bool> = false>
    __device__ void RunRead(const SrcDescs& src_descs,
                            const SrcBuffers& src_bufs,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // loop over space-filling curve
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            auto src_vectors = generate_vectors<SrcDatas, SrcScalarPerVector>();
            auto dst_vectors = generate_vectors<DstDatas, DstScalarPerVector>();
            auto oob_vectors = generate_oob_vectors<DstDatas, DstScalarPerVector>();

            // copy data from src_bufs into src_vectors
            static_for<0, nSrc, 1>{}([&](auto i) {
                using src_vector_t = typename remove_cvref_t<decltype(src_vectors[i])>::type;

                const bool is_src_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(src_descs[i],
                                                                                src_coords_[i]);

                oob_vectors(i).template AsType<bool>()(I0) = is_src_valid;
                src_vectors(i).template AsType<src_vector_t>()(I0) =
                    src_bufs[i].template Get<src_vector_t>(src_coords_[i].GetOffset(), true);
            });

            constexpr auto get_elem_op_vec_len = []() {
                if constexpr(is_detected<is_pack8_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack8_invocable)
                        return math::min(8, SrcScalarPerVector);
                }
                if constexpr(is_detected<is_pack4_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack4_invocable)
                        return math::min(4, SrcScalarPerVector);
                }
                if constexpr(is_detected<is_pack2_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack2_invocable)
                        return math::min(2, SrcScalarPerVector);
                }
                return 1;
            };

            constexpr index_t elem_op_vec_len = get_elem_op_vec_len();

            // apply pointwise function
            static_for<0, SrcScalarPerVector / elem_op_vec_len, 1>{}([&](auto i) {
                // get reference to src data
                const auto src_data_refs = generate_tie(
                    // return type should be lvalue
                    [&](auto iSrc) -> const auto& {
                        using SrcData = remove_cvref_t<tuple_element_t<iSrc.value, SrcDatas>>;

                        using elem_op_vec_t = typename vector_type<SrcData, elem_op_vec_len>::type;

                        return src_vectors[iSrc].template AsType<elem_op_vec_t>()[i];
                    },
                    Number<nSrc>{});

                // get reference to dst data
                auto dst_data_refs = generate_tie(
                    // return type should be lvalue
                    [&](auto iDst) -> auto& {
                        using DstData = remove_cvref_t<tuple_element_t<iDst.value, DstDatas>>;

                        using elem_op_vec_t = typename vector_type<DstData, elem_op_vec_len>::type;

                        return dst_vectors(iDst).template AsType<elem_op_vec_t>()(i);
                    },
                    Number<nDst>{});

                // apply pointwise function
                // pointwise function signature:
                // element_op_(dst_data_refs[I0],
                //             dst_data_refs[I1],
                //             ...,
                //             src_data_refs[I0],
                //             src_data_refs[I1],
                //             ...)
                unpack2(element_op_, dst_data_refs, src_data_refs);
            });

            dst_vectors_tuple_(thread_scratch_id)(iAccess) = dst_vectors;
            oob_vectors_tuple_(thread_scratch_id)(iAccess) = oob_vectors;

            // move coordinate
            if constexpr(iAccess.value != num_access - 1)
            {
                constexpr auto forward_step = SrcSpaceFillingCurve::GetForwardStep(iAccess);

                static_for<0, nSrc, 1>{}([&](auto i) {
                    move_tensor_coordinate(src_descs[i],
                                           src_coords_(i),
                                           make_tensor_coordinate_step(src_descs[i], forward_step));
                });
            }
        });

        // move coordinate back to slice origin (or not)
        static_for<0, nSrc, 1>{}([&](auto i) {
            if constexpr(SrcResetCoordinateAfterRunFlags::At(i))
            {
                const auto src_reset_step =
                    make_tensor_coordinate_step(src_descs[i], GetSrcCoordinateResetStep());

                move_tensor_coordinate(src_descs[i], src_coords_(i), src_reset_step);
            }
        });
    }

    // DstDescs: Tuple<const DstDesc0&, const DstDesc1&, ...>
    // DstBuffers: Tuple<const DstBuffer0&, const DstBuffer1&, ...>
    template <typename DstBuffers,
              index_t ThreadScratchId                                   = 0,
              enable_if_t<DstDescs::Size() == DstBuffers::Size(), bool> = false>
    __device__ void RunWrite(const DstDescs& dst_descs,
                             DstBuffers dst_bufs,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // loop over space-filling curve
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            auto dst_vectors = dst_vectors_tuple_[thread_scratch_id][iAccess];
            auto oob_vectors = oob_vectors_tuple_[thread_scratch_id][iAccess];

            // copy data from buf_vectors into dst_bufs
            static_for<0, nDst, 1>{}([&](auto i) {
                using dst_vector_t = typename remove_cvref_t<decltype(dst_vectors[i])>::type;

                const bool is_dst_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_descs[i],
                                                                                dst_coords_[i]);

                constexpr InMemoryDataOperationEnum DstInMemOp =
                    static_cast<InMemoryDataOperationEnum>(DstInMemOps::At(i.value));

                const bool is_src_valid = oob_vectors[i].template AsType<bool>()[I0];

                const auto dst_t = is_src_valid ? dst_vectors[i].template AsType<dst_vector_t>()[I0]
                                                : dst_vector_t{0};

                dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(
                    dst_coords_[i].GetOffset(), is_dst_valid, dst_t);
            });

            // move coordinate
            if constexpr(iAccess.value != num_access - 1)
            {
                constexpr auto forward_step = DstSpaceFillingCurve::GetForwardStep(iAccess);

                static_for<0, nDst, 1>{}([&](auto i) {
                    move_tensor_coordinate(dst_descs[i],
                                           dst_coords_(i),
                                           make_tensor_coordinate_step(dst_descs[i], forward_step));
                });
            }
        });

        static_for<0, nDst, 1>{}([&](auto i) {
            if constexpr(DstResetCoordinateAfterRunFlags::At(i))
            {
                const auto dst_reset_step =
                    make_tensor_coordinate_step(dst_descs[i], GetDstCoordinateResetStep());

                move_tensor_coordinate(dst_descs[i], dst_coords_(i), dst_reset_step);
            }
        });
    }

    // SrcDescs: Tuple<const SrcDesc0&, const SrcDesc1&, ...>
    // SrcBuffers: Tuple<const SrcBuffer0&, const SrcBuffer1&, ...>
    // DstDescs: Tuple<const DstDesc0&, const DstDesc1&, ...>
    // DstBuffers: Tuple<const DstBuffer0&, const DstBuffer1&, ...>
    template <typename SrcBuffers,
              typename DstBuffers,
              enable_if_t<SrcDescs::Size() == SrcBuffers::Size() &&
                              DstDescs::Size() == DstBuffers::Size(),
                          bool> = false>
    __device__ void Run(const SrcDescs& src_descs,
                        const SrcBuffers& src_bufs,
                        const DstDescs& dst_descs,
                        DstBuffers dst_bufs)
    {
        RunRead(src_descs, src_bufs);
        RunWrite(dst_descs, dst_bufs);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        if constexpr(num_access == 0)
        {
            return typename SrcSpaceFillingCurve::Index{};
        }
        else
        {
            return SrcSpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        if constexpr(num_access == 0)
        {
            return typename DstSpaceFillingCurve::Index{};
        }
        else
        {
            return DstSpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});
        }
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <index_t ISrc>
    __device__ void MoveSrcSliceWindow(const SrcDescs& src_descs,
                                       Number<ISrc> iSrc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRunFlags::At(iSrc)
                ? src_slice_origin_step_idx
                : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_descs[iSrc], adjusted_step_idx);

        move_tensor_coordinate(src_descs[iSrc], src_coords_(iSrc), adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <index_t IDst>
    __device__ void MoveDstSliceWindow(const DstDescs& dst_descs,
                                       Number<IDst> iDst,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRunFlags::At(iDst)
                ? dst_slice_origin_step_idx
                : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_descs[iDst], adjusted_step_idx);

        move_tensor_coordinate(dst_descs[iDst], dst_coords_(iDst), adjusted_step);
    }

    private:
    using SrcVectorsType = decltype(generate_vectors<SrcDatas, SrcScalarPerVector>());
    using DstVectorsType = decltype(generate_vectors<DstDatas, DstScalarPerVector>());

    using OOBVectorsType = decltype(generate_oob_vectors<DstDatas, DstScalarPerVector>());

    static constexpr auto num_access = SrcSpaceFillingCurve::GetNumOfAccess();

    using DstVectorTuple = StaticallyIndexedArray<DstVectorsType, num_access>;
    using OOBVectorTuple = StaticallyIndexedArray<OOBVectorsType, num_access>;

    StaticallyIndexedArray<DstVectorTuple, NumThreadScratch> dst_vectors_tuple_;
    StaticallyIndexedArray<OOBVectorTuple, NumThreadScratch> oob_vectors_tuple_;

    SrcCoords src_coords_;
    DstCoords dst_coords_;
    const ElementwiseOperation element_op_;
};

} // namespace ck
