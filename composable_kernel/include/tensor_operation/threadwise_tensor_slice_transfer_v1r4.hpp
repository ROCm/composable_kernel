#ifndef CK_THREADWISE_TENSOR_SLICE_TRANSFER_V1R4_HPP
#define CK_THREADWISE_TENSOR_SLICE_TRANSFER_V1R4_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

// Assume:
//   1. src:
//     1. Src0Desc is known at compile-time
//     2. Src0Buffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename Src0Data,
          typename DstData,
          typename Src0Desc,
          typename DstDesc,
          typename SrcElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum_t DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<Src0Desc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r4
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r4(
        const DstDesc& dst_desc,
        const Index& dst_slice_origin_idx,
        const SrcElementwiseOperation src_element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx)),
          src_element_op_{src_element_op}
    {
        static_assert(Src0Desc::IsKnownAtCompileTime(),
                      "wrong! Src0Desc need to known at compile-time");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx,
              typename Src0Buffer,
              typename DstBuffer,
              typename DstStepHacks>
    __device__ void Run(const Src0Desc&,
                        const SrcSliceOriginIdx&,
                        const Src0Buffer& src0_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf,
                        const DstStepHacks& dst_step_hacks)
    {
        static_assert(Src0Desc::IsKnownAtCompileTime(),
                      "wrong! Src0Desc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(Src0Buffer::IsStaticBuffer(), "wrong! Src0Buffer need to be StaticBuffer");

        // Src0Desc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<Src0Desc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

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
        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i]
                                         ? ordered_access_idx[i]
                                         : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                       dst_scalar_per_access;
            }();

            typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;

            using dst_vector_t =
                typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

            // copy data from src0_buf into dst_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + dst_data_idx + i * dst_scalar_step_in_vector);

                // apply element-wise operation and type convert
                dst_vector.template AsType<DstData>()(i) =
                    type_convert<DstData>(src_element_op_(src0_buf[Number<src_offset>{}]));
            });

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            if constexpr(DstInMemOp == InMemoryDataOperationEnum_t::Set)
            {
                dst_buf.template Set<dst_vector_t>(
                    dst_coord_.GetOffset(),
                    is_dst_valid,
                    dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);
            }
            else if constexpr(DstInMemOp == InMemoryDataOperationEnum_t::AtomicAdd)
            {
                dst_buf.template AtomicAdd<dst_vector_t>(
                    dst_coord_.GetOffset(),
                    is_dst_valid,
                    dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);
            }
            else if constexpr(DstInMemOp == InMemoryDataOperationEnum_t::Add)
            {

                typename vector_type_maker<DstData, DstScalarPerVector>::type tmp;
                tmp.template AsType<dst_vector_t>()(Number<0>{}) =
                    dst_buf.template Get<dst_vector_t>(dst_coord_.GetOffset(), is_dst_valid);

                static_for<0, DstScalarPerVector, 1>{}([&](auto t) {
                    dst_vector.template AsType<DstData>()(t) += tmp.template AsType<DstData>()[t];
                });

                dst_buf.template Set<dst_vector_t>(
                    dst_coord_.GetOffset(),
                    is_dst_valid,
                    dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);
            }

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_forward_steps[dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_backward_steps[dim_access_order[i]]);
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

    template <typename SrcSliceOriginIdx, typename Src0Buffer, typename DstBuffer>
    __device__ void Run(const Src0Desc&,
                        const SrcSliceOriginIdx&,
                        const Src0Buffer& src0_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        Run(Src0Desc{}, SrcSliceOriginIdx{}, src0_buf, dst_desc, dst_buf, dst_step_hacks);
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_access_lengths[j] + ordered_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate dst data index after last iteration in Run(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
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
    SrcElementwiseOperation src_element_op_;
}; // namespace ck

} // namespace ck
#endif
