#ifndef CK_THREADWISE_TENSOR_SLICE_TRANSFER_USING_SPACE_FILLING_CURVE_HPP
#define CK_THREADWISE_TENSOR_SLICE_TRANSFER_USING_SPACE_FILLING_CURVE_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_space_filling_curve.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

// namespace detail {
// // TODO: How to fix this? It uses an struct instead of lambda because lambda
// // doesn't have constructor
// template <index_t VectorDim, index_t ScalarPerVector>
// struct lambda_scalar_per_access
// {
//     __host__ __device__ constexpr auto operator()(index_t i) const
//     {
//         return (i == VectorDim) ? ScalarPerVector : 1;
//     }
// };
//
// template <index_t VectorDim>
// struct lambda_scalar_step_in_vector
// {
//     __host__ __device__ constexpr auto operator()(index_t i) const
//     {
//         return (i == VectorDim) ? 1 : 0;
//     }
// };
// } // namespace detail

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
          typename DstElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum_t DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3_using_space_filling_curve
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3_using_space_filling_curve(
        const DstDesc& dst_desc,
        const Index& dst_slice_origin_idx,
        const DstElementwiseOperation& dst_element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx)),
          dst_element_op_{dst_element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx,
              typename SrcBuffer,
              typename DstBuffer,
              typename DstStepHacks>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf,
                        const DstStepHacks& dst_step_hacks)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
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

        using SpaceFillingCurve =
            SpaceFillingCurve<SliceLengths, DimAccessOrder, remove_cv_t<decltype(dst_scalar_per_access)>>;

        constexpr auto num_accesses = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_accesses, 1>{}([&](auto idx_1d) {

            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector);
            typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;

            using dst_vector_t =
                typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

            // copy data from src_buf into dst_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                SrcData dst_v;

                // apply element-wise operation
                dst_element_op_(dst_v, src_buf[Number<src_offset>{}]);

                // apply type convert
                dst_vector.template AsType<DstData>()(i) = type_convert<DstData>(dst_v);
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
        constexpr index_t ntransform_dst = remove_cvref_t<DstDesc>::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        Run(SrcDesc{}, SrcSliceOriginIdx{}, src_buf, dst_desc, dst_buf, dst_step_hacks);
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

                static_for<1, i, 1>{}([&](auto j) {
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
    const DstElementwiseOperation dst_element_op_;
}; // struct ThreadwiseTensorSliceTransfer_v1r3_using_space_filling_curve

} // namespace ck
#endif

