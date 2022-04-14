#ifndef CK_THREADWISE_TENSOR_SLICE_TRANSFER_AVX2_HPP
#define CK_THREADWISE_TENSOR_SLICE_TRANSFER_AVX2_HPP

#include "common_header.hpp"
#include "data_type_cpu.hpp"
#include "../../gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_space_filling_curve.hpp"
#include "dynamic_buffer_cpu.hpp"
#include <immintrin.h>

namespace ck {
namespace cpu {

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. src_slice_origin and dst_slice_origin are not known at compile-time,
//   3. always use __mm256 register to hold continuous 8 dword, so if fast-changing
//      dim is a complex dimension, better re-consider layout (e.g NCHW is not good if non 1x1)
//   4. RunGeneric() can handle any case (by not using ymm), but performance are not guranteed
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector, // src/dst must use same vector size, aka src/dst both need same
                                   // avx/float register
          InMemoryDataOperationEnum DstInMemOp,
          bool SrcResetCoordinateAfterRun,
          bool DstResetCoordinateAfterRun>
struct ThreadwiseTensorSliceTransferAvx2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    constexpr ThreadwiseTensorSliceTransferAvx2(const SrcDesc& src_desc,
                                                const Index& src_slice_origin,
                                                const DstDesc& dst_desc,
                                                const Index& dst_slice_origin,
                                                const ElementwiseOperation& element_op)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          element_op_(element_op)
    {
        static_assert(SliceLengths::At(Number<VectorDim>{}) % ScalarPerVector == 0,
                      "wrong! cannot evenly divide");
    }

    void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        // In GPU this function is used for set per-thread index based on threadIdx.x
        // But for CPU, no need to call this function.
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, typename DstBuffer>
    void RunGeneric(const SrcDesc& src_desc,
                    const SrcBuffer& src_buf,
                    const DstDesc& dst_desc,
                    DstBuffer& dst_buf)
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto scalar_per_access = generate_sequence(
            ck::detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(scalar_per_access)>>;

        // loop over space-filling curve
        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        // std::cout<<"num_access:"<<num_access<<std::endl;

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            using src_vector_type = ck::cpu::vector_type_maker_t<SrcData, ScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            using dst_vector_type = ck::cpu::vector_type_maker_t<DstData, ScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // printf("[%s] ", is_src_valid ? "y":"n");
            // print_multi_index(src_coord_.GetIndex());
            // printf("----");
            // print_multi_index(src_coord_.GetHiddenIndex());

            // printf(":%d", src_coord_.GetOffset());
            // printf("\n");

            // copy data from src_buf into src_vector_container
            auto src_vector_container = src_vector_type{
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid)};

            auto dst_vector_container = dst_vector_type{};

            // apply pointwise operation
            // static_for<0, ScalarPerVector, 1>{}([&](auto i) {
            //     element_op_(dst_vector_container.template AsType<DstData>()(i),
            //                 src_vector_container.template AsType<SrcData>()[i]);
            // });
            element_op_(dst_vector_container.template AsType<dst_vector_t>(),
                        src_vector_container.template AsType<src_vector_t>());

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // printf(" -> ");
            // print_multi_index(dst_coord_.GetIndex());
            // printf(":%d", dst_coord_.GetOffset());

            // printf(", src:0x%x, dst:0x%x",
            // *reinterpret_cast<uint32_t*>(&src_vector_container.template AsType<src_vector_t>()),
            //                               *reinterpret_cast<uint32_t*>(&dst_vector_container.template
            //                               AsType<dst_vector_t>()));
            // printf("\n");

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>());

            // move coordinate
            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);
                move_tensor_coordinate(
                    src_desc, src_coord_, make_tensor_coordinate_step(src_desc, forward_step));
                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }

        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    static constexpr auto GetCoordinateResetStep()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(scalar_per_access)>>;

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

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = SrcResetCoordinateAfterRun
                                           ? src_slice_origin_step_idx
                                           : src_slice_origin_step_idx + GetCoordinateResetStep();

        printf(" GetCoordinateResetStep:");
        print_multi_index(GetCoordinateResetStep());

        printf(" adjusted_step_idx:");
        print_multi_index(adjusted_step_idx);

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        printf(" adjusted_step:");
        print_multi_index(adjusted_step.GetIndexDiff());
        printf("\n");

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx = DstResetCoordinateAfterRun
                                           ? dst_slice_origin_step_idx
                                           : dst_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    SrcCoord src_coord_;
    DstCoord dst_coord_;
    const ElementwiseOperation element_op_;
};

} // namespace cpu
} // namespace ck

#endif
