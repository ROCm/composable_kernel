// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector,
          bool SrcResetCoordinateAfterRun,
          bool DstResetCoordinateAfterRun>
struct ThreadwiseTensorSliceTransfer_v6r1r2
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    __device__ constexpr ThreadwiseTensorSliceTransfer_v6r1r2(
        const SrcDesc& src_desc,
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

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, typename DstBuffer, InMemoryDataOperationEnum DstInMemOp>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(scalar_per_access)>>;

        // loop over space-filling curve
        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            using src_vector_type = vector_type_maker_t<SrcData, ScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            using dst_vector_type = vector_type_maker_t<DstData, ScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf into src_vector_container
            auto src_vector_container = src_vector_type{
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid)};

            auto dst_vector_container = dst_vector_type{};

#if 1
            // Emin @debug
            // Debug: Print source vector data if valid
            if (threadIdx.x == 0 && threadIdx.y == 0 && is_src_valid) {
                // printf("Threadwise_tensor slice v6r1r2 line 108: Src Vector Data at idx %d: %f\n", static_cast<int>(idx_1d.value), static_cast<float>());
                
                // printf("BlockId %d -  Threadwise_tensor slice v6r1r2 line 111: Src Vector Data at idx %d: %f \n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value), static_cast<float>(src_vector_container.template AsType<SrcData>().At(Number<0>{})));

                // Trying alternative way instead of above
                uint16_t src_vector_container_bf16_value = src_vector_container.template AsType<SrcData>().At(Number<0>{}) ;
                uint32_t fp32_bits = static_cast<uint32_t>(src_vector_container_bf16_value) << 16 ;
                // float src_vector_container_fp32_value = *reinterpret_cast<float*>(&fp32_bits) ;
                float src_vector_container_fp32_value;
                memcpy(&src_vector_container_fp32_value, &fp32_bits, sizeof(float));

                printf("BlockId %d -  Threadwise_tensor slice v6r1r2 line 120: Src Vector Data at idx %d: %f \n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value), src_vector_container_fp32_value);

                // printf("Threadwise_tensor slice v6r1r2 line 108: Src Vector Data at idx %d: %hu \n", static_cast<int>(idx_1d.value), src_vector_container.template AsType<SrcData>().At(Number<0>{}));
            }
            // Emin @debug

#endif

            // apply pointwise operation
            static_for<0, ScalarPerVector, 1>{}([&](auto i) {
                SrcData v;

                // Emin @added
                

                // apply element-wise operation
                element_op_(v, src_vector_container.template AsType<SrcData>()[i]);

#if 1
                // Emin @debug
                // Debug: Print element-wise operation result
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    //printf("Threadwise_tensor slice v6r1r2 line 121 : Element-wise Operation Result at idx %d: %f\n", static_cast<int>(i.value), static_cast<float>(v));

                    uint16_t v_bf16_value  = v ;
                    uint32_t fp32_bits_v = static_cast<uint32_t>(v_bf16_value) << 16 ;

                    float v_fp32_value;
                    memcpy(&v_fp32_value, &fp32_bits_v, sizeof(float));

                    printf("Threadwise_tensor slice v6r1r2 line 147 : Element-wise Operation Result at idx %d: %f\n", static_cast<int>(i.value), v_fp32_value);

                }

                // Emin @added
                __syncthreads();

#endif

 // Emin @debug
#if 0
                 // Debug: Print SrcData before and after applying element-wise operation
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    // printf("Threadwise_tensor_slice_v6r1r2 line 127 : SrcData before element-wise op at idx %d: %f \n", static_cast<int>(i.value), static_cast<float>(src_vector_container.template AsType<SrcData>().At(Number<i>{})));
                    // printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 127 : SrcData before element-wise op at idx %d , i %d: %hu \n", static_cast<int>(blockIdx.x) ,  static_cast<int>(idx_1d.value),  static_cast<int>(i.value), src_vector_container.template AsType<SrcData>().At(Number<i>{}));
                    // // printf("SrcData after element-wise op at idx %d: %f \n", static_cast<int>(i.value), static_cast<float>(v));
                    // printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 129 : SrcData after element-wise op at idx %d , i %d: %hu \n" , static_cast<int>(blockIdx.x) ,  static_cast<int>(idx_1d.value) , static_cast<int>(i.value), v);
                
                    printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 165 : SrcData before element-wise op at idx %d , i %d: %f \n", static_cast<int>(blockIdx.x) ,  static_cast<int>(idx_1d.value),  static_cast<int>(i.value), src_vector_container_fp32_value);
                    // printf("SrcData after element-wise op at idx %d: %f \n", static_cast<int>(i.value), static_cast<float>(v));
                    printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 167 : SrcData after element-wise op at idx %d , i %d: %f \n" , static_cast<int>(blockIdx.x) ,  static_cast<int>(idx_1d.value) , static_cast<int>(i.value), v_fp32_value);
                }

                // Emin @added
                __syncthreads();
#endif
                // apply type convert
                dst_vector_container.template AsType<DstData>()(i) = type_convert<DstData>(v);

                // Emin @added
                __syncthreads();

#if 1
                // Emin @debug
                // Debug: Print type conversion result
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    // printf("Threadwise_tensor slice v6r1r2 line 121 : Type Conversion Result at idx %d: %f\n", static_cast<int>(i.value), static_cast<float>(dst_vector_container.template AsType<DstData>()[i]));
                    //   printf("DstData after type conversion at idx %d: %f \n", static_cast<int>(i.value), static_cast<float>(dst_vector_container.template AsType<DstData>().At(Number<i>{})));
                    // printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 140 : DstData after type conversion at idx %d, i  %d: %hu \n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value) , static_cast<int>(i.value), dst_vector_container.template AsType<DstData>().At(Number<i>{}));
                
                    uint16_t dst_vector_container_bf16_value  = dst_vector_container.template AsType<DstData>().At(Number<i>{}) ;
                    uint32_t fp32_bits_dst_vector_container = static_cast<uint32_t>(dst_vector_container_bf16_value) << 16 ;

                    float dst_vector_container_fp32_value;
                    memcpy(&dst_vector_container_fp32_value, &fp32_bits_dst_vector_container, sizeof(float));


                    //printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 140 : DstData after type conversion at idx %d, i  %d: %f \n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value) , static_cast<int>(i.value), static_cast<float>(dst_vector_container.template AsType<DstData>().At(Number<i>{})));
                    printf("BlockId %d -  Threadwise_tensor_slice_v6r1r2 line 140 : DstData after type conversion at idx %d, i  %d: %f \n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value) , static_cast<int>(i.value), dst_vector_container_fp32_value);
               
                }

                // Emin @added
                __syncthreads();

#endif
            });

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>()[I0]);

#if 0
            // Emin @debug
            //  // Debug: Print data before copying from dst_vector into dst_buf
            if (threadIdx.x == 0 && threadIdx.y == 0 && is_dst_valid) {
                // printf("Dst Vector Data being copied to dst_buf at idx %d: %v4hu", static_cast<int>(idx_1d.value), dst_buf.template AsType<DstData>().At(I0));
                // printf("BlockId %d -  Dst Vector Data being copied to dst_buf at idx %d: %hu\n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value), dst_buf.template Get<dst_vector_t>(dst_coord_.GetOffset(), is_dst_valid));

                printf("BlockId %d -  Dst Vector Data being copied to dst_buf at idx %d: %hu\n", static_cast<int>(blockIdx.x) , static_cast<int>(idx_1d.value), dst_buf.template Get<dst_vector_t>(dst_coord_.GetOffset(), is_dst_valid));
            }

#endif

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

    __device__ static constexpr auto GetCoordinateResetStep()
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
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = SrcResetCoordinateAfterRun
                                           ? src_slice_origin_step_idx
                                           : src_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
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

} // namespace ck
