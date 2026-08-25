// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"

#include "amdgcn_mma.hpp"
#include "mma_selector.hpp"
#include "mma_traits.hpp"
#include "mma_transforms.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class  MmaPipelineBase
 * @brief  CRTP base class that implements the common Mma pipeline logic shared by
 *         all concrete pipeline drivers (e.g., dense wave-wise, sparse, etc.).
 *
 * @tparam Derived The concrete CRTP-derived pipeline class. Must expose:
 *                 - Type aliases: @c AWarpTensor, @c BWarpTensor, @c CWarpTensor, @c MmaOp
 *                 - Transform aliases: @c ATransform, @c BTransform, @c CTransform, @c DTransform
 *                 - A static @c execImpl(std::tuple<A,B,C>&) method.
 *
 * @par The pipeline performs the following steps in @c exec():
 *      1. Apply pre-transforms to input buffers (A, B, C).
 *      2. Delegate to @c Derived::execImpl for the actual mma loop.
 *      3. Apply post-transform to output buffer (D).
 *      When CTranspose is used, the A and B inputs are swapped before step 1.
 */
template <typename Derived>
struct MmaPipelineBase
{
    /**
     * @brief Entry point: execute the full Mma pipeline (transforms + mma loop + output).
     * @tparam ATensor Type of the A WaveTile tensor (static_distributed_tensor).
     * @tparam BTensor Type of the B WaveTile tensor (static_distributed_tensor).
     * @tparam CTensor Type of the C (accum) WaveTile tensor (static_distributed_tensor).
     * @param  a     Input WaveTile A.
     * @param  b     Input WaveTile B.
     * @param  accum Input/output accumulator WaveTile C.
     * @return The output WaveTile D after accumulation and post-transform.
     */
    template <typename... Params, typename ATensor, typename BTensor, typename CTensor>
    CK_TILE_DEVICE static decltype(auto) exec(const ATensor& a, const BTensor& b, CTensor& accum)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            if constexpr(Derived::CTranspose)
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(b);
                decltype(auto) b_transformed = Derived::BTransform::exec(a);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<Params..., SwapReuse_<true>>(
                    a_transformed, b_transformed, c_transformed);
                return Derived::DTransform::exec(c_transformed);
            }
            else
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(a);
                decltype(auto) b_transformed = Derived::BTransform::exec(b);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<Params...>(a_transformed, b_transformed, c_transformed);
                return Derived::DTransform::exec(c_transformed);
            }
        }
        else
        {
            // Return the unsupported exec. This should print a runtime warning. (amdgcn_mma.hpp)
            // Code should not reach here, but HOST/DEVICE compile passes are
            // weirdly intertwined and instead of having constexpr in the calling
            // site (tests) we do this. See also changes by this commit.
            return Derived::MmaOp::template exec<Params...>({}, {}, {});
        }
    }

    // CAB = (C, A, B).
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c, const ATensor& a, const BTensor& b) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<remove_cvref_t<CTensor>,
                                                               typename Derived::CWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<ATensor>,
                                                               typename Derived::AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>,
                                                               typename Derived::BWarpTensor>);
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsScale &&
                     MmaOpTraits<typename Derived::MmaOp>::IsMfma)
        {
            // GFX950 MFMA with (0,0) scale args
            exec<Params...>(a, b, c, 0, 0);
        }
        else
        {
            // GFX1250 WMMA with no scale args
            exec<Params...>(a, b, c);
        }
    }

    // AB = (A, B)
    // Same as CAB when C is not pre-existing
    template <typename... Params, typename ATensor, typename BTensor>
    CK_TILE_DEVICE auto operator()(const ATensor& a, const BTensor& b) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<remove_cvref_t<ATensor>,
                                                               typename Derived::AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>,
                                                               typename Derived::BWarpTensor>);
        typename Derived::CWarpTensor c;
        for(index_t i = 0; i < Derived::CWarpTensor::get_thread_buffer_size(); ++i)
        {
            c.get_thread_buffer()[i] = typename Derived::CDataType{0};
        }
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsScale &&
                     MmaOpTraits<typename Derived::MmaOp>::IsMfma)
        {
            exec<Params...>(a, b, c, 0, 0);
        }
        else
        {
            exec<Params...>(a, b, c);
        }
        return c;
    }

    template <typename... Params,
              typename ATensor,
              typename BTensor,
              typename CTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static decltype(auto) exec(ATensor& a,
                                              BTensor& b,
                                              CTensor& accum,
                                              const ScaleADataType& scale_A,
                                              const ScaleBDataType& scale_B)
    {
        static_assert(MmaOpTraits<typename Derived::MmaOp>::IsScale,
                      "This exec variant is intended for scale policy structs");

        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            if constexpr(Derived::CTranspose)
            {
                // TODO: Figure out which combination of a/b, scale_A/B, and opselA/B needs to be
                // AB-swapped in order to get correct results. Note that WarpGemmParamsParser
                // already seems to swap opselA and B.
                decltype(auto) a_transformed = Derived::ATransform::exec(b);
                decltype(auto) b_transformed = Derived::BTransform::exec(a);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<Params...>(
                    a_transformed, b_transformed, c_transformed, scale_B, scale_A);
                return Derived::DTransform::exec(c_transformed);
            }
            else
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(a);
                decltype(auto) b_transformed = Derived::BTransform::exec(b);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<Params...>(
                    a_transformed, b_transformed, c_transformed, scale_A, scale_B);
                return Derived::DTransform::exec(c_transformed);
            }
        }
        else
        {
            return Derived::MmaOp::exec({}, {}, {}); // Return unsupported exec. See comment above.
        }
    }

    // Scale operations
    // CABSS = (C, A, B, ScaleA, ScaleB)
    // TODO: Add support for other scale types.
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c,
                                   const ATensor& a,
                                   const BTensor& b,
                                   const int32_t& a_scale,
                                   const int32_t& b_scale) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<remove_cvref_t<CTensor>,
                                                               typename Derived::CWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<ATensor>,
                                                               typename Derived::AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>,
                                                               typename Derived::BWarpTensor>);
        exec<Params...>(a, b, c, a_scale, b_scale);
    }

    // ABSS = (A, B, ScaleA, ScaleB)
    // Same as CABSS, but C is not pre-existing
    template <typename... Params, typename ATensor, typename BTensor>
    CK_TILE_DEVICE auto operator()(const ATensor& a,
                                   const BTensor& b,
                                   const int32_t& a_scale,
                                   const int32_t& b_scale) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<remove_cvref_t<ATensor>,
                                                               typename Derived::AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<remove_cvref_t<BTensor>,
                                                               typename Derived::BWarpTensor>);
        typename Derived::CWarpTensor c;
        for(index_t i = 0; i < Derived::CWarpTensor::get_thread_buffer_size(); ++i)
        {
            c.get_thread_buffer()[i] = typename Derived::CDataType{0};
        }
        exec<Params...>(a, b, c, a_scale, b_scale);
        return c;
    }
};
} // namespace ck_tile::core::arch::mma
