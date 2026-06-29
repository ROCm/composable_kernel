// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif

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
    CK_TILE_DEVICE static decltype(auto) exec(ATensor& a, BTensor& b, CTensor& accum)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            if constexpr(Derived::CTranspose)
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(b);
                decltype(auto) b_transformed = Derived::BTransform::exec(a);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<Params...>(a_transformed, b_transformed, c_transformed);
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

    // Entry point for dense and sparse operations. TODO: Add c_vec = a_vec * b_vec variant.
    // TODO: Parse params with WarpGemmParamsParser<>
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c, ATensor& a, const BTensor& b) const
    {
        exec<Params...>(a, b, c);
    }

    template <typename... Params,
              typename ATensor,
              typename BTensor,
              typename CTensor,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static decltype(auto)
    exec(ATensor& a, BTensor& b, CTensor& accum, ScaleADataType& scale_A, ScaleBDataType& scale_B)
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
                    a_transformed, b_transformed, c_transformed, scale_A, scale_B);
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

    // Entry point for scale operations. TODO: Add c_vec = a_vec * b_vec variant (+ scaleless
    // variant?)
    // TODO: Add support for other scale types.
    // TODO: Parse params with WarpGemmParamsParser<>
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c,
                                   const ATensor& a,
                                   const BTensor& b,
                                   const int32_t& a_scale,
                                   const int32_t& b_scale) const
    {
        exec<Params...>(a, b, c, a_scale, b_scale);
    }
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept MmaPipelineI
 * @brief  Expresses the meta-data interface required for a CRTP MmaPipeline.
 */
template <typename Derived>
concept MmaPipelineInterface = std::derived_from<Derived, MmaPipelineBase<Derived>>;

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
