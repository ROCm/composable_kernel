// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

#include "amdgcn_mma.hpp"
#include "mma_selector.hpp"
#include "mma_traits.hpp"
#include "mma_transforms.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile::core::arch::mma {

/*! @enum MmaPipelineOptionFlag
 * @brief Individual option flags for configuring MmaPipeline behavior.
 */
enum struct MmaPipelineOptionFlag : unsigned
{
    NONE       = 0x0, ///< No flags set
    ABSwap     = 0x1, ///< Swap A and B inputs to transpose the C output
    COMPRESS_A = 0x2, ///< Enable compressed (sparse) A matrix input
};

/**
 * @struct MmaPipelineOptionFlags
 * @brief  Type-safe bitmask wrapper for combining @ref MmaPipelineOptionFlag values.
 * @par    Provides bitwise OR, AND, NOT, and equality operators for composing
 *         and querying pipeline option flags.
 */
struct MmaPipelineOptionFlags
{
    using Type = std::underlying_type_t<MmaPipelineOptionFlag>;

    explicit constexpr MmaPipelineOptionFlags() : mFlags(0) {}
    explicit constexpr MmaPipelineOptionFlags(Type value) : mFlags(value) {}
    constexpr MmaPipelineOptionFlags(MmaPipelineOptionFlag singleFlag) : mFlags(toType(singleFlag))
    {
    }
    constexpr MmaPipelineOptionFlags(const MmaPipelineOptionFlags& original)
        : mFlags(original.mFlags)
    {
    }

    constexpr MmaPipelineOptionFlags& operator|=(MmaPipelineOptionFlag addValue)
    {
        mFlags |= toType(addValue);
        return *this;
    }
    constexpr MmaPipelineOptionFlags operator|(MmaPipelineOptionFlag addValue) const
    {
        MmaPipelineOptionFlags result(*this);
        result |= addValue;
        return result;
    }
    constexpr MmaPipelineOptionFlags& operator&=(MmaPipelineOptionFlag maskValue)
    {
        mFlags &= toType(maskValue);
        return *this;
    }
    constexpr MmaPipelineOptionFlags operator&(MmaPipelineOptionFlag maskValue) const
    {
        MmaPipelineOptionFlags result(*this);
        result &= maskValue;
        return result;
    }
    constexpr MmaPipelineOptionFlags operator~() const
    {
        MmaPipelineOptionFlags result(*this);
        result.mFlags = ~result.mFlags;
        return result;
    }
    constexpr bool testFlag(MmaPipelineOptionFlag flag) const
    {
        return (flag == MmaPipelineOptionFlag::NONE) ? mFlags == toType(flag) : *this & flag;
    }
    constexpr operator bool() const { return mFlags != toType(MmaPipelineOptionFlag::NONE); }
    constexpr bool operator==(Type rhs) const { return mFlags == rhs; }

    private:
    Type mFlags;
    static constexpr Type toType(MmaPipelineOptionFlag f) { return static_cast<Type>(f); }
};

constexpr bool operator==(MmaPipelineOptionFlags::Type lhs, const MmaPipelineOptionFlags& rhs)
{
    return rhs == lhs;
}

/**
 * @class  MmaPipelineBase
 * @brief  CRTP base class that implements the common Mma pipeline logic shared by
 *         all concrete pipeline drivers (e.g., dense wave-wise, sparse, etc.).
 *
 * @tparam Flags_  Compile-time bitmask of @ref MmaPipelineOptionFlag controlling
 *                 pipeline behavior (e.g., C transposition, A compression).
 * @tparam Derived The concrete CRTP-derived pipeline class. Must expose:
 *                 - Type aliases: @c AWarpTensor, @c BWarpTensor, @c CWarpTensor, @c MmaOp
 *                 - Transform aliases: @c ATransform, @c BTransform, @c CTransform, @c DTransform
 *                 - A static @c execImpl(std::tuple<A,B,C>&) method.
 *
 * @par The pipeline performs the following steps in @c exec():
 *      1. Apply pre-transforms to input buffers (A, B, C).
 *      2. Delegate to @c Derived::execImpl for the actual mma loop.
 *      3. Apply post-transform to output buffer (D).
 *      When @c ABSwap is set, the A and B inputs are swapped before step 1.
 */
// TODO: c++20: use MmaPipelineOptionFlags directly
template <MmaPipelineOptionFlags::Type Flags_, typename Derived>
struct MmaPipelineBase
{
    static constexpr auto Flags = MmaPipelineOptionFlags(Flags_);

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
    template <typename ATensor, typename BTensor, typename CTensor>
    CK_TILE_DEVICE static decltype(auto) exec(ATensor& a, BTensor& b, CTensor& accum)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            if constexpr(Flags & MmaPipelineOptionFlag::ABSwap)
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(b);
                decltype(auto) b_transformed = Derived::BTransform::exec(a);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::execImpl(a_transformed, b_transformed, c_transformed);
                return Derived::DTransform::exec(c_transformed);
            }
            else
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(a);
                decltype(auto) b_transformed = Derived::BTransform::exec(b);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::execImpl(a_transformed, b_transformed, c_transformed);
                return Derived::DTransform::exec(c_transformed);
            }
        }
        else
        {
            // Return the unsupported exec. This should print a runtime warning. (amdgcn_mma.hpp)
            // Code should not reach here, but HOST/DEVICE compile passes are
            // weirdly intertwined and instead of having constexpr in the calling
            // site (tests) we do this. See also changes by this commit.
            return Derived::MmaOp::exec({}, {}, {});
        }
    }

    // Entry point for dense and sparse operations. TODO: Add c_vec = a_vec * b_vec variant.
    // TODO: Parse params with WarpGemmParamsParser<>
    template <typename... Params, typename CTensor, typename ATensor, typename BTensor>
    CK_TILE_DEVICE void operator()(CTensor& c, ATensor& a, const BTensor& b) const
    {
        exec(a, b, c);
    }

    template <index_t opselA,
              index_t opselB,
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
            if constexpr(Flags & MmaPipelineOptionFlag::ABSwap)
            {
                // TODO: Figure out which combination of a/b, scale_A/B, and opselA/B needs to be
                // AB-swapped in order to get correct results. Note that WarpGemmParamsParser
                // already seems to swap opselA and B.
                decltype(auto) a_transformed = Derived::ATransform::exec(b);
                decltype(auto) b_transformed = Derived::BTransform::exec(a);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<opselA, opselB>(
                    a_transformed, b_transformed, c_transformed, scale_A, scale_B);
                return Derived::DTransform::exec(c_transformed);
            }
            else
            {
                decltype(auto) a_transformed = Derived::ATransform::exec(a);
                decltype(auto) b_transformed = Derived::BTransform::exec(b);
                decltype(auto) c_transformed = Derived::CTransform::exec(accum);
                Derived::template execImpl<opselA, opselB>(
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
        using P = WarpGemmParamsParser<Params...>;
        exec<P::op_sel_a, P::op_sel_b>(a, b, c, a_scale, b_scale);
    }
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

#include <concepts>

/**
 * @concept MmaPipelineI
 * @brief  Expresses the meta-data interface required for a CRTP MmaPipeline.
 */
template <typename Derived, MmaPipelineOptionFlags::Type Flags>
concept MmaPipelineInterface = std::derived_from<Derived, MmaPipelineBase<Flags, Derived>>;

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
