// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

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
 *                 - Type aliases: @c InternalAVecT, @c InternalBVecT, @c InternalCVecT,
 *                   @c CVecType, @c MmaOp
 *                 - Transform aliases: @c ATransform, @c BTransform, @c CTransform,
 *                   @c DTransform
 *                 - A static @c execImpl(std::tuple<A,B,C>&) method.
 *
 * @par The pipeline performs the following steps in @c exec():
 *      1. Apply pre-transforms and format input buffers (A, B, C).
 *      2. Delegate to @c Derived::execImpl for the actual mma loop.
 *      3. Apply post-transform and format the output buffer (D) back to the user type.
 *      When @c ABSwap is set, the A and B inputs are swapped before step 1.
 */
// TODO: c++20: use MmaPipelineOptionFlags directly
template <MmaPipelineOptionFlags::Type Flags_, typename Derived>
struct MmaPipelineBase
{
    static constexpr auto Flags = MmaPipelineOptionFlags(Flags_);

    private:
    /**
     * @brief Reconstruct a tuple with its first element passed through @c formatBuffer
     *        while preserving all remaining elements unchanged.
     * @tparam DstT  Target type for the formatted first element.
     * @tparam SrcT  Forwarding-reference type of the input tuple.
     * @tparam Is    Index pack for elements 1..N-1 of the tuple.
     * @param  inputTuple The source tuple whose first element will be formatted.
     * @return A new tuple with the formatted first element and the remaining elements forwarded.
     */
    template <typename DstT, typename SrcT, std::size_t... Is>
    CK_TILE_DEVICE static auto formatBufferTupleImpl(SrcT&& inputTuple, std::index_sequence<Is...>)
    {
        auto&& first_elem = std::get<0>(std::forward<SrcT>(inputTuple));
        using FirstElemResultType =
            decltype(formatBuffer<DstT>(std::forward<decltype(first_elem)>(first_elem)));
        using InputTupleType = ck_tile::remove_cvref_t<SrcT>;
        return std::tuple<FirstElemResultType, std::tuple_element_t<Is + 1, InputTupleType>...>(
            formatBuffer<DstT>(std::forward<decltype(first_elem)>(first_elem)),
            std::get<Is + 1>(std::forward<SrcT>(inputTuple))...);
    }

    /**
     * @brief Format (reinterpret-cast) a buffer to the hardware-native vector type @p DstT.
     *
     * Three cases are handled:
     * - **Tuple**: recursively format the first element via @c formatBufferTupleImpl,
     *   preserving any metadata in the remaining tuple elements.
     * - **Array / Pointer**: forwarded unchanged.
     * - **Scalar / Vector**: reinterpret-cast to @p DstT (sizes must match).
     *
     * @tparam DstT        The target hardware vector type.
     * @tparam SrcT        Forwarding-reference type of the input buffer.
     * @param  inputBuffer The buffer to format.
     * @return A reference (or value) of type @p DstT corresponding to @p inputBuffer.
     */
    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static decltype(auto) formatBuffer(SrcT&& inputBuffer)
    {
        using DecayedSrcT = ck_tile::remove_cvref_t<SrcT>;

        // If SrcT is a tuple, extract the first element (the vector) and format it
        // while preserving all remaining elements (metadata)
        if constexpr(is_std_tuple_v<DecayedSrcT>)
        {
            // Create index sequence for all remaining elements (skip first)
            constexpr std::size_t tuple_size = std::tuple_size_v<DecayedSrcT>;
            return formatBufferTupleImpl<DstT>(std::forward<SrcT>(inputBuffer),
                                               std::make_index_sequence<tuple_size - 1>{});
        }
        else if constexpr(std::is_array_v<DecayedSrcT> || std::is_pointer_v<DecayedSrcT>)
        {
            return std::forward<SrcT>(inputBuffer);
        }
        else
        {
            static_assert(sizeof(DstT) == sizeof(DecayedSrcT), "Size mismatch in formatBuffer");

            using QualifiedDstT =
                std::conditional_t<std::is_const_v<DecayedSrcT>, DstT const, DstT>;

            return reinterpret_cast<QualifiedDstT&>(inputBuffer);
        }
    }

    protected:
    /** @brief Query whether a specific @ref MmaPipelineOptionFlag is set. */
    template <MmaPipelineOptionFlag Flag>
    constexpr CK_TILE_DEVICE static bool hasFlag()
    {
        return Flags.testFlag(Flag);
    }

    /**
     * @brief Apply a transform **then** format the result to @p DstT.
     *        Used for input operands (A, B, C) before the mma loop.
     */
    template <typename DstT, typename Transform, typename... Args>
    CK_TILE_DEVICE static auto preApplyTransform(Args&&... args)
    {
        return formatBuffer<DstT>(Transform::exec(std::forward<Args>(args)...));
    }

    /**
     * @brief Format a buffer to @p DstT **then** apply a transform.
     *        Used for the output operand (D) after the mma loop.
     */
    template <typename DstT, typename Transform, typename... Args>
    CK_TILE_DEVICE static auto postApplyTransform(Args&&... args)
    {
        return Transform::exec(formatBuffer<DstT>(std::forward<Args>(args)...));
    }

    /**
     * @brief Apply the per-operand pre-transforms and buffer formatting to A, B, and C.
     * @return A @c std::tuple of the transformed (A, B, C, [scaleA, scaleB]) vectors ready for the
     * mma loop.
     */
    template <typename ATransformInputs,
              typename BTransformInputs,
              typename CTransformInputs,
              typename... ExtraArgs>
    CK_TILE_DEVICE static decltype(auto) applyTransformsToInputs(ATransformInputs&& a,
                                                                 BTransformInputs&& b,
                                                                 CTransformInputs&& accum,
                                                                 ExtraArgs&&... extras)
    {
        using InternalAVecT = typename Derived::InternalAVecT;
        using InternalBVecT = typename Derived::InternalBVecT;
        using InternalCVecT = typename Derived::InternalCVecT;

        using ATransform = typename Derived::ATransform;
        using BTransform = typename Derived::BTransform;
        using CTransform = typename Derived::CTransform;

        return std::make_tuple(
            preApplyTransform<InternalAVecT, ATransform>(std::forward<ATransformInputs>(a)),
            preApplyTransform<InternalBVecT, BTransform>(std::forward<BTransformInputs>(b)),
            preApplyTransform<InternalCVecT, CTransform>(std::forward<CTransformInputs>(accum)),
            std::forward<ExtraArgs>(extras)...);
    }

    /**
     * @brief Apply the post-transform and buffer formatting to the C (accumulator) output.
     * @param c_result The accumulator to post-process.
     * @return The final D output in the user-facing vector type.
     */
    template <typename CTransformResult>
    CK_TILE_DEVICE static auto applyTransformToOutput(CTransformResult&& c_result)
    {
        static_assert(!is_std_tuple_v<decltype(c_result)>,
                      "If CTransform returns more than the vector, update this function.");

        using CVecT      = typename Derived::CVecType;
        using DTransform = typename Derived::DTransform;
        return postApplyTransform<CVecT, DTransform>(c_result);
    }

    public:
    /**
     * @brief Entry point: execute the full Mma pipeline (transforms + mma loop + output).
     * @tparam VecTA Type of the A WaveTile buffer.
     * @tparam VecTB Type of the B WaveTile buffer.
     * @tparam VecTC Type of the C (accumulator) WaveTile buffer.
     * @param  a     Input WaveTile A.
     * @param  b     Input WaveTile B.
     * @param  accum Input/output accumulator WaveTile C.
     * @return The output WaveTile D after accumulation and post-transform.
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            constexpr bool swap_a_and_b = hasFlag<MmaPipelineOptionFlag::ABSwap>();

            auto transformed_inputs = [&]() {
                if constexpr(swap_a_and_b)
                {
                    return applyTransformsToInputs(
                        std::forward<VecTB>(b), std::forward<VecTA>(a), std::forward<VecTC>(accum));
                }
                else
                {
                    return applyTransformsToInputs(
                        std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
                }
            }();

            Derived::execImpl(transformed_inputs);

            auto&& [a_result, b_result, c_result] = std::move(transformed_inputs);
            return applyTransformToOutput(std::move(c_result));
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

    template <typename VecTA,
              typename VecTB,
              typename VecTC,
              typename ScaleADataType,
              typename ScaleBDataType>
    CK_TILE_DEVICE static decltype(auto)
    exec(VecTA&& a, VecTB&& b, VecTC&& accum, ScaleADataType&& scale_A, ScaleBDataType&& scale_B)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            static_assert(MmaOpTraits<typename Derived::MmaOp>::IsScale,
                          "This exec variant is intended for scale policy structs");
            constexpr bool swap_a_and_b = hasFlag<MmaPipelineOptionFlag::ABSwap>();

            auto transformed_inputs = applyTransformsToInputs(
                swap_a_and_b ? std::forward<VecTB>(b) : std::forward<VecTA>(a),
                swap_a_and_b ? std::forward<VecTA>(a) : std::forward<VecTB>(b),
                std::forward<VecTC>(accum),
                swap_a_and_b ? std::forward<ScaleBDataType>(scale_B)
                             : std::forward<ScaleADataType>(scale_A),
                swap_a_and_b ? std::forward<ScaleADataType>(scale_A)
                             : std::forward<ScaleBDataType>(scale_B));

            Derived::execImpl(transformed_inputs);

            auto&& [a_result, b_result, c_result, scale_A_result, scale_B_result] =
                std::move(transformed_inputs);
            return applyTransformToOutput(std::move(c_result));
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
