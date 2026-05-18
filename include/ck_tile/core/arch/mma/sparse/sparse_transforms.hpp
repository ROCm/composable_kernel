// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_transforms.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include <cstdint>

namespace ck_tile::core::arch::mma {

namespace sparse::detail {
/**
 * @brief Compress A vector for 2:4 structured sparsity instruction by moving all non-zero
 * elements into lower part of a_vec to half its effective size.
 * @param a_vec Vector to be compressed.
 * @tparam ADataType The data type of a_vec
 * @tparam CompressedSize The target compression size
 * @tparam AVec The vector type of a_vec (deduced)
 * @return Packed 32‑bit word containing **CompressedSize** 2‑bit fields.
 *         Each field encodes the original position (0–3) of the corresponding
 *         non‑zero element in the input. If fewer than CompressedSize
 *         non‑zeros are found, remaining fields default to 2 (see below).
 */
template <typename ADataType, index_t CompressedSize, typename AVec>
static CK_TILE_DEVICE int32_t compress_a_impl(AVec& a_vec)
{
    // idx holds one 2‑bit index per output element (total CompressedSize entries).
    // It is initialized to the pattern 0b10 for every field. This matches
    // what the hardware expects when there are fewer than two non‑zero values
    // in a 4‑element group – the unused output is treated as coming from slot 2.
    // The loop below will clear and set each field as real non‑zeros are seen.
    int32_t idx = 0;
    static_for<0, CompressedSize, 1>{}([&](auto k) { idx |= (2u << (2u * k)); });

    static_for<0, CompressedSize / 2, 1>{}([&](auto i) {
        ADataType nonzero_elems[2] = {a_vec[i * 4 + 2], a_vec[i * 4 + 3]};
        int32_t non_zero_pos       = 0;

        static_for<0, 4, 1>{}([&](auto j) {
            if(static_cast<float>(a_vec[i * 4 + j]) != 0.0f)
            {
                nonzero_elems[non_zero_pos] = a_vec[i * 4 + j];
                // clear the two‑bit field for this output and insert j
                idx &= ~(0b11u << (2u * (i * 2 + non_zero_pos)));
                idx |= static_cast<uint32_t>(j) << (2u * (i * 2 + non_zero_pos));
                ++non_zero_pos;
            }
        });
        a_vec[i * 2]     = nonzero_elems[0];
        a_vec[i * 2 + 1] = nonzero_elems[1];
    });

    return idx;
}
} // namespace sparse::detail

/**
 * @class SparseCompressTransform
 * @brief Performs 2:4 structured sparsity compression to the vector v and produces an index mask.
 * @note  Returns a tuple of two. The first element is the vector v with the same scalar type but
 *        its size halved. The second element is the index mask.
 */
template <index_t CompressionRatio>
struct SparseCompressTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType& v)
    {
        using VecTraits                         = vector_traits<remove_cvref_t<VecType>>;
        using ScalarT                           = typename VecTraits::scalar_type;
        static constexpr auto VecN              = VecTraits::vector_size;
        static constexpr index_t CompressedSize = VecN / CompressionRatio;
        using VecCompressed                     = ext_vector_t<ScalarT, CompressedSize>;

        static_assert(VecN % CompressionRatio == 0, "VecN must be divisible by CompressionRatio");
        static_assert(CompressedSize > 0, "CompressedSize must be > 0");

        const auto idx = sparse::detail::compress_a_impl<ScalarT, CompressedSize>(v);

        // TODO c++20: Use bit_cast
        return std::tuple<VecCompressed&, int32_t>(
            *std::launder(reinterpret_cast<VecCompressed*>(&v)), idx);
    }
};

/**
 * @class MmaDefaultTransformsSparse
 * @brief Implements the default transforms for Sparse
 *
 * For 2:4 structured sparsity with inline register metadata:
 *  - ATransform: 2:4 structured sparsity compression
 *  - BTransform: Pass-through (sparse operands already formatted)
 *  - CTransform: Pass-through (input accumulator)
 *  - DTransform: Pass-through (output accumulator as-is)
 */
template <index_t CompressionRatio>
struct MmaDefaultTransformsSparse
{
    using ATransform = SparseCompressTransform<CompressionRatio>;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 * @class MmaTransformsDefaultSelector
 * @brief Specialization for Sparse MFMA transforms
 *        Provides default transform selection for sparse operations
 *
 * @tparam MmaOp Sparse MMA operation
 * @tparam CompilerTarget The compiler target
 */
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target CompilerTarget>
// TODO: c++20 requires(is_mma_op_sparse(MmaOp))
template <typename MmaOp, typename CompilerTarget>
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    std::enable_if_t<MmaOp::OpFamily == MmaOpFamily::SPARSE>>
{
    using SelectedTransforms = MmaDefaultTransformsSparse<MmaOp::kCompressionRatio>;
};

} // namespace ck_tile::core::arch::mma
