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

/// Number of int32_t words needed to store CompressedSize 2-bit idx fields.
template <index_t CompressedSize>
static constexpr index_t idx_words_needed = (CompressedSize * 2 + 31) / 32;

/**
 * @class SparseIdxPack
 * @brief Variable-length container for 2:4 structured sparsity index metadata.
 *
 * Each compressed element produces a 2-bit index field encoding the original
 * position (0-3) within its group of 4.  When composing multiple MMA fragments
 * in M and K dimensions within a WaveTile, the total number of index bits can
 * exceed 32.  This struct packs the index fields into an array of int32_t words,
 * sized at compile time.
 *
 * @tparam NumWords Number of int32_t words needed to store all index fields.
 */
template <index_t NumWords>
struct SparseIdxPack
{
    static_assert(NumWords > 0, "SparseIdxPack requires at least 1 word");
    int32_t words[NumWords] = {};
};

/**
 * @brief Compress A vector for 2:4 structured sparsity instruction by moving all non-zero
 * elements into lower part of a_vec to half its effective size.
 * @param a_vec Vector to be compressed.
 * @tparam ADataType The data type of a_vec
 * @tparam CompressedSize The target compression size
 * @tparam AVec The vector type of a_vec (deduced)
 * @return SparseIdxPack containing **CompressedSize** 2-bit fields packed
 *         across one or more int32_t words.  Each field encodes the original
 *         position (0-3) of the corresponding non-zero element in the input.
 *         If fewer than CompressedSize non-zeros are found, remaining fields
 *         default to 2 (see below).
 */
template <typename ADataType, index_t CompressedSize, typename AVec>
static CK_TILE_DEVICE auto compress_a_impl(AVec& a_vec)
{
    static constexpr index_t NumIdxWords = idx_words_needed<CompressedSize>;
    // idx holds one 2-bit index per output element (total CompressedSize entries),
    // packed across NumIdxWords int32_t words.
    // It is initialized to the pattern 0b10 for every field. This matches
    // what the hardware expects when there are fewer than two non-zero values
    // in a 4-element group - the unused output is treated as coming from slot 2.
    // The loop below will clear and set each field as real non-zeros are seen.
    SparseIdxPack<NumIdxWords> idx{};
    static_for<0, CompressedSize, 1>{}([&](auto k) {
        constexpr uint32_t bit_pos = static_cast<uint32_t>(k) * 2u;
        constexpr uint32_t word    = bit_pos / 32u;
        constexpr uint32_t shift   = bit_pos % 32u;
        idx.words[word] |= static_cast<int32_t>(2u << shift);
    });

    static_for<0, CompressedSize / 2, 1>{}([&](auto i) {
        ADataType nonzero_elems[2] = {a_vec[i * 4 + 2], a_vec[i * 4 + 3]};
        int32_t non_zero_pos       = 0;

        static_for<0, 4, 1>{}([&](auto j) {
            if(static_cast<float>(a_vec[i * 4 + j]) != 0.0f)
            {
                nonzero_elems[non_zero_pos] = a_vec[i * 4 + j];
                // clear the two-bit field for this output and insert j
                const uint32_t field_idx =
                    static_cast<uint32_t>(i) * 2u + static_cast<uint32_t>(non_zero_pos);
                const uint32_t bit_pos = field_idx * 2u;
                const uint32_t word    = bit_pos / 32u;
                const uint32_t shift   = bit_pos % 32u;
                idx.words[word] &= ~static_cast<int32_t>(0b11u << shift);
                idx.words[word] |= static_cast<int32_t>(static_cast<uint32_t>(j) << shift);
                ++non_zero_pos;
            }
        });
        a_vec[i * 2]     = nonzero_elems[0];
        a_vec[i * 2 + 1] = nonzero_elems[1];
    });

    return idx;
}
/**
 * @brief Extract the per-fragment sparsity index from a packed idx pack.
 * After whole-wave-tile compression, the returned idx packs 2-bit fields for
 * every compressed output element across one or more int32_t words.
 * @return A single int32_t with this fragment's 2-bit fields at the
 *         least-significant positions, suitable for passing to the MMA builtin.
 */
template <uint32_t FragCompressedSize, uint32_t FragsK, index_t NumIdxWords>
static CK_TILE_DEVICE int32_t extract_fragment_idx(const SparseIdxPack<NumIdxWords>& idx,
                                                   uint32_t m,
                                                   uint32_t k)
{
    static constexpr uint32_t IdxBitsPerFrag = FragCompressedSize * 2;
    const auto fragLinearIdx                 = m * FragsK + k;
    const auto totalBitOffset                = fragLinearIdx * IdxBitsPerFrag;
    const auto wordIdx                       = totalBitOffset / 32u;
    const auto bitInWord                     = totalBitOffset % 32u;

    uint32_t result = static_cast<uint32_t>(idx.words[wordIdx]) >> bitInWord;

    // If fragment bits span a word boundary, stitch in bits from the next word.
    // (This is a safety measure; it should not occur when IdxBitsPerFrag is a
    // power-of-2 divisor of 32, which is always the case for current MMA ops.)
    if constexpr(NumIdxWords > 1)
    {
        if(bitInWord != 0 && bitInWord + IdxBitsPerFrag > 32u)
        {
            result |= static_cast<uint32_t>(idx.words[wordIdx + 1]) << (32u - bitInWord);
        }
    }

    return static_cast<int32_t>(result);
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
        using IdxType =
            sparse::detail::SparseIdxPack<sparse::detail::idx_words_needed<CompressedSize>>;

        static_assert(VecN % CompressionRatio == 0, "VecN must be divisible by CompressionRatio");
        static_assert(CompressedSize > 0, "CompressedSize must be > 0");

        auto idx = sparse::detail::compress_a_impl<ScalarT, CompressedSize>(v);

        return std::tuple<VecCompressed&, IdxType>(*ck_tile::bit_cast<VecCompressed*>(&v), idx);
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
