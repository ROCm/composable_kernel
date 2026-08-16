// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_transforms.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/ext_vector_base.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

#include <tuple>
#include <type_traits>

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
// Packed-sub-byte awareness (bugs 2 & 3).
//
// CompressedSize here is measured in PHYSICAL storage elements of ADataType
// (e.g. whole pk_int4_t BYTES, each holding 2 packed 4-bit values). The
// original code indexed a_vec[] and ran its "group of 4 / keep <=2" scan at
// that PHYSICAL granularity -- correct for PackedSize==1 (int8/fp8/fp16/...),
// but WRONG for PackedSize>1 (pk_int4_t): structured 2:4 sparsity is defined
// over LOGICAL elements (4-bit values), not physical storage words. Verified
// against the RDNA4 ISA (Sec 7.12.3, "Wave32: each lane has 8 index values
// per lane" for a K=32 iu4 tile whose physical AVecType is only 4 BYTES --
// i.e. one idx field per LOGICAL nibble, not per physical byte) and via a
// device-side dump: real per-nibble 2:4 data makes EVERY physical byte in a
// group of 4 bytes "nonzero" (since a byte is nonzero whenever either of its
// two nibbles survived), systematically exceeding the old code's "<=2
// nonzero per group of 4" assumption and writing out of bounds into
// nonzero_elems[2]/[3] (undefined behavior -- observed as register aliasing
// that silently corrupted the first two survivors on real hardware).
// LogicalADataType defaults to
// ADataType (preserves old behavior for every existing/non-packed caller),
// but callers that know the TRUE logical element type of a packed vector
// (e.g. pk_int4_t, whose physical ext_vector storage is represented as plain
// int8_t -- see vector_traits<ext_vector_t<pk_int4_t,N>>::scalar_type in
// vector_type.hpp, which deliberately erases pk_int4_t to int8_t because
// clang's ext_vector_type attribute needs a true scalar) can pass it
// explicitly so PackedSize is read from the REAL logical type, not the
// erased physical one. Without this, `ADataType` here is ALWAYS int8_t for
// pk_int4_t vectors (confirmed: the packedness tag is lost one layer up, in
// SparseCompressTransform::execExtVec, well before reaching this function --
// an earlier version of this fix that only checked
// numeric_traits<ADataType>::PackedSize here had NO EFFECT for exactly this
// reason, verified by a before/after diff showing byte-identical output).
template <typename ADataType, index_t CompressedSize, typename AVec, typename LogicalADataType = ADataType>
static CK_TILE_DEVICE auto compress_a_impl(AVec& a_vec)
{
    static constexpr index_t PackedSize = numeric_traits<LogicalADataType>::PackedSize;
    // LOGICAL field count: equals CompressedSize when PackedSize==1 (fully
    // backward compatible with the original formula), doubles it for
    // pk_int4_t-style 2-packed types.
    static constexpr index_t LogicalCompressedSize = CompressedSize * PackedSize;
    static constexpr index_t NumIdxWords            = idx_words_needed<LogicalCompressedSize>;

    // idx holds one 2-bit index per LOGICAL output element (total
    // LogicalCompressedSize entries), packed across NumIdxWords int32_t
    // words. Initialized to the pattern 0b10 for every field -- see below.
    SparseIdxPack<NumIdxWords> idx{};
    static_for<0, LogicalCompressedSize, 1>{}([&](auto k) {
        constexpr uint32_t bit_pos = static_cast<uint32_t>(k) * 2u;
        constexpr uint32_t word    = bit_pos / 32u;
        constexpr uint32_t shift   = bit_pos % 32u;
        idx.words[word] |= static_cast<int32_t>(2u << shift);
    });

    // Shared idx-field writer (used by both the scalar and packed paths).
    auto set_idx_field = [&](uint32_t field_idx, uint32_t j_value) {
        const uint32_t bit_pos = field_idx * 2u;
        const uint32_t word    = bit_pos / 32u;
        const uint32_t shift   = bit_pos % 32u;
        idx.words[word] &= ~static_cast<int32_t>(0b11u << shift);
        idx.words[word] |= static_cast<int32_t>(j_value << shift);
    };

    if constexpr(PackedSize == 1)
    {
        // Original scalar path, byte-granular groups of 4 -- unchanged.
        static_for<0, CompressedSize / 2, 1>{}([&](auto i) {
            // Fix (bug 1): a group with fewer than
            // 2 real non-zeros leaves one (or both) output slots unassigned
            // by the scan below; its idx stays at the default 2 (see
            // comment above). For that default idx to be safe REGARDLESS of
            // what value the other slot claims for position 2, an
            // unassigned slot's VALUE must be a true zero -- not copied
            // from some fixed input position, because that position can
            // itself be the group's real (only) survivor. The original
            // code read defaults from {a[2], a[3]}: this is only safe when
            // position 3 is *guaranteed* zero (true for the canonical
            // "keep slots 0,2" synthetic pattern) -- for genuinely
            // arbitrary-position 2:4 data (e.g. groups whose sole survivor
            // sits at position 3, or, subtly, at position 2 -- both occur
            // in real Quark-quantized weights, which are only 97.7%
            // exactly-2:4), that default silently duplicates a real value
            // onto a mismatched idx (measured: {0,0,0,44} reconstructed as
            // {0,0,44,44}, or in the slot0-collides-with-slot1 direction, a
            // legitimate lone survivor at position 2 got double-counted
            // against B, producing small K-independent errors even on the
            // "canonical" synthetic pattern whenever its random fill
            // happened to leave slot 0 zero and slot 2 as the true lone
            // survivor). An always-zero default is correct in every case:
            // 0/1/2-nonzero groups all still reconstruct exactly, and any
            // idx collision on an unassigned slot contributes nothing.
            ADataType nonzero_elems[2] = {static_cast<ADataType>(0), static_cast<ADataType>(0)};
            int32_t non_zero_pos       = 0;

            static_for<0, 4, 1>{}([&](auto j) {
                if(static_cast<float>(a_vec[i * 4 + j]) != 0.0f)
                {
                    nonzero_elems[non_zero_pos] = a_vec[i * 4 + j];
                    set_idx_field(static_cast<uint32_t>(i) * 2u + static_cast<uint32_t>(non_zero_pos),
                                  static_cast<uint32_t>(j));
                    ++non_zero_pos;
                }
            });
            a_vec[i * 2]     = nonzero_elems[0];
            a_vec[i * 2 + 1] = nonzero_elems[1];
        });
    }
    else
    {
        static_assert(PackedSize == 2 && std::is_same_v<LogicalADataType, pk_int4_t>,
                      "compress_a_impl packed path only implements pk_int4_t (2x4-bit) currently");
        // ARCH SCOPE: the packed path below (and in particular the SWAP +
        // XOR-1 idx mapping) is EMPIRICALLY MEASURED ON gfx1201 (RDNA4
        // v_swmmac_*_iu4). No gfx9/CDNA pk_int4 SPARSE op exists in the tree
        // today, so this path is gfx12-only in practice -- but the transforms
        // selector is not architecture-gated, so if a CDNA pk_int4 sparse op
        // is ever added, the mapping MUST be re-measured on that hardware
        // before this path is trusted there. (The scalar PackedSize==1 branch
        // above is architecture-generic: its fix corrects a plain
        // out-of-spec default and changes behavior identically everywhere.)
        //
        // CONVENTION SCOPE: independently measured one-hot sweeps (bare
        // v_swmmac_*_iu4_w32 builtins, no CK code in the binary) show the
        // gfx1201 hardware idx law is a plain IDENTITY in raw nibble
        // coordinates: idx field i pairs compressed nibble i with the B
        // nibble at raw offset equal to the field value. The SWAP and XOR-1
        // below are therefore not a hardware quirk -- they are the
        // coordinate change from CK's packed convention
        // (CK_TILE_USE_PK4_LAYOUT_SHUFFLE: logical element 0 = HIGH nibble)
        // into raw nibble order. config.hpp currently defines that macro
        // unconditionally and pk_int4.hpp tests it with #ifdef, so its
        // #else branches are unreachable dead code; if the convention ever
        // becomes switchable, these constants must change WITH it -- a
        // source-convention dependency, independent of architecture.
        // Packed-nibble path (pk_int4_t): each iteration consumes 2
        // PHYSICAL input bytes (= 4 LOGICAL nibbles, one real 2:4 group) and
        // produces 1 PHYSICAL output byte (the <=2 survivor nibbles,
        // packed). CompressedSize physical output bytes -> CompressedSize
        // iterations (unlike the scalar path's CompressedSize/2, because
        // here one physical byte carries 2 logical elements).
        //
        // Nibble ordering matches pk_int4_t's own CK_TILE_USE_PK4_LAYOUT_SHUFFLE
        // convention (on by default, see pk_int4.hpp): "logical element 0"
        // of a packed byte is the HIGH nibble, "element 1" is the LOW
        // nibble -- so within a group of 4 logical positions (j=0..3) built
        // from 2 consecutive physical bytes (byte0=a_vec[2i], byte1=a_vec[2i+1]):
        // j=0 -> byte0 high, j=1 -> byte0 low, j=2 -> byte1 high, j=3 -> byte1 low.
        // Raw byte-pointer view of a_vec: clang's native ext_vector_type
        // operator[] does not support a WRITE with a single (non-scaled)
        // compile-time-constant index in this context (only the scalar
        // path's `a_vec[i*2]=...` pattern compiles) -- sidestep entirely by
        // reinterpreting a_vec as a flat uint8_t* for both the packed reads
        // and the packed write below (same technique already validated in
        // a standalone probe's fill code).
        uint8_t * a_bytes = reinterpret_cast<uint8_t *>(&a_vec);
        static_for<0, CompressedSize, 1>{}([&](auto i) {
            const uint8_t byte0 = a_bytes[2 * static_cast<uint32_t>(i) + 0];
            const uint8_t byte1 = a_bytes[2 * static_cast<uint32_t>(i) + 1];
            auto get_nibble     = [](uint8_t byte, bool high) -> int8_t {
                uint8_t nib = high ? ((byte >> 4) & 0x0Fu) : (byte & 0x0Fu);
                int8_t val  = static_cast<int8_t>(nib);
                if(val & 0x08) val = static_cast<int8_t>(val | 0xF0); // sign-extend 4-bit
                return val;
            };
            const int8_t nib[4] = {
                get_nibble(byte0, true), get_nibble(byte0, false),
                get_nibble(byte1, true), get_nibble(byte1, false)};

            // Same true-zero-default fix as the scalar path (bug 1),
            // applied at nibble granularity.
            //
            // Empirical metadata mapping (bug 3): the swmmac IU4 hardware reads the two
            // idx fields of a group SWAPPED relative to which compressed
            // value they govern, AND applies a "XOR 1" transform to the
            // field's raw 2-bit value to get the real reconstructed
            // position. Empirically determined (device measurement, not
            // derived from the ISA doc alone -- the general Sec 7.12.3
            // pseudocode does not capture this IU4-specific detail) via a
            // well-formed (always-exactly-2-survivor, no default-idx
            // ambiguity) position-pair sweep on real hardware: for a group
            // whose HIGH-nibble survivor (my slot 0, found first in scan
            // order) sat at true position j0 and LOW-nibble survivor (slot
            // 1, found second) sat at true position j1, the hardware
            // reconstructs correctly ONLY when field(i*2+1) [normally
            // "idx1"] is written as (j0 XOR 1) and field(i*2+0) [normally
            // "idx0"] is written as (j1 XOR 1) -- i.e. slot0's position goes
            // into the OTHER field, XORed with 1, and vice versa. Verified
            // against all 6 position-pair combinations (C(4,2)) with exact
            // arithmetic match (predicted C01 == measured C01 in every
            // case) before being applied here. This ALSO fully explains an
            // earlier single-survivor sweep's "always reconstructs at
            // position 3" finding: the untouched default field (value 2,
            // unused-slot placeholder) XORed with 1 gives 3, constant,
            // regardless of the real survivor's true position -- exactly
            // what was observed.
            int8_t survivor[2]  = {0, 0};
            int32_t non_zero_pos = 0;
            static_for<0, 4, 1>{}([&](auto j) {
                if(nib[static_cast<uint32_t>(j)] != 0)
                {
                    survivor[non_zero_pos] = nib[static_cast<uint32_t>(j)];
                    // slot 0 (HIGH nibble) -> field i*2+1; slot 1 (LOW nibble) -> field i*2+0.
                    const uint32_t target_field =
                        static_cast<uint32_t>(i) * 2u + (1u - static_cast<uint32_t>(non_zero_pos));
                    set_idx_field(target_field, static_cast<uint32_t>(j) ^ 1u);
                    ++non_zero_pos;
                }
            });
            // Re-pack: element0 (survivor[0]) -> HIGH nibble, element1
            // (survivor[1]) -> LOW nibble, matching the same shuffle
            // convention used to unpack above (and used by the register-map
            // fill on the host/caller side).
            const uint8_t out_byte = static_cast<uint8_t>(
                ((static_cast<uint8_t>(survivor[0]) & 0x0Fu) << 4) |
                (static_cast<uint8_t>(survivor[1]) & 0x0Fu));
            // Output bytes are always written at index <= input bytes read
            // (i <= 2i), so writing through the same raw-byte view in
            // increasing i order never clobbers not-yet-read input data.
            a_bytes[static_cast<uint32_t>(i)] = out_byte;
        });
    }

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
 * @brief Performs 2:4 structured sparsity compression on a static_distributed_tensor representing A
 *        and produces an index mask.
 * @note  Returns a tuple of two. The first element is an ext_vector containing all the compressed
 *        elements. The second element is the index mask.
 */
template <index_t CompressionRatio>
struct SparseCompressTransform
{
    /**
     * This function takes A in uncompressed form as a big ext_vector, and returns an owned
     * compressed ext_vector.
     *
     * LogicalADataType defaults to the vector's scalar type, so every existing call site --
     * including direct execExtVec() callers that know nothing about packing -- behaves exactly
     * as before. exec() below (the real Pipeline::exec() path) passes the TRUE logical element
     * type explicitly; that is the only way compress_a_impl can observe PackedSize > 1, which
     * sub-byte types such as pk_int4_t require so that the index mask carries one entry per
     * LOGICAL element rather than one per storage byte.
     */
    template <typename VecType,
              typename LogicalADataType =
                  typename vector_traits<remove_cvref_t<VecType>>::scalar_type>
    CK_TILE_DEVICE static auto execExtVec(VecType input)
    {
        using VecTraits                         = vector_traits<remove_cvref_t<VecType>>;
        using ScalarT                           = typename VecTraits::scalar_type;
        static constexpr auto VecN              = VecTraits::vector_size;
        static constexpr index_t CompressedSize = VecN / CompressionRatio;
        using VecCompressed                     = ext_vector_t<ScalarT, CompressedSize>;
        using IdxType =
            sparse::detail::SparseIdxPack<sparse::detail::idx_words_needed<
                CompressedSize * numeric_traits<LogicalADataType>::PackedSize>>;

        static_assert(VecN % CompressionRatio == 0, "VecN must be divisible by CompressionRatio");
        static_assert(CompressedSize > 0, "CompressedSize must be > 0");
        static_assert(!std::is_reference_v<VecCompressed>,
                      "Sparse compression must own its transformed vector");

        auto idx        = sparse::detail::
            compress_a_impl<ScalarT, CompressedSize, VecType, LogicalADataType>(input);
        auto compressed = *ck_tile::bit_cast<VecCompressed*>(&input);

        return std::tuple<VecCompressed, IdxType>{compressed, idx};
    }

    /**
     * This function takes A in uncompressed form as a static_distributed tensor and returns an
     * owned compressed ext_vector. Returning an owned value keeps const inputs valid after the
     * transform call completes.
     */
    template <typename ATensor>
    CK_TILE_DEVICE static auto exec(const ATensor& a_tensor)
    {
        // Properties of ATensor as a big ext vector.
        using ADataType        = typename ATensor::DataType; // TRUE logical type (e.g. pk_int4_t)
        constexpr index_t VecN = ATensor::get_thread_buffer_size();
        using VecType          = ext_vector_t<ADataType, VecN>;

        return execExtVec<VecType, ADataType>(
            a_tensor.get_thread_buffer().template get_as<VecType>().template at<0>());
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
