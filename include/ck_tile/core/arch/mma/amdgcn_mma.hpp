// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/ignore.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck_tile::core::arch::mma {

/**---------------------------------------------------
 *  Meaning of amdgcn_mma layout parameters (general)
 * ---------------------------------------------------
 *
 * The fragment (MmaTile) sizes and layout constants in the amdgcn_mma struct describe the mapping
 * between intrinsic input / output matrix elements and vector registers (lane x vector_item space).
 * Note that we end up having a mapping for A, B and C separately, although those for A and B are
 * usually similar if not identical. All mappings can be described as an unmerge operation on one of
 * the matrix dims (either K for AB or M for C), followed by remerging of the resulting subdims and
 * raw other dim into the Lane and Vector_item dimensions. When considering an unmerge operation on
 * a dimension K, we can label the resulting sub-dimensions as K0, K1, and K2, where K0 is the size
 * of the fastest changing dimension. K0 is also referred to as "The size of the first unmerge", and
 * K1 would be "The size of the second unmerge". There are never more than 2 unmerge operations, and
 * unmerge operations may be trivial (unmerge size of 1). Example double unmerge of size {3, 2} of a
 * K dimension of size 12:
 *
 * K  K2 K1 K0
 * 0  0  0  0
 * 1  0  0  1
 * 2  0  1  0
 * 3  0  1  1
 * 4  0  2  0
 * 5  0  2  1
 * 6  1  0  0
 * 7  1  0  1
 * 8  1  1  0
 * 9  1  1  1
 * 10 1  2  0
 * 11 1  2  1
 *
 * Note that K0 = 2 (first unmerge size, fastest changing), K1 = 3 (second unmerge size,
 * second-fastest changing), and K2 = 12 / 2 / 3 = 2 (outermost dimension, whatever is left).
 *
 * If we were to use this unmerge op to describe an A matrix layout in registers, we might have for
 * example that L (lane dim) is composed of K1 and M, and V (vector_item dim) is composed of K2 and
 * K0. Compactly described, this would be K{3, 2} L{K1M} V{K2K0}, and if the M dimension was 2 we
 * would have the following layout (6 lanes, 4 vector items each):
 *
 *    | V0       | V1       | V2       | V3       |
 * L0 | M=0 K=0  | M=0 K=1  | M=0 K=6  | M=0 K=7  |
 * L1 | M=1 K=0  | M=1 K=1  | M=1 K=6  | M=1 K=7  |
 * L2 | M=0 K=2  | M=0 K=3  | M=0 K=8  | M=0 K=9  |
 * L3 | M=1 K=2  | M=1 K=3  | M=1 K=8  | M=1 K=9  |
 * L4 | M=0 K=4  | M=0 K=5  | M=0 K=10 | M=0 K=11 |
 * L5 | M=1 K=4  | M=1 K=5  | M=1 K=10 | M=1 K=11 |
 *
 * Note that all A matrix elements are now placed in a unique (lane, vector_item). In case a Repeat
 * dimension is used, every single matrix element is mapped to multiple (Lane, vector_item)
 * locations, usually along the Lane dimension.
 *
 * Check out TileDistrEncRegMap which can print full forward and backward mapping tables for any
 * register mapping (expressed as a tile distribution encoding).
 *
 * ------------------------------------------
 *  Individual amdgcn_mma layout parameters
 * ------------------------------------------
 *
 * -- ABKPerLane --
 * The number of K dim elements in each lane. Always the same for A and B, even when they have
 * different layouts. In terms of unmerge sizes, it's equal to K0 * K2, i.e the product of the sizes
 * of the outermost and innermost dimensions after a double K unmerge.
 *
 * -- A / B NumAccess --
 * These two variables describe the size of the outermost dimension if two unmerge operations are
 * required for K (so K2). Alternatively it can be described as the number of sets the vector
 * dimension, which houses a number of K indices, is split up into. We may be able to actually
 * remove the A / B NumAccess from the amdgcn struct, but it sort of depends on how load and store
 * tile work and whether we want the mid-level code to always have to know about this. There are
 * only two reasons for the A / B NumAccess to ever not be 1, and they are different types of
 * reasons:
 *
 * (logical correctness). Applies to scale MFMA fp8, which due to the index matrix layout does not
 * allow arbitrary K perms to simplify layouts. This means the layout can only properly be described
 * with a Num Access value which is a multiple of 2.
 *
 * (load / store manipulation). It seems like the load and store tile functions end up looking for
 * the size of the smallest unmerged K dimension (K0) to determine how many elements should be
 * loaded at a time. Different Num Access values will lead to different load / store behavior, even
 * if logically equivalent.
 *
 * -- A / B Repeat --
 * Variable indicating that all matrix values are represented multiple times in the vector
 * registers, typically repeating in the lane dimension. This is always equal to the repeat value
 * used in Tile Distribution encodings. There are two reasons to have non-trivial (non-1) value
 * here: MFMA block-hiding to create oblong "virtual" intrinsics, and RDNA3 input repetition.
 *
 * -- CMPerLane --
 * The number of M dim elements in each lane. In terms of unmerge sizes, it's equal to M0 * M2, i.e
 * the product of the sizes of the outermost and innermost dimensions after a double M unmerge. This
 * does not count a potential increased M dimension size from block hiding. In this case, we have M
 * = kCMBlock * M2 * M1 * M0 instead.
 *
 * -- CNumAccess --
 * Same as A / B NumAccess but for the M dim (so M2), but the mid-level code doesn't care about this
 * and will not try to request a specific value. Absolutely needed for logical correctness of
 * register mappings since we can not perform arbitrary M permutations without messing up the A
 * layout. This does not count a potential increased M dimension size from block hiding. In this
 * case, we have M = kCMBlock * M2 * M1 * M0 instead.
 */

/**
 *  @class  amdgcn_mma_base
 *  @brief  Base class for amdgcn_mma structs to avoid a lot of code duplication. Also puts
 *          all generic parameter derivations and static asserts in one place. Houses all of the
 *          amdgcn struct types and variables, except for the exec() function.
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          uint32_t WaveSize_,
          index_t kABKPerLane_,
          index_t kAKNumAccess_,
          index_t kARepeat_,
          index_t kBKNumAccess_,
          index_t kBRepeat_,
          index_t kCMPerLane_,
          index_t kCMNumAccess_,
          typename OpType_,
          MmaOpFamily OpFamily_>
struct amdgcn_mma_base
{
    using OpType                          = OpType_;
    static constexpr MmaOpFamily OpFamily = OpFamily_;

    // Data types
    using ADataType = ADataType_;
    using BDataType = BDataType_;
    using CDataType = CDataType_;

    // Fragment (MmaTile) sizes, check description above.
    static constexpr index_t kM = FragM; // M = M2 * M1 * M0 (* kCMBlocks when block-hiding)
    static constexpr index_t kN = FragN;
    static constexpr index_t kK = FragK; // K = K2 * K1 * K0

    // Layout constants, check description above.
    static constexpr index_t kABKPerLane  = kABKPerLane_;  // K2 * K0
    static constexpr index_t kAKNumAccess = kAKNumAccess_; // K2
    static constexpr index_t kARepeat     = kARepeat_;     // RDNA3 repetition and MFMA block-hiding
    static constexpr index_t kBKNumAccess = kBKNumAccess_; // K2
    static constexpr index_t kBRepeat     = kBRepeat_;     // RDNA3 repetition and MFMA block-hiding
    static constexpr index_t kCMPerLane   = kCMPerLane_;   // M2 * M0
    static constexpr index_t kCMNumAccess = kCMNumAccess_; // M2

    // K-dimension compression ratio for A matrix, always 2 for sparse intrinsics.
    static constexpr index_t kCompressionRatio = (OpFamily == MmaOpFamily::SPARSE) ? 2 : 1;

    // Layout checks
    static_assert(kK % kABKPerLane == 0);
    static_assert(kABKPerLane % kAKNumAccess == 0);
    static_assert(kABKPerLane % kBKNumAccess == 0);
    static_assert(kCMPerLane % kCMNumAccess == 0);

    // Register types (derived)
    static constexpr index_t WaveSize = WaveSize_;
    static_assert((kM * kK * kARepeat) % (WaveSize * kCompressionRatio) == 0);
    static_assert((kN * kK * kBRepeat) % WaveSize == 0);
    static_assert((kM * kN) % WaveSize == 0);

    using AVecType = ext_vector_t<ADataType, kM * kK * kARepeat / WaveSize / kCompressionRatio>;
    using BVecType = ext_vector_t<BDataType, kN * kK * kBRepeat / WaveSize>;
    using CVecType = ext_vector_t<CDataType, kM * kN / WaveSize>;

    // Block-hiding / repeat related traits (derived)
    static_assert(kARepeat == kBRepeat || !std::is_same_v<OpType, WmmaOp>);
    static_assert(kARepeat == 1 || kBRepeat == 1 || !std::is_same_v<OpType, MfmaOp>);
    static constexpr index_t kCMBlocks = std::is_same_v<OpType, MfmaOp> ? kBRepeat : 1;
    static constexpr index_t kCNBlocks = std::is_same_v<OpType, MfmaOp> ? kARepeat : 1;
    static_assert(kM % (kCMBlocks * kCMPerLane) == 0);
    static_assert(kN % kCNBlocks == 0);

    // For the C matrix, the block dimension B is either put in the Vector dimension or the Lane
    // dimension. We can tell which by checking if we get the right Vector size.
    static constexpr bool CBlockDimInVecDim =
        kCMBlocks * kCNBlocks * kCMPerLane == vector_traits<CVecType>::vector_size;
};

/**
 * @struct Unsupported
 * @brief  Meta-tag to indicate unsupported amdgcn_mma instance.
 */
struct Unsupported;

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

#include <concepts>
/**
 * @concept HasExecSignature
 * @brief  Helper concept for exec signature check.
 */
template <typename MmaOp, typename... ExecArgs>
concept HasExecSignature = requires {
    {
        MmaOp::exec(typename MmaOp::AVecType{},
                    typename MmaOp::BVecType{},
                    typename MmaOp::CVecType{},
                    std::declval<ExecArgs>()...)
    } -> std::convertible_to<typename MmaOp::CVecType>;
};

/**
 * @concept MmaOpI
 * @brief  Expresses the meta-data interface required for each MmaOp policy.
 */
// TODO: Make sure this actually matches amdgcn_mma.
template <typename MmaOp>
concept MmaOpI = requires(MmaOp op) {
    // Requires an op context
    typename MmaOp::OpType;
    { MmaOp::OpFamily } -> std::convertible_to<MmaOpFamily>;

    // Captures types for inputs / outputs to mma function
    typename MmaOp::ADataType;
    typename MmaOp::BDataType;
    typename MmaOp::CDataType;
    typename MmaOp::AVecType;
    typename MmaOp::BVecType;
    typename MmaOp::CVecType;
    // Captures CK-specific layout properties
    { MmaOp::kABKPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kAKNumAccess } -> std::convertible_to<unsigned int>;
    { MmaOp::kARepeat } -> std::convertible_to<unsigned int>;
    { MmaOp::kBKNumAccess } -> std::convertible_to<unsigned int>;
    { MmaOp::kBRepeat } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMNumAccess } -> std::convertible_to<unsigned int>;
    { MmaOp::kCompressionRatio } -> std::convertible_to<unsigned int>;
} && (HasExecSignature<MmaOp> || HasExecSignature<MmaOp, int> || HasExecSignature<MmaOp, int, int>);

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 *  @class  amdgcn_mma
 *  @brief  This is the default MmaOp policy.
 *          Instances of this class are to be used as MmaOp policies.
 *          Light builtin wrapper for mfma / wmma instructions. This class's job is to
 *          provide a uniform interface to invoke the appropriate instruction
 *          based on the template parameters provided. This interface is to bridge
 *          the gap between the ck_tile API types and the native __builtin types.
 *  @tparam ADataType Datatype of input A
 *  @tparam BDataType Datatype of input B
 *  @tparam CDataType Datatype of accumulator
 *  @tparam FragM M-dimension of mma intrinsic (MmaTile)
 *  @tparam FragN N-dimension of mma intrinsic (MmaTile)
 *  @tparam FragK K-dimension of mma intrinsic (MmaTile)
 *  @tparam CtrlFlags Control flags for mma operation
 *  @tparam CompilerTarget The current compiler target
 *  @tparam OpFamily_ The type of operation (dense, sparse, scale, etc.)
 *  @tparam Enabler SFINAE enabler
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          typename CtrlFlags,
          typename CompilerTarget,
          MmaOpFamily OpFamily_,
          typename Enabler = void>
// clang-format off
//                                 | A B C DataTypes      |MNK + WaveSize |AParams |BPar |CPar |
struct amdgcn_mma : amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 1u, 1u, 1u, 1u, 1, 1, 1, 1, 1, 1, 1, Unsupported, MmaOpFamily::UNDEFINED>
// clang-format on
{
    // This is a default pass-through implementation that doesn't do anything practical.
    CK_TILE_DEVICE static auto
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        // Prints once across all thread blocks and threads.
        static __device__ int printed = 0;
        if(threadIdx.x == 0 && atomicCAS(&printed, 0, 1) == 0)
        {
            printf("[WARNING] Running amdgcn_mma dummy exec function!\n");
        }

        ignore(regsA, regsB);
        return regsC; // No-op, just return C
    }
};

} // namespace ck_tile::core::arch::mma
#pragma clang diagnostic pop

// Include the implementations
#include "wmma/wmma.hpp" // should be included before the below headers

#include "mfma/mfma.hpp"
#include "scale/scale.hpp"
#include "sparse/sparse.hpp"
