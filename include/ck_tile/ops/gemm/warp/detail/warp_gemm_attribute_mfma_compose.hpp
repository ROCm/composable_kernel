// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
// smfmac (structured sparsity)
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
namespace ck_tile {
namespace detail {
namespace wg_attr_compose {

// Basic config/state that policies transform at compile-time
struct NoneTag;

enum class SwizzleKind
{
    None,
    A,
    B
};

template <typename Impl_,
          bool TransposeC_,
          SwizzleKind Swizzle_,
          index_t SFactor_,
          index_t KIter_,
          WGAttrNumAccessEnum NumAccess_>
struct State
{
    using Impl                                     = remove_cvref_t<Impl_>;
    static constexpr bool TransposeC               = TransposeC_;
    static constexpr SwizzleKind Swizzle           = Swizzle_;
    static constexpr index_t SFactor               = SFactor_;
    static constexpr index_t KIter                 = KIter_;
    static constexpr WGAttrNumAccessEnum NumAccess = NumAccess_;
    static constexpr index_t NumAccValue           = static_cast<index_t>(NumAccess);
};

// Default base state
template <typename Impl, WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
using BaseState = State<Impl, false, SwizzleKind::None, 1, 1, NumAccess>;

// Helpers to compute derived constants and types from State

// A/B data/vec types under transpose flag
template <class S>
struct ABTypes
{
    using ADataType =
        std::conditional_t<S::TransposeC, typename S::Impl::BDataType, typename S::Impl::ADataType>;
    using BDataType =
        std::conditional_t<S::TransposeC, typename S::Impl::ADataType, typename S::Impl::BDataType>;

    using AVecBase =
        std::conditional_t<S::TransposeC, typename S::Impl::BVecType, typename S::Impl::AVecType>;
    using BVecBase =
        std::conditional_t<S::TransposeC, typename S::Impl::AVecType, typename S::Impl::BVecType>;

    using AVecType = ext_vector_t<ADataType, vector_traits<AVecBase>::vector_size * S::KIter>;
    using BVecType = ext_vector_t<BDataType, vector_traits<BVecBase>::vector_size * S::KIter>;
};

// C types/shape (C does not change type with policies)
template <class S>
struct CTypes
{
    using CDataType = typename S::Impl::CDataType;
    using CVecType  = typename S::Impl::CVecType;
};

// kM/kN/kK/kKPerThread under transpose and iterateK

template <class S>
struct KShape
{
    static constexpr index_t kM          = S::TransposeC ? S::Impl::kN : S::Impl::kM;
    static constexpr index_t kN          = S::TransposeC ? S::Impl::kM : S::Impl::kN;
    static constexpr index_t kK          = S::Impl::kK * S::KIter;
    static constexpr index_t kKPerThread = S::Impl::kABKPerLane * S::KIter;
    static constexpr index_t kCMLane     = S::Impl::kCMLane; // unchanged
};

// Lane helpers

template <class S>
struct Lanes
{
    static constexpr index_t AMLane = S::TransposeC ? S::Impl::kBNLane : S::Impl::kAMLane;
    static constexpr index_t BNLane = S::TransposeC ? S::Impl::kAMLane : S::Impl::kBNLane;
};

// Encoding builders centralizing existing logic, parameterized by S

// A encodings with NumAccess and IterateK folded, with multi-block and swizzle
template <class S>
CK_TILE_DEVICE static constexpr auto MakeAWarpDstrEncoding()
{
    constexpr index_t NumAccValue = S::NumAccValue;

    // Helper lambdas for the base A-encoding and the swizzled variant
    auto base_enc = []() {
        if constexpr(S::Impl::kAMBlock == 1 && S::Impl::kBNBlock == 1)
        {
            if constexpr(NumAccValue == 1)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Lanes<S>::AMLane>,
                          sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2>,
                    sequence<1>>{};
            }
            else
            {
                static_assert(KShape<S>::kKPerThread % NumAccValue == 0,
                              "kKPerThread must be divisible by NumAccess");
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Lanes<S>::AMLane>,
                          sequence<NumAccValue,
                                   S::Impl::kABKLane,
                                   S::Impl::kABKPerLane * S::KIter / NumAccValue>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<1, 0>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{};
            }
        }
        else if constexpr(S::Impl::kAMBlock == 1 && 1 < S::Impl::kBNBlock)
        {
            static_assert(NumAccValue == 1,
                          "Multiple access is not supported when using multi-block");
            return tile_distribution_encoding<
                sequence<S::Impl::kBNBlock>,
                tuple<sequence<Lanes<S>::AMLane>,
                      sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                tuple<sequence<0, 2, 1>>,
                tuple<sequence<0, 0, 0>>,
                sequence<2>,
                sequence<1>>{};
        }
        else // 1 < AMBlock && BNBlock == 1
        {
            static_assert(NumAccValue == 1,
                          "Multiple access is not supported when using multi-block");
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Impl::kAMBlock, Lanes<S>::AMLane>,
                      sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                tuple<sequence<1, 2, 1>>,
                tuple<sequence<0, 0, 1>>,
                sequence<2>,
                sequence<1>>{};
        }
    };

    auto swizzled_enc = []() {
        return tile_distribution_encoding<
            sequence<>,
            tuple<
                sequence<S::Impl::kAMLane / (S::Impl::kCMLane * S::SFactor * S::Impl::kCM1PerLane),
                         S::Impl::kCMLane,
                         S::SFactor,
                         S::Impl::kCM1PerLane>,
                sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
            tuple<sequence<2, 1, 1, 1, 1>>,
            tuple<sequence<0, 0, 2, 1, 3>>,
            sequence<2>,
            sequence<1>>{};
    };

    if constexpr(S::Swizzle == SwizzleKind::A)
        return swizzled_enc();
    else
        return base_enc();
}

// B encodings with NumAccess and IterateK folded, with multi-block and swizzle
template <class S>
CK_TILE_DEVICE static constexpr auto MakeBWarpDstrEncoding()
{
    constexpr index_t NumAccValue = S::NumAccValue;

    auto base_enc = []() {
        if constexpr(S::Impl::kAMBlock == 1 && S::Impl::kBNBlock == 1)
        {
            if constexpr(NumAccValue == 1)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Lanes<S>::BNLane>,
                          sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2>,
                    sequence<1>>{};
            }
            else
            {
                static_assert(KShape<S>::kKPerThread % NumAccValue == 0,
                              "kKPerThread must be divisible by NumAccess");
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Lanes<S>::BNLane>,
                          sequence<NumAccValue,
                                   S::Impl::kABKLane,
                                   S::Impl::kABKPerLane * S::KIter / NumAccValue>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<1, 0>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{};
            }
        }
        else if constexpr(S::Impl::kAMBlock == 1 && 1 < S::Impl::kBNBlock)
        {
            static_assert(NumAccValue == 1,
                          "Multiple access is not supported when using multi-block");
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Impl::kBNBlock, Lanes<S>::BNLane>,
                      sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                tuple<sequence<1, 2, 1>>,
                tuple<sequence<0, 0, 1>>,
                sequence<2>,
                sequence<1>>{};
        }
        else // 1 < AMBlock && BNBlock == 1
        {
            static_assert(NumAccValue == 1,
                          "Multiple access is not supported when using multi-block");
            return tile_distribution_encoding<
                sequence<S::Impl::kAMBlock>,
                tuple<sequence<Lanes<S>::BNLane>,
                      sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
                tuple<sequence<0, 2, 1>>,
                tuple<sequence<0, 0, 0>>,
                sequence<2>,
                sequence<1>>{};
        }
    };

    auto swizzled_enc = []() {
        return tile_distribution_encoding<
            sequence<>,
            tuple<
                sequence<S::Impl::kAMLane / (S::Impl::kCMLane * S::SFactor * S::Impl::kCM1PerLane),
                         S::Impl::kCMLane,
                         S::SFactor,
                         S::Impl::kCM1PerLane>,
                sequence<S::Impl::kABKLane, S::Impl::kABKPerLane * S::KIter>>,
            tuple<sequence<2, 1, 1, 1, 1>>,
            tuple<sequence<0, 0, 2, 1, 3>>,
            sequence<2>,
            sequence<1>>{};
    };

    if constexpr(S::Swizzle == SwizzleKind::B)
        return swizzled_enc();
    else
        return base_enc();
}

// C distribution encoding with transpose and multi-block

template <class S>
CK_TILE_DEVICE static constexpr auto MakeCWarpDstrEncoding()
{
    constexpr bool HasSwizzle = (S::Swizzle == SwizzleKind::A) || (S::Swizzle == SwizzleKind::B);

    auto make_m_splits = []() {
        if constexpr(HasSwizzle)
        {
            // Swizzled M splits
            return sequence<S::Impl::kCM0PerLane / S::SFactor,
                            S::Impl::kCMLane,
                            S::Impl::kCM1PerLane * S::SFactor>{};
        }
        else
        {
            return sequence<S::Impl::kCM0PerLane, S::Impl::kCMLane, S::Impl::kCM1PerLane>{};
        }
    };

    if constexpr(S::Impl::kAMBlock == 1 && S::Impl::kBNBlock == 1)
    {
        if constexpr(!S::TransposeC)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<decltype(make_m_splits()), sequence<S::Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else // TransposeC
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Impl::kCNLane>, decltype(make_m_splits())>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
        }
    }
    else if constexpr(S::Impl::kAMBlock == 1 && 1 < S::Impl::kBNBlock)
    {
        if constexpr(!S::TransposeC)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<decltype(make_m_splits()), sequence<S::Impl::kBNBlock * S::Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Impl::kBNBlock * S::Impl::kCNLane>, decltype(make_m_splits())>,
                tuple<sequence<2, 1>>,
                tuple<sequence<1, 0>>,
                sequence<2, 2>,
                sequence<0, 2>>{};
        }
    }
    else if constexpr(1 < S::Impl::kAMBlock && S::Impl::kBNBlock == 1)
    {
        if constexpr(!S::TransposeC)
        {
            return tile_distribution_encoding<sequence<>,
                                              tuple<sequence<S::Impl::kCM0PerLane,
                                                             S::Impl::kAMBlock * S::Impl::kCMLane,
                                                             S::Impl::kCM1PerLane>,
                                                    sequence<S::Impl::kCNLane>>,
                                              tuple<sequence<1, 2>>,
                                              tuple<sequence<1, 0>>,
                                              sequence<1, 1>,
                                              sequence<0, 2>>{};
        }
        else
        {
            return tile_distribution_encoding<sequence<>,
                                              tuple<sequence<S::Impl::kCNLane>,
                                                    sequence<S::Impl::kCM0PerLane,
                                                             S::Impl::kAMBlock * S::Impl::kCMLane,
                                                             S::Impl::kCM1PerLane>>,
                                              tuple<sequence<2, 1>>,
                                              tuple<sequence<1, 0>>,
                                              sequence<2, 2>,
                                              sequence<0, 2>>{};
        }
    }
}

// Detect smfmac by Impl type: provide a small trait that is true for smfmac attribute impls
template <class T>
struct is_smfmac_impl : std::false_type
{
};
template <typename AType_,
          typename BType_,
          typename AccType_,
          index_t MPerWave_,
          index_t NPerWave_,
          index_t KPerWave_,
          WGAttrCtlEnum C>
struct is_smfmac_impl<
    WarpGemmAttributeSmfmacImpl<AType_, BType_, AccType_, MPerWave_, NPerWave_, KPerWave_, C>>
    : std::true_type
{
};

// Final composed attribute
// Primary (MFMA) and SMFMA-specialized ComposedAttribute
template <class S, bool IsSmfmac = is_smfmac_impl<typename S::Impl>::value>
struct ComposedAttribute
{
    using Impl    = typename S::Impl;
    using ATypes  = ABTypes<S>;
    using CTypesT = CTypes<S>;

    static constexpr index_t kM          = KShape<S>::kM;
    static constexpr index_t kN          = KShape<S>::kN;
    static constexpr index_t kK          = KShape<S>::kK;
    static constexpr index_t kKPerThread = KShape<S>::kKPerThread;
    static constexpr index_t kCMLane     = KShape<S>::kCMLane;

    using ADataType = typename ATypes::ADataType;
    using BDataType = typename ATypes::BDataType;
    using CDataType = typename CTypesT::CDataType;

    using AVecType = typename ATypes::AVecType;
    using BVecType = typename ATypes::BVecType;
    using CVecType = typename CTypesT::CVecType;

    using AWarpDstrEncoding = decltype(MakeAWarpDstrEncoding<S>());
    using BWarpDstrEncoding = decltype(MakeBWarpDstrEncoding<S>());
    using CWarpDstrEncoding = decltype(MakeCWarpDstrEncoding<S>());

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return S::KIter; }

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(S::KIter == 1)
        {
            if constexpr(S::TransposeC)
            {
                // When TransposeC, composed A/B are swapped relative to Impl.
                // Pass b_vec as Impl::AVecType and a_vec as Impl::BVecType.
                Impl{}(c_vec,
                       reinterpret_cast<const typename Impl::AVecType&>(b_vec),
                       reinterpret_cast<const typename Impl::BVecType&>(a_vec),
                       bool_constant<post_nop_>{});
            }
            else
            {
                Impl{}(c_vec,
                       reinterpret_cast<const typename Impl::AVecType&>(a_vec),
                       reinterpret_cast<const typename Impl::BVecType&>(b_vec),
                       bool_constant<post_nop_>{});
            }
        }
        else
        {
            using buf_a = thread_buffer<typename Impl::AVecType, S::KIter>;
            using buf_b = thread_buffer<typename Impl::BVecType, S::KIter>;

            static_for<0, S::KIter, 1>{}([&](auto iKIter) {
                if constexpr(S::TransposeC)
                {
                    // Swap mapping: b_vec -> Impl::AVecType, a_vec -> Impl::BVecType
                    Impl{}(c_vec,
                           reinterpret_cast<const buf_b&>(b_vec)
                               .template get_as<typename Impl::AVecType>()[iKIter],
                           reinterpret_cast<const buf_a&>(a_vec)
                               .template get_as<typename Impl::BVecType>()[iKIter],
                           bool_constant<post_nop_>{});
                }
                else
                {
                    Impl{}(c_vec,
                           reinterpret_cast<const buf_a&>(a_vec)
                               .template get_as<typename Impl::AVecType>()[iKIter],
                           reinterpret_cast<const buf_b&>(b_vec)
                               .template get_as<typename Impl::BVecType>()[iKIter],
                           bool_constant<post_nop_>{});
                }
            });
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(S::KIter == 1)
        {
            if constexpr(S::TransposeC)
            {
                // Swap mapping: b_vec -> Impl::AVecType, a_vec -> Impl::BVecType
                return Impl{}(reinterpret_cast<const typename Impl::AVecType&>(b_vec),
                              reinterpret_cast<const typename Impl::BVecType&>(a_vec));
            }
            else
            {
                return Impl{}(reinterpret_cast<const typename Impl::AVecType&>(a_vec),
                              reinterpret_cast<const typename Impl::BVecType&>(b_vec));
            }
        }
        else
        {
            using buf_a       = thread_buffer<typename Impl::AVecType, S::KIter>;
            using buf_b       = thread_buffer<typename Impl::BVecType, S::KIter>;
            constexpr auto I0 = number<0>{};

            CVecType c_vec;
            if constexpr(S::TransposeC)
            {
                // Swap mapping: b_vec -> Impl::AVecType, a_vec -> Impl::BVecType
                c_vec = Impl{}(reinterpret_cast<const buf_b&>(b_vec)
                                   .template get_as<typename Impl::AVecType>()[I0],
                               reinterpret_cast<const buf_a&>(a_vec)
                                   .template get_as<typename Impl::BVecType>()[I0]);

                static_for<1, S::KIter, 1>{}([&](auto iKIter) {
                    Impl{}(c_vec,
                           reinterpret_cast<const buf_b&>(b_vec)
                               .template get_as<typename Impl::AVecType>()[iKIter],
                           reinterpret_cast<const buf_a&>(a_vec)
                               .template get_as<typename Impl::BVecType>()[iKIter]);
                });
            }
            else
            {
                c_vec = Impl{}(reinterpret_cast<const buf_a&>(a_vec)
                                   .template get_as<typename Impl::AVecType>()[I0],
                               reinterpret_cast<const buf_b&>(b_vec)
                                   .template get_as<typename Impl::BVecType>()[I0]);

                static_for<1, S::KIter, 1>{}([&](auto iKIter) {
                    Impl{}(c_vec,
                           reinterpret_cast<const buf_a&>(a_vec)
                               .template get_as<typename Impl::AVecType>()[iKIter],
                           reinterpret_cast<const buf_b&>(b_vec)
                               .template get_as<typename Impl::BVecType>()[iKIter]);
                });
            }
            return c_vec;
        }
    }
};

// SMFMA specialization: forbid Transpose and Swizzle for now; KIter must be 1
// TODO: enable swizzle, transpose for smfmac
template <class S>
struct ComposedAttribute<S, true>
{
    using Impl = typename S::Impl;

    static_assert(!S::TransposeC, "smfmac TransposeC is not supported in composed attributes");
    static_assert(S::Swizzle == SwizzleKind::None,
                  "smfmac Swizzle is not supported in composed attributes");
    static_assert(S::KIter == 1, "smfmac IterateK is not supported (KIter must be 1)");

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM                = Impl::kM;
    static constexpr index_t kN                = Impl::kN;
    static constexpr index_t kK                = Impl::kK;
    static constexpr index_t kKPerThread       = Impl::kABKPerLane;
    static constexpr index_t kCMLane           = Impl::kCMLane;
    static constexpr index_t kCompressionRatio = Impl::CompressionRatio;

    // Reuse the encodings defined by the smfmac attribute wrapper for consistency
    using AWarpDstrEncoding = typename WarpGemmAttributeSmfmac<Impl>::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttributeSmfmac<Impl>::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttributeSmfmac<Impl>::CWarpDstrEncoding;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
        Impl{}(c_vec, a_vec, b_vec, idx, bool_constant<post_nop_>{});
    }

    // c_vec = a_vec * b_vec[idx]
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const BVecType& b_vec,
                                       const int32_t& idx) const
    {
        CVecType c_vec{0};
        Impl{}(c_vec, a_vec, b_vec, idx);
        return c_vec;
    }
};

// Policy wrappers produce a new State from an old one

template <index_t KIter>
struct PolicyIterateK
{
    template <class S>
    using apply =
        State<typename S::Impl, S::TransposeC, S::Swizzle, S::SFactor, KIter, S::NumAccess>;
};

struct PolicyTransposeC
{
    template <class S>
    using apply = State<typename S::Impl, true, S::Swizzle, S::SFactor, S::KIter, S::NumAccess>;
};

template <index_t SFactor>
struct PolicySwizzleA
{
    template <class S>
    using apply =
        State<typename S::Impl, S::TransposeC, SwizzleKind::A, SFactor, S::KIter, S::NumAccess>;
};

template <index_t SFactor>
struct PolicySwizzleB
{
    template <class S>
    using apply =
        State<typename S::Impl, S::TransposeC, SwizzleKind::B, SFactor, S::KIter, S::NumAccess>;
};

} // namespace wg_attr_compose
} // namespace detail
} // namespace ck_tile

// High-level alias to construct a composed WarpGemm attribute via CoreDispatcher and policies
#include "ck_tile/ops/gemm/warp/warp_gemm_core_dispatcher.hpp"

namespace ck_tile {

// Helper that uses core dispatcher to select MFMA vs smfmac and composes policies only
template <bool TransposeC,
          bool SwizzleA,
          bool UseStructuredSparsity,
          typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          WGAttrNumAccessEnum NumAccess>
struct ComposePolicies
{
    using CD   = WarpGemmCoreDispatcher<AType,
                                        BType,
                                        AccType,
                                        MPerWave,
                                        NPerWave,
                                        KPerWave,
                                        UseStructuredSparsity>;
    using Impl = typename CD::Impl;

    static_assert(Impl::kK > 0, "Invalid K dimension");
    static_assert(Impl::kK <= KPerWave, "KPerWave must smaller and equal than Impl::kK");

    static constexpr index_t KIter = KPerWave / Impl::kK;

    static constexpr bool IsSmfmac = detail::wg_attr_compose::is_smfmac_impl<Impl>::value;

    // First, setup a default state as the base state.
    using S0 = detail::wg_attr_compose::BaseState<Impl, NumAccess>;

    // The order of policies matters so we compose them with the order: Kiter->TransposeC->SwizzleA.
    using S1 = std::conditional_t<
        (KIter == 1),
        S0,
        typename detail::wg_attr_compose::PolicyIterateK<KIter>::template apply<S0>>;
    using S2 =
        std::conditional_t<TransposeC,
                           typename detail::wg_attr_compose::PolicyTransposeC::template apply<S1>,
                           S1>;
    // Match dispatcher behavior: when TransposeC is enabled, the swizzle applies to B
    // (i.e., SwizzleBTransposedCDistribution). Otherwise apply swizzle to A.
    using S3 = std::conditional_t<
        SwizzleA && !TransposeC,
        typename detail::wg_attr_compose::PolicySwizzleA<2>::template apply<S2>,
        std::conditional_t<SwizzleA && TransposeC,
                           typename detail::wg_attr_compose::PolicySwizzleB<2>::template apply<S2>,
                           S2>>;

    // For SMFMA, use the dedicated WarpGemmSmfmacImpl wrapper with the SMFMA attribute.
    // For MFMA (default), use the regular WarpGemmImpl over the composed attribute.
    using type = std::conditional_t<
        IsSmfmac,
        ck_tile::WarpGemmSmfmacImpl<detail::wg_attr_compose::ComposedAttribute<S3>>,
        ck_tile::WarpGemmImpl<detail::wg_attr_compose::ComposedAttribute<S3>>>;
};

// Wrapper struct to match usage as MakeWarpGemm<...>::Type
template <bool TransposeC,
          bool SwizzleA,
          typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool UseStructuredSparsity    = false,
          WGAttrNumAccessEnum NumAccess = WGAttrNumAccessEnum::Single>
struct MakeWarpGemm
{
    using Type = typename ComposePolicies<TransposeC,
                                          SwizzleA,
                                          UseStructuredSparsity,
                                          AType,
                                          BType,
                                          AccType,
                                          MPerWave,
                                          NPerWave,
                                          KPerWave,
                                          NumAccess>::type;
};

} // namespace ck_tile
