// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile {

template <typename Impl, WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct AWarpDstrEncodingTrait
{
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using ADataType                    = typename Impl::ADataType;
    static constexpr index_t kKPerLane = Impl::kAK0PerLane * Impl::kAK1PerLane;

    static constexpr auto get_encoding()
    {
        if constexpr(AttrNumAccessV == 0)
        {
            return tile_distribution_encoding<
                sequence<Impl::kRepeat>,
                tuple<sequence<Impl::kAMBlock, Impl::kAMLane>,
                      sequence<Impl::kAK0PerLane, Impl::kABKLane, Impl::kAK1PerLane>>,
                tuple<typename Impl::kABPs2RHssMajor>,
                tuple<typename Impl::kABPs2RHssMinor>,
                typename Impl::kABYs2RHsMajor,
                typename Impl::kABYs2RHsMinor>{};
        }
        else
        {
            constexpr bool UsePackNumAccess =
                (AttrNumAccessV & static_cast<index_t>(WGAttrNumAccessEnum::PackedFlag)) != 0;
            if constexpr(UsePackNumAccess)
            {
                constexpr index_t PackNumAccessV =
                    AttrNumAccessV & ~static_cast<index_t>(WGAttrNumAccessEnum::PackedFlag);
                return tile_distribution_encoding<
                    sequence<Impl::kRepeat>,
                    tuple<sequence<Impl::kAMBlock, Impl::kAMLane>,
                          sequence<Impl::kAK0PerLane,
                                   Impl::kABKLane,
                                   PackNumAccessV,
                                   Impl::kAK1PerLane / PackNumAccessV>>,
                    tuple<typename Impl::kABPs2RHssMajor>,
                    tuple<typename Impl::kABPs2RHssMinor>,
                    sequence<1, 2, 2, 2>,
                    sequence<0, 0, 2, 3>>{};
            }
            else
            {
                return tile_distribution_encoding<
                    sequence<Impl::kRepeat>,
                    tuple<sequence<Impl::kAMBlock, Impl::kAMLane>,
                          sequence<AttrNumAccessV, Impl::kABKLane, kKPerLane / AttrNumAccessV>>,
                    tuple<typename Impl::kABPs2RHssMajor>,
                    tuple<typename Impl::kABPs2RHssMinor>,
                    sequence<1, 2, 2>,
                    sequence<0, 0, 2>>{};
            }
        }
    }

    using type = decltype(get_encoding());
};

template <typename Impl, WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct BWarpDstrEncodingTrait
{
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using BDataType                    = typename Impl::BDataType;
    static constexpr index_t kKPerLane = Impl::kBK0PerLane * Impl::kBK1PerLane;

    static constexpr auto get_encoding()
    {
        if constexpr(AttrNumAccessV == 0)
        {
            return tile_distribution_encoding<
                sequence<Impl::kRepeat>,
                tuple<sequence<Impl::kBNBlock, Impl::kBNLane>,
                      sequence<Impl::kBK0PerLane, Impl::kABKLane, Impl::kBK1PerLane>>,
                tuple<typename Impl::kABPs2RHssMajor>,
                tuple<typename Impl::kABPs2RHssMinor>,
                typename Impl::kABYs2RHsMajor,
                typename Impl::kABYs2RHsMinor>{};
        }
        else
        {
            constexpr bool UsePackNumAccess =
                (AttrNumAccessV & static_cast<index_t>(WGAttrNumAccessEnum::PackedFlag)) != 0;
            if constexpr(UsePackNumAccess)
            {
                constexpr index_t PackNumAccessV =
                    AttrNumAccessV & ~static_cast<index_t>(WGAttrNumAccessEnum::PackedFlag);
                return tile_distribution_encoding<
                    sequence<Impl::kRepeat>,
                    tuple<sequence<Impl::kBNBlock, Impl::kBNLane>,
                          sequence<Impl::kBK0PerLane,
                                   Impl::kABKLane,
                                   PackNumAccessV,
                                   Impl::kBK1PerLane / PackNumAccessV>>,
                    tuple<typename Impl::kABPs2RHssMajor>,
                    tuple<typename Impl::kABPs2RHssMinor>,
                    sequence<1, 2, 2, 2>,
                    sequence<0, 0, 2, 3>>{};
            }
            else
            {
                return tile_distribution_encoding<
                    sequence<Impl::kRepeat>,
                    tuple<sequence<Impl::kBNBlock, Impl::kBNLane>,
                          sequence<AttrNumAccessV, Impl::kABKLane, kKPerLane / AttrNumAccessV>>,
                    tuple<typename Impl::kABPs2RHssMajor>,
                    tuple<typename Impl::kABPs2RHssMinor>,
                    sequence<1, 2, 2>,
                    sequence<0, 0, 2>>{};
            }
        }
    }

    using type = decltype(get_encoding());
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCMBlock, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNBlock, Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename Impl>
struct CTransposedWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNBlock, Impl::kCNLane>,
              sequence<Impl::kCMBlock, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<typename Impl::kCTPs2RHssMajor>,
        tuple<typename Impl::kCTPs2RHssMinor>,
        typename Impl::kCTYs2RHsMajor,
        typename Impl::kCTYs2RHsMinor>;
};

namespace detail {
template <typename T, typename = void>
struct mx_type_enable_or_void
{
    using type = void;
};
template <typename T>
struct mx_type_enable_or_void<T, std::void_t<typename T::MXTypeEnableType>>
{
    using type = typename T::MXTypeEnableType;
};
} // namespace detail

template <typename WarpGemmAttributeWmmaImpl_,
          bool kTransC                       = false,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    // When kTransC is true and A/B types differ, we need an impl with swapped types.
    // Propagate MXTypeEnable (e.g., WmmaScale16Tag) so the transposed impl uses the
    // same WmmaTraits specialization family.
    using TransposedImpl = std::conditional_t<
        kTransC && !std::is_same_v<typename Impl::ADataType, typename Impl::BDataType>,
        WarpGemmAttributeWmmaImpl<
            WmmaTraits<typename Impl::TraitsType::ArchType,
                       typename Impl::BDataType,
                       typename Impl::ADataType,
                       typename Impl::CDataType,
                       Impl::kM,
                       Impl::kN,
                       Impl::kK,
                       typename detail::mx_type_enable_or_void<typename Impl::TraitsType>::type>>,
        Impl>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM      = Impl::kM;
    static constexpr index_t kN      = Impl::kN;
    static constexpr index_t kK      = Impl::kK;
    static constexpr index_t kCMLane = Impl::kCMLane;

    static_assert(Impl::kAK0PerLane * Impl::kAK1PerLane == Impl::kBK0PerLane * Impl::kBK1PerLane);
    static constexpr index_t kKPerThread = Impl::kAK0PerLane * Impl::kAK1PerLane;
    static constexpr index_t kAKPack     = Impl::kAK1PerLane;
    static constexpr index_t kBKPack     = Impl::kBK1PerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl, AttrNumAccessA>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl, AttrNumAccessB>::type;

    // kCM0PerLane = 1, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16
    using CWarpDstrEncoding =
        std::conditional_t<kTransC,
                           typename CTransposedWarpDstrEncodingTrait<Impl>::type,
                           typename CWarpDstrEncodingTrait<Impl>::type>;

    // c_vec += a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(c_vec, b_vec, a_vec);
        }
        else
        {
            Impl{}.template operator()<Params...>(c_vec, a_vec, b_vec);
        }
    }

    // c_vec = a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(b_vec, a_vec);
        }
        else
        {
            return Impl{}.template operator()<Params...>(a_vec, b_vec);
        }
    }

    // c_out = a_vec * b_vec + c_vec : fp32 accumulate, narrowed C output (e.g. bf16)
    template <typename... Params>
    CK_TILE_DEVICE auto
    mac_downconvert(const CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}.template mac_downconvert<Params..., SwapReuse_<true>>(
                c_vec, b_vec, a_vec);
        }
        else
        {
            return Impl{}.template mac_downconvert<Params...>(c_vec, a_vec, b_vec);
        }
    }

    // c_vec += a_vec * b_vec
    template <typename... Params, typename AScaleType, typename BScaleType>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const AScaleType& a_scale,
                                   const BVecType& b_vec,
                                   const BScaleType& b_scale) const
    {
        if constexpr(kTransC)
        {
            TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(
                c_vec, b_vec, b_scale, a_vec, a_scale);
        }
        else
        {
            Impl{}.template operator()<Params...>(c_vec, a_vec, a_scale, b_vec, b_scale);
        }
    }

    // c_vec = a_vec * b_vec
    template <typename... Params, typename AScaleType, typename BScaleType>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const AScaleType& a_scale,
                                       const BVecType& b_vec,
                                       const BScaleType& b_scale) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(
                b_vec, b_scale, a_vec, a_scale);
        }
        else
        {
            return Impl{}.template operator()<Params...>(a_vec, a_scale, b_vec, b_scale);
        }
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          index_t M_Warp_Tile,
          index_t N_Warp_Tile,
          index_t K_Warp_Tile>
CK_TILE_HOST bool check_wmma_supported()
{
    if(is_gfx120_supported())
    {
        return has_wmma_traits_v<gfx120_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx125_supported())
    {
        return has_wmma_traits_v<gfx125_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx11_supported())
    {
        return has_wmma_traits_v<gfx11_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else
    {
        return false;
    }
}

} // namespace ck_tile
