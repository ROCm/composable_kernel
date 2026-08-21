// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "device_prop.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ck_tile {
namespace detail {
template <typename GemmConfig, typename T, typename = void>
struct b_contiguous_items_per_access
{
    // Storage units per 16 bytes, NOT elements. Deliberately packing-unaware: the
    // consumers that have not opted in derive their device-side granularity the same
    // way, and changing this would desynchronize them. Opt in with
    // BContiguousItemsPerAccess (see items_per_128b_access) when the device side is
    // packing-aware.
    static constexpr int value = 16 / static_cast<int>(sizeof(T));
};

template <typename GemmConfig, typename T>
struct b_contiguous_items_per_access<GemmConfig,
                                     T,
                                     std::void_t<decltype(GemmConfig::BContiguousItemsPerAccess)>>
{
    // PackedSize specified
    static constexpr int value = GemmConfig::BContiguousItemsPerAccess;
};
} // namespace detail

template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
{
    if(t->get_lengths().size() != 2)
    {
        throw std::runtime_error("Host tensor is not rank 2 tensor.");
    }
    int m_   = t->get_lengths()[0];
    int aqk_ = t->get_lengths()[1];

    if(aqk_ % block_aq_k != 0)
    {
        throw std::runtime_error("shuffle_aq needs a aqk of multiple times of block_aq_k.");
    }
    ck_tile::HostTensor<T> t_view({m_, aqk_ / block_aq_k, block_aq_k});
    std::copy(t->begin(), t->end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {1, 0, 2});
}

template <typename T>
auto shuffle_bq(const ck_tile::HostTensor<T>* t, int block_bq_k)
{
    const auto& lengths = t->get_lengths();
    const size_t rank   = lengths.size();

    // Validate block_bq_k divisibility based on rank
    int bqk_dim = (rank == 5) ? lengths[4] : (rank == 2) ? lengths[0] : -1;

    if(bqk_dim < 0)
    {
        throw std::runtime_error("shuffle_bq expects either rank-2 or rank-5 tensor, got rank " +
                                 std::to_string(rank));
    }

    if(bqk_dim % block_bq_k != 0)
    {
        throw std::runtime_error("shuffle_bq needs bqk dimension to be a multiple of block_bq_k.");
    }

    // For TilePermuteN
    if(rank == 5)
    {
        // Handle 5D tensor: [n, nrepeat, nwarp, n_warp_tile, bqk]
        ck_tile::HostTensor<T> t_view({static_cast<int>(lengths[0]),
                                       static_cast<int>(lengths[1]),
                                       static_cast<int>(lengths[2]),
                                       static_cast<int>(lengths[3]),
                                       bqk_dim / block_bq_k,
                                       block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {4, 0, 1, 2, 3, 5});
    }
    else // rank == 2
    {
        // Handle 2D tensor: [bqk, n]
        int n_ = lengths[1];
        ck_tile::HostTensor<T> t_view({n_, bqk_dim / block_bq_k, block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {1, 0, 2});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t, const GemmConfig& gemmConfig)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        int kABK0PerLane           = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
                                       gemmConfig.N_Warp_Tile,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
    }
    else if(ck_tile::is_gfx11_supported())
    {
        int divisor = 1;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
                                       gemmConfig.N_Warp_Tile,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       divisor,
                                       gemmConfig.K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else
    {
        constexpr int KLane = ck_tile::get_warp_size() / GemmConfig::N_Warp_Tile;
        constexpr int ItemsPerAccess =
            std::min(detail::b_contiguous_items_per_access<GemmConfig, T>::value,
                     GemmConfig::K_Warp_Tile / KLane);

        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
                                       gemmConfig.N_Warp_Tile,
                                       k_ / ItemsPerAccess,
                                       ItemsPerAccess});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 1, 3});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b(t, GemmConfig{});
}

template <typename GemmConfig, typename T>
auto bq_permuteN(const ck_tile::HostTensor<T>& t, index_t group_n)
{
    assert(t.get_lengths().size() == 2);

    int n_                = t.get_lengths()[1];
    int bqk_              = t.get_lengths()[0];
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;

    ck_tile::HostTensor<T> t_view({n_ / (GemmConfig::N_Tile / group_n),
                                   GemmConfig::N_Warp,
                                   GemmConfig::N_Warp_Tile / group_n,
                                   NRepeat,
                                   bqk_});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 2, 4});
}

template <typename GemmConfig, index_t BlockedXDLNPerWarp, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t,
                        const GemmConfig& gemmConfig,
                        number<BlockedXDLNPerWarp>)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];
    constexpr ck_tile::index_t NRepeat =
        GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;
    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor = 2;
        int kABK1PerLane = min(16 / static_cast<int>(sizeof(T)), gemmConfig.K_Warp_Tile / divisor);
        int kABK0PerLane = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 5, 6, 2, 7});
    }
    else
    {
        static_assert(NRepeat % BlockedXDLNPerWarp == 0,
                      "wrong! NRepeat must be a multiple of BlockedXDLNPerWarp");
        constexpr int KLane = ck_tile::get_warp_size() / GemmConfig::N_Warp_Tile;
        constexpr int ItemsPerAccess =
            std::min(detail::b_contiguous_items_per_access<GemmConfig, T>::value,
                     GemmConfig::K_Warp_Tile / KLane);
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat / BlockedXDLNPerWarp,
                                       BlockedXDLNPerWarp,
                                       k_ / ItemsPerAccess,
                                       ItemsPerAccess});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 5, 2, 6});
    }
}

template <typename GemmConfig, typename T, index_t BlockedXDLNPerWarp = 1>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b_permuteN(t, GemmConfig{}, number<BlockedXDLNPerWarp>{});
}

namespace detail {
// The shuffled views below are built with packed strides and truncating divisions, so they
// can be smaller than the source: when a dimension is not a multiple of its tile, or when
// the source carries leading-dim stride padding. The std::copy that follows writes the
// whole source, so a smaller destination is a heap overflow.
template <typename T>
void check_shuffle_view_fits(const ck_tile::HostTensor<T>& src,
                             const ck_tile::HostTensor<T>& view,
                             const char* who)
{
    if(view.size() < src.size())
    {
        throw std::runtime_error(std::string(who) +
                                 ": destination smaller than source; every dimension must be a "
                                 "multiple of its tile and B must be unpadded.");
    }
}
} // namespace detail

template <typename FlatmmConfig, typename T>
auto shuffle_b_v0(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if(ck_tile::is_gfx11_supported())
    {
        int divisor = 1;
        ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Warp_Tile,
                                       FlatmmConfig::N_Warp_Tile,
                                       k_ / FlatmmConfig::K_Warp_Tile,
                                       divisor,
                                       FlatmmConfig::K_Warp_Tile / divisor});
        detail::check_shuffle_view_fits(t, t_view, "shuffle_b_v0");
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else
    {
        // A config may override the granularity via BContiguousItemsPerAccess; the value
        // must match the device-side B granularity for the consuming pipeline.
        constexpr int MaxVecSize = detail::b_contiguous_items_per_access<FlatmmConfig, T>::value;
        // because ck_tile::get_warp_size returns 64 in host side
        int KLane =
            (ck_tile::is_wave32() ? (ck_tile::get_warp_size() / 2) : (ck_tile::get_warp_size())) /
            FlatmmConfig::N_Warp_Tile;
        int ItemsPerAccess = std::min(MaxVecSize, FlatmmConfig::K_Warp_Tile / KLane);

        ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Warp_Tile,
                                       FlatmmConfig::N_Warp_Tile,
                                       k_ / ItemsPerAccess,
                                       ItemsPerAccess});
        detail::check_shuffle_view_fits(t, t_view, "shuffle_b_v0");
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 1, 3});
    }
}

template <typename FlatmmConfig, typename T>
auto shuffle_b_v1(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    constexpr int MaxVecSize     = 16 / sizeof(T);
    constexpr int KLane          = ck_tile::get_warp_size() / FlatmmConfig::N_Warp_Tile;
    constexpr int ItemsPerAccess = std::min(MaxVecSize, FlatmmConfig::K_Warp_Tile / KLane);
    constexpr int NRepeat = FlatmmConfig::N_Tile / FlatmmConfig::N_Warp_Tile / FlatmmConfig::N_Warp;

    ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Tile,
                                   FlatmmConfig::N_Warp,
                                   FlatmmConfig::N_Warp_Tile,
                                   NRepeat,
                                   k_ / ItemsPerAccess,
                                   ItemsPerAccess});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 2, 5});
}

} // namespace ck_tile
