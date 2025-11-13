#pragma once
#include <stdexcept>

namespace ck_tile {
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

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        constexpr int kABK0PerLane = GemmConfig::K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
    }
    else
    {
        int divisor = 1;
        if(ck_tile::is_gfx11_supported())
        {
            divisor = 1;
        }
        else
        {
            assert(is_wave32() == false);
            divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
        }
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       divisor,
                                       GemmConfig::K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_bq_permuteN(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);

    int n_                = t.get_lengths()[1];
    int bqk_              = t.get_lengths()[0];
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;

    ck_tile::HostTensor<T> t_view(
        {n_ / GemmConfig::N_Tile, GemmConfig::N_Warp, GemmConfig::N_Warp_Tile, NRepeat, bqk_});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 2, 4});
}

template <typename GemmConfig, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_                = t.get_lengths()[1];
    int k_                = t.get_lengths()[0];
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;
    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        constexpr int kABK0PerLane = GemmConfig::K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Tile,
                                       GemmConfig::N_Warp,
                                       GemmConfig::N_Warp_Tile,
                                       NRepeat,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 6, 5, 2, 7});
    }
    else
    {
        int divisor = 1;
        if(ck_tile::is_gfx11_supported())
        {
            divisor = 1;
        }
        else
        {
            assert(is_wave32() == false);
            divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
        }
        ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Tile,
                                       GemmConfig::N_Warp,
                                       GemmConfig::N_Warp_Tile,
                                       NRepeat,
                                       k_ / GemmConfig::K_Warp_Tile,
                                       divisor,
                                       GemmConfig::K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 5, 2, 6});
    }
}
} // namespace ck_tile
