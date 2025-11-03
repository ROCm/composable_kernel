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
    int n_                = t.get_lengths()[1];
    int k_                = t.get_lengths()[0];
    constexpr int divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
    ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                   GemmConfig::N_Warp_Tile,
                                   k_ / GemmConfig::K_Warp_Tile,
                                   divisor,
                                   GemmConfig::K_Warp_Tile / divisor});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
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
    constexpr int divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;

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
} // namespace ck_tile
