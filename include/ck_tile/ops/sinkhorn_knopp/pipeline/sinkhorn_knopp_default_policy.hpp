#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"

namespace ck_tile {

struct SinkhornKnoppDefaultPolicy : public Reduce2dDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTransposedXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::ThreadTile_N>,
                    sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::ThreadTile_M>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }
};

} // namespace ck_tile