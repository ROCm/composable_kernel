// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {
//  pre-defined indexing adaptor used for indexing(scatter/gather)

// this version cache the index inside thread register(which is also prefered in real senario)
// however it's user's responsibility that each thread only provide one indexing, which means
// move coordinate will not change on this dim
template <typename IndexingType>
struct indexing_adaptor_onshot_cached
{

    CK_TILE_HOST_DEVICE constexpr indexing_adaptor_onshot_cached() = default;
    CK_TILE_HOST_DEVICE constexpr indexing_adaptor_onshot_cached(const IndexingType& idx)
        : cached_idx_(idx)
    {
    }
    IndexingType cached_idx_;

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& /*idx_up*/) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = cached_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& /*idx_low*/,
                                                const UpIdx& /*idx_up*/) const
    {
        // TODO: nonthing changed here
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(number<0>{}) = idx_diff_up[number<0>{}];

        // pass the diff to lower, but not changing the actually index
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<IndexingType>::value;
    }
};
#define Using_Gather 1
template <typename IndexingType>
struct indexing_adaptor
{

    CK_TILE_HOST_DEVICE constexpr indexing_adaptor() = default;
    CK_TILE_HOST_DEVICE constexpr indexing_adaptor(const IndexingType* idx) : cached_idx_(idx) {}
    const IndexingType* cached_idx_;
#if Using_Gather
    mutable index_t pre_up_index_  = 0;
    mutable index_t pre_low_index_ = 0;
#endif

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = *(cached_idx_ + idx_up[number<0>{}]);
#if Using_Gather
        pre_up_index_  = idx_up[number<0>{}];
        pre_low_index_ = idx_low(number<0>{});
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            printf("\n first index from  %d to  %d  \n", idx_up[number<0>{}], idx_low(number<0>{}));
        }
#endif
#endif
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& /*idx_low*/,
                                                const UpIdx& /*idx_up*/) const
    {
        // TODO: nonthing changed here
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");
#if !Using_Gather
        idx_diff_low(number<0>{}) = idx_diff_up[number<0>{}];
#else
        int up_index              = idx_diff_up[number<0>{}] + pre_up_index_;
        int low_index             = *(cached_idx_ + up_index);
        idx_diff_low(number<0>{}) = low_index - pre_low_index_;

        pre_up_index_  = up_index;
        pre_low_index_ = low_index;
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            printf("\n index form %d to %d, diff  from  %d to  %d  \n",
                   up_index,
                   low_index,
                   idx_diff_up[number<0>{}],
                   idx_diff_low(number<0>{}));
        }
#endif
#endif

        // pass the diff to lower, but not changing the actually index
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<IndexingType>::value;
    }
};
} // namespace ck_tile
