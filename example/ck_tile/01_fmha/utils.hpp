// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>

#include "ck_tile/core/container/span.hpp"

enum class mode_enum
{
    batch = 0,
    group
};

std::ostream& operator<<(std::ostream& stream, mode_enum mode)
{
    return stream << (mode == mode_enum::batch ? "batch" : "group");
}

std::vector<int32_t> to_seqstarts(ck_tile::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

std::vector<int32_t> generate_seqlens(mode_enum mode,
                                      unsigned count,
                                      int32_t seqlens_sum,
                                      std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    std::vector<int32_t> seqlens(count, seqlens_sum);

    if(mode == mode_enum::group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlens_sum * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens is always greater than 0
            if(seqlens[to_decrease] == 1)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            --seqlens[to_decrease];
            ++seqlens[to_increase];
        }
    }

    return seqlens;
}

std::vector<int32_t> generate_seqstarts(mode_enum mode,
                                        unsigned count,
                                        int32_t seqlens_sum,
                                        std::optional<unsigned> seed = std::nullopt)
{
    return to_seqstarts(generate_seqlens(mode, count, seqlens_sum, seed));
}

int env_get_int(const char* var_name, int default_int)
{
    char* v = getenv(var_name);
    int r   = default_int;
    if(v)
        r = atoi(v);
    return r;
}
