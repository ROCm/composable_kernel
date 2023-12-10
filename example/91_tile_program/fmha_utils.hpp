// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <optional>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#pragma once

enum class Mode : unsigned
{
    Batch,
    Group
};

inline std::ostream& operator<<(std::ostream& stream, Mode mode)
{
    return stream << (mode == Mode::Batch ? "batch" : "group");
}

/// TODO: make sure result is valid for MaskUpperTriangleFromBottomRightPredicate
std::vector<int32_t> generate_seqstarts(Mode mode,
                                        unsigned count,
                                        int32_t seqlens_sum,
                                        std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    const std::vector<int32_t> seqlens = [&]() {
        std::vector<int32_t> original_seqlens(count, seqlens_sum);

        if(mode == Mode::Group && 1 < count)
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
                if(original_seqlens[to_decrease] == 1)
                {
                    continue;
                }

                const size_type to_increase = (to_decrease + next_step()) % count;

                --original_seqlens[to_decrease];
                ++original_seqlens[to_increase];
            }
        }

        return original_seqlens;
    }();

    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    return seqstarts;
}
