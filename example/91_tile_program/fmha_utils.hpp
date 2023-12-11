// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cassert>
#include <functional>
#include <optional>
#include <ostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "ck/utility/span.hpp"

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

inline std::vector<int32_t> to_seqstarts(ck::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

inline std::vector<int32_t> generate_seqlens_q(Mode mode,
                                               unsigned count,
                                               int32_t seqlens_q_sum,
                                               std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    std::vector<int32_t> seqlens_q(count, seqlens_q_sum);

    if(mode == Mode::Group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlens_q_sum * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens_q is always greater than 0
            if(seqlens_q[to_decrease] == 1)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            --seqlens_q[to_decrease];
            ++seqlens_q[to_increase];
        }
    }

    return seqlens_q;
}

inline std::tuple<std::vector<int32_t>, std::vector<int32_t>> generate_seqlens_seqstarts_q(
    Mode mode, unsigned count, int32_t seqlens_q_sum, std::optional<unsigned> seed = std::nullopt)
{
    const std::vector<int32_t> seqlens_q = generate_seqlens_q(mode, count, seqlens_q_sum, seed);
    return std::make_tuple(seqlens_q, to_seqstarts(seqlens_q));
}

inline std::vector<int32_t> generate_seqlens_k(Mode mode,
                                               unsigned count,
                                               int32_t seqlens_k_sum,
                                               ck::span<const int32_t> seqlens_q,
                                               int32_t seqlens_q_sum,
                                               std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);
    assert(seqlens_q.size() == count);

    std::vector<int32_t> seqlens_k(count, seqlens_k_sum);

    if(mode == Mode::Group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlens_k_sum * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens_k is always greater than 0 & greater than
            // corresponding elements in seqlens_q
            if(seqlens_k[to_decrease] == 1 ||
               (seqlens_q_sum < seqlens_k_sum &&
                seqlens_k[to_decrease] <= seqlens_q[to_decrease] + 1))
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            --seqlens_k[to_decrease];
            ++seqlens_k[to_increase];
        }
    }

    return seqlens_k;
}

inline std::vector<int32_t> generate_seqstarts_k(Mode mode,
                                                 unsigned count,
                                                 int32_t seqlens_k_sum,
                                                 ck::span<const int32_t> seqlens_q,
                                                 int32_t seqlens_q_sum,
                                                 std::optional<unsigned> seed = std::nullopt)
{
    return to_seqstarts(
        generate_seqlens_k(mode, count, seqlens_k_sum, seqlens_q, seqlens_q_sum, seed));
}
