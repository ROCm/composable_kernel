// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <tuple>

template <typename DataType>
std::tuple<ck_tile::HostTensor<DataType>, ck_tile::HostTensor<DataType>>
generate_cos_sin(ck_tile::index_t seqlen,
                 ck_tile::index_t rotary_dim,
                 std::optional<unsigned> seed = std::nullopt)
{
    std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_real_distribution<float> generator(0.0f, 1.0f);

    const ck_tile::index_t num_rows = seqlen * 2;
    const ck_tile::index_t num_cols = rotary_dim / 2;

    using std::begin, std::end;

    ck_tile::HostTensor<float> angle({num_rows, num_cols});
    std::generate(begin(angle), end(angle), std::bind(generator, std::ref(random_engine)));

    ck_tile::HostTensor<DataType> cos({num_rows, num_cols});
    std::transform(begin(angle), end(angle), [](float origin_value) {
        return ck_tile::type_convert<DataType>(std::cos(origin_value));
    });

    ck_tile::HostTensor<DataType> sin({num_rows, num_cols});
    std::transform(begin(angle), end(angle), [](float origin_value) {
        return ck_tile::type_convert<DataType>(std::sin(origin_value));
    });

    return std::make_tuple(cos, sin);
}

ck_tile::index_t generate_seqlen_offset(ck_tile::index_t seqlen,
                                        std::optional<unsigned> seed = std::nullopt)
{
    std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
    return std::uniform_int_distribution<ck_tile::index_t>{0, seqlen}(random_engine);
}

template <typename DataType>
std::tuple<ck_tile::HostTensor<DataType>, ck_tile::HostTensor<DataType>>
index_cos_sin(const ck_tile::HostTensor<DataType>& cos,
              const ck_tile::HostTensor<DataType>& sin,
              ck_tile::index_t seqlen_offset,
              ck_tile::index_t seqlen)
{
    assert(cos.get_num_of_dimension() == 2 && sin.get_num_of_dimension() == 2);
    assert(cos.get_length(0) == sin.get_length(0) && cos.get_length(1) == sin.get_length(1));

    assert(seqlen_offset + seqlen <= cos.get_length(0));

    const ck_tile::index_t num_rows = seqlen;
    const ck_tile::index_t num_cols = cos.get_length(1);

    ck_tile::HostTensor<DataType> cos_pt({num_rows, num_cols});
    cos_pt.ForEach([&](auto& self, auto i) { self(i) = cos(i[0] + seqlen_offset, i[1]); });

    ck_tile::HostTensor<DataType> sin_pt({num_rows, num_cols});
    sin_pt.ForEach([&](auto& self, auto i) { self(i) = sin(i[0] + seqlen_offset, i[1]); });

    return std::make_tuple(cos_pt, sin_pt);
}
