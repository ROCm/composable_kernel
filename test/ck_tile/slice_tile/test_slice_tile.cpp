// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include <type_traits>

#if 0
void test_reverse_slice_sequence()
{
    constexpr auto r = ck_tile::reverse_slice_sequence(ck_tile::sequence<2, 2, 1, 4, 4>{},
                                                       ck_tile::number<2>{},
                                                       ck_tile::sequence<1, 0, 1, 0, 1>{});

    decltype(r){}.qqq();
}
#endif

// clang-format off
template<typename X_Origin_ = ck_tile::sequence<0, 0>, typename Y_Origin_ = ck_tile::sequence<0, 0, 0, 0>>
void test_slice_distribution_from_x_case_0(X_Origin_ = {}, Y_Origin_ = {})
{
    using namespace ck_tile;

    constexpr auto r = detail::slice_distribution_from_x(
        make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                        tuple<sequence<1, 4, 16>, sequence<2, 2, 1, 4, 4>>,
                                        //             Y  P  P             Y  P  Y  P  Y
                                        tuple<sequence<1, 2>, sequence<2, 1>>,
                                        tuple<sequence<1, 1>, sequence<3, 2>>,
                                        sequence<1, 2, 2, 2>,
                                        sequence<0, 0, 2, 4>>{}),
        X_Origin_{},
        sequence<64, 16>{});

    using sliced_dist_enc = remove_cvref_t<decltype(r[number<0>{}].get_static_tile_distribution_encoding())>;
    using target_dist_enc = tile_distribution_encoding<sequence<>,
                                        tuple<sequence<1, 4, 16>, sequence<1, 2, 1, 4, 2>>,
                                        //             Y  P  P             Y  P  Y  P  Y
                                        tuple<sequence<1, 2>, sequence<2, 1>>,
                                        tuple<sequence<1, 1>, sequence<3, 2>>,
                                        sequence<1, 2, 2, 2>,
                                        sequence<0, 0, 2, 4>>;

    static_assert(std::is_same_v<sliced_dist_enc, target_dist_enc>);

    using sliced_y_origins = remove_cvref_t<decltype(r[number<1>{}])>;
    using sliced_y_lengths = remove_cvref_t<decltype(r[number<2>{}])>;
    static_assert(std::is_same_v<sliced_y_origins, Y_Origin_>);
    static_assert(std::is_same_v<sliced_y_lengths, sequence<1, 1, 1, 2>>);

}
// clang-format on

void test_slice_distribution_from_x()
{
    using namespace ck_tile;
    test_slice_distribution_from_x_case_0();
    // test_slice_distribution_from_x_case_0(sequence<0, 16>{}, sequence<0, 0, 0, 2>{});
}

int main()
{
    // test_reverse_slice_sequence();
    test_slice_distribution_from_x();
}
