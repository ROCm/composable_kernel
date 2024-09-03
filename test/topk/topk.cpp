// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#ifndef TEST_TOPK_VERBOSE
#define TEST_TOPK_VERBOSE 0
#endif

int test_reference_topk()
{
    bool result = true;
    // clang-format off
    {
        auto dump_2d = [](const auto& t_, auto f_){
            size_t row_ = t_.get_length(0);
            size_t col_ = t_.get_length(1);
            for(size_t i_r = 0; i_r < row_; i_r ++) {
                for(size_t i_c = 0; i_c < col_; i_c++) {
                    f_(t_({i_r, i_c}));
                }
                printf("\n");
            }
        };
        constexpr int row = 3;
        constexpr int col = 5;
        constexpr int k = 2;
        ck_tile::HostTensor<float> x({row, col});
        
        x.mData = { -1.6227,  0.3671, -0.8517,  0.6236,  0.1757,
                    -0.7019, -0.4472,  0.4214, -1.8082,  0.8718,
                    -0.4285, -1.6937, -0.5943, -0.3363, -1.2313};
        ck_tile::HostTensor<float> y_values({row, k});
        ck_tile::HostTensor<int> y_indices({row, k});
        ck_tile::reference_topk(x, y_values, y_indices, k);
#if TEST_TOPK_VERBOSE
        dump_2d(y_values, [](auto f_){ printf("%.4f, ", f_);});
        dump_2d(y_indices, [](auto i_){ printf("%2i, ", i_);});
#endif
        std::vector<float> expected_topk_v{ 0.6236,  0.3671,
                                            0.8718,  0.4214,
                                           -0.3363, -0.4285};
        std::vector<int> expected_topk_i{ 3,  1,
                                          4,  2,
                                          3,  0};
        result &= std::equal(y_values.begin(), y_values.end(), expected_topk_v.begin(),
                [](auto x_, auto y_){ return std::fabs(x_ - y_) < 1e-9; });
        result &= std::equal(y_indices.begin(), y_indices.end(), expected_topk_i.begin(),
                [](auto x_, auto y_){ return x_ == y_; });
    }
#if TEST_TOPK_VERBOSE
    printf("-----------------------\n");
#endif
    {
        // TODO: sorted=false in torch seems not correct?
        auto dump_2d = [](const auto& t_, auto f_){
            size_t row_ = t_.get_length(0);
            size_t col_ = t_.get_length(1);
            for(size_t i_r = 0; i_r < row_; i_r ++) {
                for(size_t i_c = 0; i_c < col_; i_c++) {
                    f_(t_({i_r, i_c}));
                }
                printf("\n");
            }
        };
        constexpr int row = 2;
        constexpr int col = 6;
        constexpr int k = 4;
        ck_tile::HostTensor<float> x({row, col});
        
        x.mData = { 0.7693,  0.0300,  0.1465, -0.4806, -0.2512,  0.5678,
                   -0.0956,  0.0404, -0.2719,  1.3804,  0.3790, -0.3885};
        ck_tile::HostTensor<float> y_values({row, k});
        ck_tile::HostTensor<int> y_indices({row, k});
        ck_tile::reference_topk(x, y_values, y_indices, k, -1/*dim*/, true/*largest*/, false/*sorted*/);
#if TEST_TOPK_VERBOSE
        dump_2d(y_values, [](auto f_){ printf("%.4f, ", f_);});
        dump_2d(y_indices, [](auto i_){ printf("%2i, ", i_);});
#endif
        std::vector<float> expected_topk_v{ 0.5678,  0.7693,  0.1465,  0.0300,
                                            0.0404,  0.3790,  1.3804, -0.0956};
        std::vector<int> expected_topk_i{ 5, 0, 2, 1,
                                          1, 4, 3, 0};
        result &= std::equal(y_values.begin(), y_values.end(), expected_topk_v.begin(),
                [](auto x_, auto y_){ return std::fabs(x_ - y_) < 1e-9; });
        result &= std::equal(y_indices.begin(), y_indices.end(), expected_topk_i.begin(),
                [](auto x_, auto y_){ return x_ == y_; });
    }
#if TEST_TOPK_VERBOSE
    printf("-----------------------\n");
#endif
    {
        auto dump_2d = [](const auto& t_, auto f_){
            size_t row_ = t_.get_length(0);
            size_t col_ = t_.get_length(1);
            for(size_t i_r = 0; i_r < row_; i_r ++) {
                for(size_t i_c = 0; i_c < col_; i_c++) {
                    f_(t_({i_r, i_c}));
                }
                printf("\n");
            }
        };
        constexpr int row = 4;
        constexpr int col = 6;
        constexpr int k = 3;
        ck_tile::HostTensor<float> x({row, col});
        
        x.mData = { -0.3015,  0.3252, -1.0818,  0.0655, -1.0700, -0.1597,
                     0.8308,  0.8426, -1.1086,  0.3898, -0.3499, -0.0201,
                    -0.9126, -0.8375,  1.2521,  2.3118, -0.1049, -0.1440,
                    -0.6896, -2.6750, -0.2664,  1.8984, -1.1777, -1.3501};
        ck_tile::HostTensor<float> y_values({row, k});
        ck_tile::HostTensor<int> y_indices({row, k});
        ck_tile::reference_topk(x, y_values, y_indices, k, -1, false);
#if TEST_TOPK_VERBOSE
        dump_2d(y_values, [](auto f_){ printf("%.4f, ", f_);});
        dump_2d(y_indices, [](auto i_){ printf("%2i, ", i_);});
#endif
        std::vector<float> expected_topk_v{ -1.0818, -1.0700, -0.3015,
                                            -1.1086, -0.3499, -0.0201,
                                            -0.9126, -0.8375, -0.1440,
                                            -2.6750, -1.3501, -1.1777};
        std::vector<int> expected_topk_i{ 2, 4, 0,
                                          2, 4, 5,
                                          0, 1, 5,
                                          1, 5, 4};
        result &= std::equal(y_values.begin(), y_values.end(), expected_topk_v.begin(),
                [](auto x_, auto y_){ return std::fabs(x_ - y_) < 1e-9; });
        result &= std::equal(y_indices.begin(), y_indices.end(), expected_topk_i.begin(),
                [](auto x_, auto y_){ return x_ == y_; });
    }
#if TEST_TOPK_VERBOSE
    printf("-----------------------\n");
#endif
    {
        auto dump_3d = [](const auto& t_, auto f_){
            size_t d0_ = t_.get_length(0);
            size_t d1_ = t_.get_length(1);
            size_t d2_ = t_.get_length(2);
            for(size_t i_0 = 0; i_0 < d0_; i_0 ++) {
                for(size_t i_1 = 0; i_1 < d1_; i_1++) {
                    for(size_t i_2 = 0; i_2 < d2_; i_2++) {
                        f_(t_({i_0, i_1, i_2}));
                    }
                    printf("\n");
                }
                printf("\n");
            }
        };
        constexpr int d0 = 3;
        constexpr int d1 = 4;
        constexpr int d2 = 2;
        constexpr int k = 2;
        ck_tile::HostTensor<float> x({d0, d1, d2});
        
        x.mData = { -0.6589,  0.9343,
                     1.1786, -0.0031,
                     0.8447, -0.5745,
                     0.1757,  1.6419,

                     0.8131, -0.9254,
                     0.7139,  0.2138,
                     0.9096,  0.4437,
                    -0.1763, -2.6305,

                    -2.2378,  0.7727,
                    -0.7492,  0.3129,
                    -1.6163, -0.8763,
                    -1.0472,  0.5557};
        ck_tile::HostTensor<float> y_values({d0, k, d2});
        ck_tile::HostTensor<int> y_indices({d0, k, d2});
        ck_tile::reference_topk(x, y_values, y_indices, k, 1);
#if TEST_TOPK_VERBOSE
        dump_3d(y_values, [](auto f_){ printf("%.4f, ", f_);});
        dump_3d(y_indices, [](auto i_){ printf("%2i, ", i_);});
#endif
        std::vector<float> expected_topk_v{ 1.1786,  1.6419,
                                            0.8447,  0.9343,

                                            0.9096,  0.4437,
                                            0.8131,  0.2138,

                                           -0.7492,  0.7727,
                                           -1.0472,  0.5557};
        std::vector<int> expected_topk_i{ 1, 3,
                                          2, 0,

                                          2, 2,
                                          0, 1,

                                          1, 0,
                                          3, 3};
        result &= std::equal(y_values.begin(), y_values.end(), expected_topk_v.begin(),
                [](auto x_, auto y_){ return std::fabs(x_ - y_) < 1e-9; });
        result &= std::equal(y_indices.begin(), y_indices.end(), expected_topk_i.begin(),
                [](auto x_, auto y_){ return x_ == y_; });
    }
    // clang-format on
    return result ? 0 : -1;
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    int rtn = test_reference_topk();

    return rtn;
}
